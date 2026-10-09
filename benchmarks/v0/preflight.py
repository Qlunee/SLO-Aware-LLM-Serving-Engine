"""Read-only dependency/model checks before model or KV allocation."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from benchmarks.defaults import model_path_default


def check_environment(model=None, tp=1):
    result = {"python": sys.executable, "python_version": sys.version.split()[0],
              "packages": {}, "errors": [], "gpus": []}
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        result["errors"].append("nano-vLLM要求Python>=3.10,<3.13")
    for name in ["torch", "triton", "transformers", "flash_attn", "xxhash", "safetensors", "tqdm"]:
        try:
            module = importlib.import_module(name)
            result["packages"][name] = getattr(module, "__version__", "installed")
            if name == "flash_attn":
                for symbol in ["flash_attn_varlen_func", "flash_attn_with_kvcache"]:
                    if not callable(getattr(module, symbol, None)):
                        raise ImportError(f"flash_attn缺少{symbol}")
        except Exception as exc:
            result["packages"][name] = None
            result["errors"].append(f"{name}: {exc}")
    if result["packages"].get("torch"):
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        if not torch.cuda.is_available():
            result["errors"].append("CUDA不可用")
        else:
            result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.device_count() < tp:
                result["errors"].append("可见GPU数小于TP")
    if model:
        root = Path(model).resolve()
        result["model"] = str(root)
        if not (root / "config.json").is_file() or not list(root.glob("*.safetensors")):
            result["errors"].append("模型缺少config.json或safetensors权重")
        elif result["packages"].get("transformers"):
            try:
                from transformers import AutoConfig, AutoTokenizer
                config = AutoConfig.from_pretrained(root, local_files_only=True)
                AutoTokenizer.from_pretrained(root, local_files_only=True)
                result["model_config"] = {
                    key: getattr(config, key, None) for key in [
                        "model_type", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "head_dim", "hidden_size", "max_position_embeddings",
                    ]
                }
                result["model_config"]["dtype"] = str(getattr(config, "dtype", None))
                if getattr(config, "dtype", None) is None:
                    result["errors"].append("冻结上游读取hf_config.dtype；当前transformers过旧或模型未声明dtype")
                if config.model_type != "qwen3":
                    result["errors"].append("冻结上游实现的是Qwen3模型")
                if config.num_key_value_heads % tp or config.num_attention_heads % tp:
                    result["errors"].append("Attention Heads不能被TP整除")
            except Exception as exc:
                result["errors"].append(f"模型加载检查: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=model_path_default())
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    if args.tp < 1:
        parser.error("tp must be positive")
    result = check_environment(args.model, args.tp)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
