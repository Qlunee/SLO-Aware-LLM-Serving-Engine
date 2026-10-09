## 分别记录项目HEAD和上游SHA，并保存源码摘要。
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

UPSTREAM_SHA = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"error": str(exc)}


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(project_root: Path) -> dict:
    return {str(p.relative_to(project_root)): sha256_file(p)
            for p in sorted((project_root / "nanovllm").rglob("*.py"))}


def model_metadata_hashes(model_path: Path) -> dict:
    names = ["config.json", "tokenizer_config.json", "tokenizer.json",
             "generation_config.json", "model.safetensors.index.json"]
    return {name: sha256_file(model_path / name) for name in names
            if (model_path / name).is_file()}


def collect_environment(
    project_root: Path,
    model: str,
    model_revision: str,
) -> dict:
    import torch
    model_path = Path(model).resolve()

    gpus = []
    p2p = {}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index,
                "name": properties.name,
                "compute_capability": [
                    properties.major, properties.minor
                ],
                "total_memory_bytes": properties.total_memory,
            })

        for source in range(torch.cuda.device_count()):
            for target in range(torch.cuda.device_count()):
                if source != target:
                    p2p[f"{source}->{target}"] = (
                        torch.cuda.can_device_access_peer(source, target)
                    )

    try:
        nccl_version = torch.cuda.nccl.version()
    except Exception:
        nccl_version = None

    weights = [
        {"file": path.name, "bytes": path.stat().st_size}
        for path in sorted(model_path.glob("*.safetensors"))
    ]

    return {
        "schema_version": 1,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "upstream_baseline_sha": UPSTREAM_SHA,
        "project_head": command_output([
            "git", "-C", str(project_root), "rev-parse", "HEAD"
        ]),
        "project_status": command_output([
            "git", "-C", str(project_root), "status", "--porcelain"
        ]),
        "source_sha256": source_hashes(project_root),
        "benchmark_source_sha256": {
            str(path.relative_to(project_root)): sha256_file(path)
            for path in sorted([
                *(project_root / "benchmarks" / "v0").glob("*.py"),
                project_root / "benchmarks" / "defaults.py",
                project_root / "configs" / "qwen3_4b.json",
            ]) if path.is_file()
        },
        "model_path": str(model_path),
        "model_revision_declared": model_revision,
        "model_metadata_sha256": model_metadata_hashes(model_path),
        "weight_manifest": weights,
        "packages": {
            name: package_version(name)
            for name in [
                "torch", "triton", "transformers",
                "flash-attn", "numpy", "xxhash", "safetensors"
            ]
        },
        "torch_cuda_version": torch.version.cuda,
        "nccl_version": nccl_version,
        "gpus": gpus,
        # 表示可访问能力，不表示NCCL一定选用了该链路。
        "p2p_capability": p2p,
        "selected_environment": {
            key: os.environ.get(key)
            for key in ["CUDA_VISIBLE_DEVICES", "NCCL_DEBUG", "NCCL_IB_DISABLE",
                        "NCCL_P2P_DISABLE", "NCCL_NET", "NCCL_SOCKET_IFNAME"]
        },
        "nvidia_smi": command_output(["nvidia-smi"]),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "nvlink": command_output([
            "nvidia-smi", "nvlink", "--status"
        ]),
    }


def save_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    )
