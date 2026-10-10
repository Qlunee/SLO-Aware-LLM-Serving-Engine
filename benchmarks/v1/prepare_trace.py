"""Create a new language-fixture trace, preserving arrivals/lengths/SLOs; never overwrite."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from benchmarks.defaults import add_model_arguments, resolve_model_arguments
from benchmarks.v0.environment import sha256_file
from benchmarks.v0.workload import load_trace, trace_sha256, validate_trace
from benchmarks.v1.fixtures import language_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    args = resolve_model_arguments(parser.parse_args(), parser)
    meta, specs = load_trace(args.trace)
    if meta["model_revision"] != args.model_revision:
        raise ValueError("model revision mismatch")
    if meta.get("model_config_sha256") != sha256_file(Path(args.model) / "config.json"):
        raise ValueError("model configuration mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    specs = [replace(s, prompt_token_ids=language_prompt(tokenizer, len(s.prompt_token_ids), i + 1))
             for i, s in enumerate(specs)]
    meta = {**meta, "synthetic_tokens": True, "prompt_kind": "repeated_language_fixture",
            "source_trace_sha256": trace_sha256(args.trace)}
    validate_trace(meta, specs)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        for row in [meta, *map(asdict, specs)]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"trace": str(target), "sha256": trace_sha256(target), "requests": len(specs)}))


if __name__ == "__main__":
    main()
