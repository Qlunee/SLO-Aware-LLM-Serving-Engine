"""统一模型基准；仅标准库，不加载GPU、模型或下载权重。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "qwen3_4b.json"
MODEL_PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
DEFAULT_MODEL_PATH = MODEL_PROFILE["model_path"]
DEFAULT_MODEL_REVISION = MODEL_PROFILE["model_revision"]


def model_path_default():
    return os.environ.get("V0_MODEL") or DEFAULT_MODEL_PATH


def revision_for(model_path, explicit_revision=None):
    """自定义模型位置必须有revision，不能误标成默认4B版本。"""
    revision = explicit_revision or os.environ.get("V0_MODEL_REVISION")
    if revision:
        if not revision.strip():
            raise ValueError("model-revision不能为空白")
        return revision
    if Path(model_path).expanduser().resolve() == Path(DEFAULT_MODEL_PATH).resolve():
        return DEFAULT_MODEL_REVISION
    raise ValueError("自定义--model/V0_MODEL时请同时设置--model-revision或V0_MODEL_REVISION")


def add_model_arguments(parser):
    parser.add_argument("--model", default=model_path_default(),
                        help="本地模型目录；默认V0_MODEL或configs/qwen3_4b.json中的Qwen3-4B")
    parser.add_argument("--model-revision", default=None,
                        help="默认V0_MODEL_REVISION或冻结4B版本；其他模型必须显式指定")


def resolve_model_arguments(args, parser):
    try:
        args.model = str(Path(args.model).expanduser().resolve())
        args.model_revision = revision_for(args.model, args.model_revision)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--field", choices=list(MODEL_PROFILE))
    choice.add_argument("--revision-for")
    args = parser.parse_args()
    if args.field:
        print(MODEL_PROFILE[args.field])
    elif args.revision_for:
        try:
            print(revision_for(args.revision_for))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        print(json.dumps(MODEL_PROFILE, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
