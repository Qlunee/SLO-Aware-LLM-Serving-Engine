"""Archive/verify V1 audit rules; no GPU imports or acceptance exemptions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("nanovllm/**/*.py", "benchmarks/**/*.py", "tests/*.py",
            "scripts/**/*.sh", "scripts/**/*.py", "configs/**/*.json")
EVIDENCE_DIRS = (
    "v1-calibrate.5xsksi5u", "v1-quality.jp5c_dbl",
    "v1-first-divergence.005ytr8a", "v1-layer-replay.s1vz25s6",
    "v1-layer-replay.kwrvxdu6", "v1-layer-replay.qv99jajm",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sources(root):
    files = sorted({p for pattern in PATTERNS for p in root.glob(pattern) if p.is_file()})
    return {str(p.relative_to(root)): sha(p) for p in files}


def check(base, entries):
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("empty or invalid hash manifest")
    base = base.resolve()
    for name, expected in entries.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe manifest path: {name}")
        path = (base/name).resolve()
        if not path.is_relative_to(base) or not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"文件变化或缺失：{path}")


def policy(root):
    return {
        "protocol_id": "v1-acceptance-audit-v1.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": load(root/"configs/qwen3_4b.json"),
        "scope": "BF16 Mixed Prefill/Decode执行与状态正确性",
        "evidence_note": "早期手动first-divergence目录没有实验时manifest；直接归档report.json与step11.pt当前哈希，不声称恢复了实验时完整性证明",
        "tracks": {
            "execution_semantics": {
                "required": [
                    "CPU状态测试通过",
                    "中间Chunk不采样，每请求每Step最多输出一个Token",
                    "无输出重复或遗漏、无跨请求KV污染、无页引用泄漏",
                    "有效KV与独立记录的投影K/V一致",
                    "真实Mixed Step且每Step仅一次完整模型前向",
                    "Prefix、跨页、抢占重算、页复用检查通过",
                    "在线日志与结果审计通过",
                ],
                "fp32_diagnostic": {
                    "top1_match_required": True, "relative_l2_max": .001, "max_abs_error_max": .02,
                },
                "attention_reference": {
                    "reference": "独立FP32因果GQA Attention", "atol": .02, "rtol": .02,
                    "note": "沿用现有Attention探针容差，不替代端到端Logits验收",
                },
            },
            "legacy_numerical_compatibility": {
                "checker": "benchmarks.v1.check_v1", "top1_match_required": True,
                "strict_allclose": {"atol": .15, "rtol": .03},
                "bf16_policy": {
                    "relative_l2_max": .02, "cosine_min": .999, "max_abs_error_max": .5,
                    "rule": "Top-1一致且误差有限，满足allclose或同时满足以上三项",
                },
                "existing_failures_remain_failures": True,
                "near_tie_is_not_automatic_pass": True,
            },
            "generation_quality": {
                "report": ["完整输出一致率及首次分叉位置",
                           "相同历史下的Logits、Top-2间隔、KL/JS和目标NLL",
                           "每个已分析分叉的状态、KV与Attention证据"],
                "rules": ["历史分叉后不直接比较Logits", "人工sanity集不代表真实任务质量",
                          "不宣称任意输入100%一致或质量无损",
                          "本轮不根据已观察结果拟合新的质量通过阈值"],
            },
        },
        "required_matrix": {
            "batch": [1, 2, 8, 32], "chunk": [256, 512, 1024, 2048],
            "tp": [1, 2], "graph": [False, True], "repeats": 3,
            "cases": ["mixed", "prefix", "preempt"],
            "graph_scope": "Mixed走Eager；纯Decode验证Graph回放",
        },
        "reporting": {
            "separate_track_statuses": True, "freeze_does_not_grant_acceptance": True,
            "full_pass_blockers": ["必测项未完成", "状态、KV或Attention检查失败",
                                   "原端到端数值验收仍失败", "源码变化或证据校验失败"],
            "revision_rule": "修改范围或阈值须另建协议版本并说明理由，用新实验验证；保留旧规则和旧结果",
        },
    }


def collect_evidence(root):
    evidence = {}
    for name in EVIDENCE_DIRS:
        folder = root/"results/raw"/name
        if name == "v1-first-divergence.005ytr8a" and not (folder/"artifact_hashes.json").exists():
            # The original inline command never wrote a manifest. Preserve that
            # provenance: pin its two existing files now, do not fabricate one.
            for filename in ("report.json", "step11.pt"):
                path = folder/filename
                evidence[str(path.relative_to(root))] = sha(path)
            continue
        # Validate the actual evidence, not just the manifest file itself.
        check(folder, load(folder/"artifact_hashes.json"))
        for filename in ("report.json", "artifact_hashes.json"):
            path = folder/filename
            evidence[str(path.relative_to(root))] = sha(path)
    return evidence


def freeze(root):
    rules = policy(root)
    before = sources(root)
    evidence = collect_evidence(root)
    raw = root/"results/raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="v1-protocol.", dir=raw))
    save(out/"policy.json", rules)
    save(out/"sources.json", before)
    save(out/"evidence.json", evidence)
    if before != sources(root):
        raise RuntimeError(f"冻结期间源码变化，未完成协议：{out}")
    check(root, evidence)
    # Manifest is written last. An empty/partial directory is never completed.
    save(out/"manifest.json", {name: sha(out/name)
         for name in ("policy.json", "sources.json", "evidence.json")})
    return out


def verify(root, folder):
    check(folder, load(folder/"manifest.json"))
    frozen = load(folder/"sources.json")
    check(root, frozen)
    if frozen != sources(root):
        raise RuntimeError("源码文件集合发生变化（新增、删除或修改）")
    evidence = load(folder/"evidence.json")
    check(root, evidence)
    for name in evidence:
        if Path(name).name == "artifact_hashes.json":
            manifest = root/name
            check(manifest.parent, load(manifest))


def latest(root):
    candidates = [p for p in (root/"results/raw").glob("v1-protocol.*")
                  if (p/"manifest.json").is_file()]
    if not candidates:
        raise RuntimeError("没有已完成的冻结协议，请先执行freeze")
    return max(candidates, key=lambda p: (p/"manifest.json").stat().st_mtime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("freeze")
    verify_parser = subs.add_parser("verify")
    target = verify_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dir", type=Path)
    target.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        folder = freeze(ROOT)
        verify(ROOT, folder)
        print("规则已冻结并校验；不代表V1验收通过。")
    else:
        folder = latest(ROOT) if args.latest else args.dir.resolve()
        verify(ROOT, folder)
        print("冻结规则、源码集合及已有证据校验通过。")
    print(f"协议目录：{folder}")
    print(f"协议SHA256：{sha(folder/'policy.json')}")


if __name__ == "__main__":
    main()
