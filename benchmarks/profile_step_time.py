"""V3 checkpoint CLI: controlled collection, P95 fit, independent evaluation."""
import argparse
import json
from pathlib import Path

from benchmarks.defaults import add_model_arguments, resolve_model_arguments
from benchmarks.v3_profile.protocol import Settings, load_profile, planned_cases


def integers(value):
    try:
        result = tuple(int(v) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("empty list")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    add_model_arguments(collect_parser)
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--split", choices=("train", "eval", "smoke"), required=True)
    collect_parser.add_argument("--seed", type=int, required=True)
    collect_parser.add_argument("--smoke", action="store_true")
    for name in ("decode_counts", "prefill_budgets", "contexts", "prefill_request_counts"):
        collect_parser.add_argument("--"+name.replace("_", "-"), type=integers,
                                    default=getattr(Settings(), name))
    for name in ("repeats", "warmup", "max_model_len", "token_budget", "max_num_seqs", "kv_block_cap"):
        collect_parser.add_argument("--"+name.replace("_", "-"), type=int,
                                    default=getattr(Settings(), name))
    collect_parser.add_argument("--memory", type=float, default=.8)
    collect_parser.add_argument("--prefill-shape", choices=("long", "exact"), default="long")
    collect_parser.add_argument("--alternate-context-tokens", type=int, default=0,
                                help="alternate Decode prompts between --contexts and this length")
    collect_parser.add_argument("--alternate-context-count", type=int, default=0,
                                help="use this many alternate-length Decode prompts first; 0 keeps alternating")
    collect_parser.add_argument("--plan-only", action="store_true", help="validate/print matrix without CUDA")
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--profiles", nargs="+", required=True)
    fit_parser.add_argument("--out", required=True)
    fit_parser.add_argument("--min-samples", type=int, default=30)
    fit_parser.add_argument("--decode-count-width", type=int, default=1)
    fit_parser.add_argument("--sample-count-width", type=int, default=1)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--profiles", nargs="+", required=True)
    eval_parser.add_argument("--predictor", required=True)
    eval_parser.add_argument("--out", required=True)
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--profiles", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        if args.seed < 0:
            parser.error("seed must be nonnegative")
        if args.smoke != (args.split == "smoke"):
            parser.error("--smoke must be paired with --split smoke")
        resolve_model_arguments(args, parser)
        values = {name: getattr(args, name) for name in Settings.__dataclass_fields__}
        if args.smoke:
            values.update(decode_counts=(0, 1, 4), prefill_budgets=(0, 256),
                          contexts=(256,),
                          prefill_request_counts=(1, 2) if args.prefill_shape == "exact" else (1,),
                          repeats=2, warmup=1)
        settings = Settings(**values)
        settings.validate()
        if args.plan_only:
            print(json.dumps({"settings": values, "cases": planned_cases(settings)}, indent=2))
            return
        from benchmarks.v3_profile.collect import collect
        result = {"profile": str(collect(args, settings))}
    elif args.command == "fit":
        from benchmarks.v3_profile.report import fit
        if args.decode_count_width < 1 or args.sample_count_width < 1:
            parser.error("bucket widths must be positive")
        widths = {name: width for name, width in (
            ("decode_count", args.decode_count_width),
            ("sample_count", args.sample_count_width)) if width > 1}
        result = fit(args.profiles, args.out, args.min_samples, widths)
    elif args.command == "evaluate":
        from benchmarks.v3_profile.report import evaluate
        result = evaluate(args.predictor, args.profiles, args.out)
    else:
        result = []
        for path in args.profiles:
            protocol, rows = load_profile(path)
            measured = [r for r in rows if not r["warmup"]]
            phases = {phase: 0 for phase in ("decode", "prefill", "mixed")}
            for row in measured:
                f = row["features"]
                phase = "mixed" if f["decode_count"] and f["prefill_tokens"] else "decode" if not f["prefill_tokens"] else "prefill"
                phases[phase] += 1
            result.append({"profile": str(Path(path).resolve()), "split": protocol["split"],
                "seed": protocol["seed"], "smoke_only": protocol["smoke_only"],
                "measured_steps": len(measured), "phases": phases,
                "has_intermediate_prefill": any(r["features"]["prefill_tokens"] and r["features"]["sample_count"] == r["features"]["decode_count"] for r in measured),
                "has_prefill_history": any(r["features"]["prefill_has_history"] for r in measured)})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
