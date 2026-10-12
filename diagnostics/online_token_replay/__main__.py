"""Independent online capture/replay CLI, intentionally outside source freezes."""
import argparse
import math


def positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "capture"):
        p = sub.add_parser(name)
        p.add_argument("--suite", required=True)
        p.add_argument("--mode", choices=("poisson", "burst", "long_short"), default="poisson")
        p.add_argument("--target-ids", nargs="+", default=["r00000014", "r00000018"])
        p.add_argument("--output-token", type=positive_int, default=23, help="1-based output index")
        p.add_argument("--out", required=True)
        if name == "prepare":
            p.add_argument("--repeats", type=positive_int, default=3)
        else:
            p.add_argument("--policy", choices=("baseline", "chunk-512"), required=True)
            p.add_argument("--max-snapshot-mib", type=float, default=512)
    p = sub.add_parser("replay")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=positive_int, default=3)
    p = sub.add_parser("report")
    p.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        if not math.isfinite(args.max_snapshot_mib) or args.max_snapshot_mib <= 0:
            parser.error("snapshot limit must be positive and finite")
        from .capture import capture
        capture(args)
    elif args.command == "replay":
        if args.repeats < 2:
            parser.error("repeat-stability diagnostics need at least 2 forwards")
        from .replay import replay
        replay(args)
    else:
        from .report import prepare, report
        {"prepare": prepare, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
