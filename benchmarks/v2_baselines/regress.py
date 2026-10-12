"""New fixed-policy GPU state scenarios. Diagnostic-only, not numerical waiver."""
import argparse
from dataclasses import asdict
from pathlib import Path
import traceback

from benchmarks.v2_stall.protocol import read, save, seal, sha
from .protocol import load, POLICIES, Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--policy", choices=POLICIES[1:], required=True)
    parser.add_argument("--case", choices=("mixed_prefix", "preempt"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    suite, out = Path(args.suite).resolve(), Path(args.out).resolve()
    protocol = load(suite)
    out.mkdir(parents=True, exist_ok=False)
    save(out/"identity.json", {"protocol_sha256": sha(suite/"protocol.json"),
         "policy": args.policy, "case": args.case, "diagnostic_only": True})
    engine = None
    try:
        import torch
        from benchmarks.v0.workload import RequestSpec
        from benchmarks.v0.metrics import RequestRecord
        from benchmarks.v2_stall.run import check_state
        from .run import make_engine
        if torch.cuda.device_count() != 1:
            raise ValueError("expose one GPU")
        torch.manual_seed(protocol["settings"]["seed"])
        engine, config = make_engine(protocol, args.policy, kv_block_cap=2 if args.case == "preempt" else 32)
        engine.start_clock()
        clock = [0]
        # Deterministic control time exercises wait protection without sleeping.
        # Output timestamps remain real, but neither clock is a performance result.
        engine.scheduler.clock = lambda: clock[0]
        records = []
        prefix_hit = mixed_seen = forced_seen = False
        count = 0

        def add(tokens):
            name = f"test-{len(records)}"
            spec = RequestSpec(name, 0, tokens, 8, 1000, 100)
            record = RequestRecord(name, 0, len(tokens), 8, 1000, 100)
            engine.submit(spec, record)
            records.append(record)

        def step():
            nonlocal prefix_hit, mixed_seen, forced_seen, count
            count += 1
            if count > 1000:
                raise AssertionError("state regression exceeded step limit")
            clock[0] += round(Settings(**protocol["settings"]).max_prefill_wait_ms*1e6)
            before = {i: len(r.output_token_ids) for i, r in engine.by_seq.items()}
            engine.step()
            check_state(engine)
            for i, r in engine.by_seq.items():
                if len(r.output_token_ids)-before.get(i, 0) not in (0, 1):
                    raise AssertionError("duplicate sampling")
            row = engine.steps[-1]
            mixed_seen |= row["num_decode"] > 0 and row["prefill_tokens"] > 0
            forced_seen |= row["forced_prefill_seq_id"] is not None
            prefix_hit |= any(s.num_cached_tokens >= 512 and len(s.block_table) >= 3
                              for s in engine.scheduler.waiting)

        def drain():
            while not engine.is_finished():
                step()

        if args.case == "preempt":
            add([100]*256)
            add([200]*256)
            drain()
            if not engine.scheduler.preemption_count:
                raise AssertionError("preemption scenario did not preempt")
        else:
            add([101]*16)
            step()
            add([201]*513)
            drain()
            if args.policy != "decode_first" and not mixed_seen:
                raise AssertionError("mixed path not exercised")
            if args.policy == "decode_first_wait" and not forced_seen:
                raise AssertionError("wait protection not exercised")
            add([301]*512+[302])
            drain()
            add([301]*512+[303])
            # Check Prefix hit at allocation, before postprocess commits progress.
            real_schedule = engine.scheduler.schedule
            starts = []
            def capture():
                batch = real_schedule()
                starts.extend(r.start_pos for r in batch.requests if r.phase != "decode")
                return batch
            engine.scheduler.schedule = capture
            drain()
            if not starts or starts[0] != 512:
                raise AssertionError("hot Prefix was not reused")
            prefix_hit = True
        if any(r.status != "completed" or len(r.output_token_ids) != 8 for r in records):
            raise AssertionError("request lost or incorrect output length")
        if engine.scheduler.block_manager.used_block_ids:
            raise AssertionError("page leak")
        load(suite)
        save(out/"requests.json", [asdict(r) for r in records])
        save(out/"steps.json", engine.steps)
        save(out/"report.json", {"state_passed": True, "policy": args.policy, "case": args.case,
            "mixed_seen": mixed_seen, "prefix_hit": prefix_hit, "forced_wait_seen": forced_seen,
            "preemptions": engine.scheduler.preemption_count,
            "v1_acceptance_passed": False, "full_model_numerics_tested": False,
            "scope": "per-Step state/metadata and fixed output counts; synthetic control clock; NOT performance or full-model numerical acceptance"})
        print("GPU state regression passed:", out)
    except Exception:
        save(out/"error.json", {"message": traceback.format_exc()})
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            seal(out)


if __name__ == "__main__":
    main()
