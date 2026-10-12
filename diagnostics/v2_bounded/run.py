"""Single fresh-process run for the bounded-warmup protocol."""
from dataclasses import asdict
from pathlib import Path
import time
import traceback

from benchmarks.v2_stall.protocol import ROOT, read, save, seal, sha
from benchmarks.v2_baselines.metrics import analyze_online
from benchmarks.v2_baselines.workload import validate_trace
from .protocol import load_suite, find_job
from .warmup import prewarm


def identity(suite, protocol, job):
    return {"schema": "v2-bounded-run-v1", "protocol_sha256": sha(suite / "protocol.json"),
            "job": job, "tp": 1, "graph": False, "observation": "full",
            "sampler": "greedy", "diagnostic_only": False,
            "warmup_sha256": protocol["warmup_sha256"],
            "tool_sha256": protocol["tool_sha256"]}


class GPUDevice:
    """Hardware boundary; CPU tests substitute only this external device."""
    def check_and_seed(self, seed):
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError("expose exactly one GPU")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def environment(self, protocol):
        from benchmarks.v0.environment import collect_environment
        return collect_environment(ROOT, protocol["model"]["path"], protocol["model"]["revision"])

    def create(self, protocol, policy):
        import torch
        from benchmarks.v2_baselines.run import make_engine
        engine, config = make_engine(protocol, policy)
        if engine.model_runner.kv_cache.dtype != torch.bfloat16:
            engine.exit()
            raise ValueError("BF16 KV required")
        return engine, config

    def synchronize(self):
        import torch
        torch.cuda.synchronize()

    def reset_peak(self):
        import torch
        torch.cuda.reset_peak_memory_stats()

    def peaks(self):
        import torch
        return {"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def run_job(suite, job_id, out, device=None, root=ROOT):
    from benchmarks.v0.driver import execute_trace
    from benchmarks.v2_stall.run import check_state
    suite, out = Path(suite).resolve(), Path(out).resolve()
    protocol = load_suite(suite, root=root)
    job = find_job(protocol, job_id)
    device = device or GPUDevice()
    out.mkdir(parents=True, exist_ok=False)
    save(out / "identity.json", identity(suite, protocol, job))
    trace = read(suite / job["trace"])
    save(out / "trace.json", trace)
    engine = None
    stage = "SETUP"
    outcome = {"status": "SETUP_FAILED", "diagnostic_only": False}
    code = 1
    try:
        device.check_and_seed(job["seed"])
        save(out / "environment.json", device.environment(protocol))
        runtime_protocol = protocol | {"settings": protocol["settings"] | {"seed": job["seed"]}}
        engine, config = device.create(runtime_protocol, job["policy"])
        stage = "WARMUP"
        warm = prewarm(engine, config, read(suite / "warmup.json"), out / "warmup")
        device.synchronize()
        # Fail before measurement if anything changed while initializing/warming.
        load_suite(suite, root=root)
        stage = "MEASUREMENT"
        device.reset_peak()
        result = execute_trace(engine, config, trace, validate_trace(trace),
                               protocol["settings"]["drain_s"], "full",
                               sleep=getattr(engine, "sleep", time.sleep))
        device.synchronize()
        state_ok = True
        try:
            check_state(engine)
        except Exception:
            state_ok = False
            save(out / "state_error.json", {"message": traceback.format_exc()})
        load_suite(suite, root=root)
        summary = analyze_online(trace, result.records, engine.steps, result.backlog,
                                 result.end_ns, protocol["settings"]["starvation_ms"])
        summary.update(policy=job["policy"], mode=job["mode"], seed=job["seed"], rps=job["rps"],
            protocol_sha256=sha(suite / "protocol.json"), job_id=job_id,
            valid_measurement=bool(not result.error and state_ok),
            end_state_check_passed=state_ok, source_unchanged=True, diagnostic_only=False,
            end_ns=result.end_ns, deadline_ns=result.deadline_ns,
            kv_blocks=config.num_kvcache_blocks,
            kv_payload_bytes=engine.model_runner.kv_cache.numel() * engine.model_runner.kv_cache.element_size(),
            warmup=warm, **device.peaks(),
            memory_scope="PyTorch allocator over measured trace INCLUDING drain; not NVML total VRAM")
        for name, value in (("summary.json", summary),
                            ("requests.json", [asdict(r) for r in result.records]),
                            ("steps.json", engine.steps), ("backlog.json", result.backlog)):
            save(out / name, value)
        if result.error:
            save(out / "error.json", {"message": result.error})
        if summary["valid_measurement"]:
            outcome["status"] = "VALID_COMPLETE" if summary["all_completed"] else "VALID_INCOMPLETE"
            code = 0 if summary["all_completed"] else 2
        else:
            outcome["status"] = "MEASUREMENT_FAILED"
    except KeyboardInterrupt:
        outcome.update(status="INTERRUPTED", exception=traceback.format_exc())
        code = 130
    except BaseException:
        outcome.update(status=f"{stage}_FAILED", exception=traceback.format_exc())
        code = 1
    finally:
        try:
            if engine is not None:
                engine.exit()
        except BaseException:
            outcome.update(status="CLEANUP_FAILED", cleanup_exception=traceback.format_exc())
            code = 1
        finally:
            save(out / "outcome.json", outcome)
            seal(out)
    print({"job": job_id, "status": outcome["status"], "out": str(out)}, flush=True)
    return code
