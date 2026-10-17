"""Finite controlled-case driver, independent of CUDA for state regression."""
from .protocol import decode_context_lengths, decode_output_limit
from .timing import timed_step


def run_case(engine, case, settings, add, observe):
    scheduler = engine.scheduler
    scheduler.max_num_batched_tokens = settings.token_budget
    scheduler.chunk_size = 256
    d, p, m = (case[k] for k in ("decode_count", "prefill_budget", "prefill_requests"))
    base = [add(length, decode_output_limit(settings, case))
            for length in decode_context_lengths(case)]
    for _ in range(10000):
        if all(scheduler.sequences[i].status.name == "RUNNING" for i in base):
            break
        engine.step()
    else:
        raise RuntimeError("Decode bootstrap exhausted")
    scheduler.max_num_batched_tokens = d+p
    scheduler.chunk_size = p//m if p else 256
    for episode in range(settings.warmup+settings.repeats):
        warmup = episode < settings.warmup
        prompt = (p//m if settings.prefill_shape == "exact" else 2*(p//m)+1) if p else 0
        inserted = [add(prompt, 2) for _ in range(m)] if p else []
        for local_step in range(100):
            batch, result, times = timed_step(engine)
            observe(case, episode, local_step, warmup, batch, result, times)
            if any(scheduler.sequences[i].is_finished for i in base):
                raise RuntimeError("base Decode request finished inside a profile case")
            if scheduler.preemption_count:
                raise RuntimeError("controlled profile unexpectedly preempted; retain invalid run")
            if not inserted or all(scheduler.sequences[i].is_finished for i in inserted):
                break
        else:
            raise RuntimeError("Prefill episode made insufficient progress")
    scheduler.max_num_batched_tokens = settings.token_budget
    scheduler.chunk_size = 256
    for _ in range(10000):
        if engine.is_finished():
            break
        engine.step()
    if not engine.is_finished():
        raise RuntimeError("finite drain exhausted")
    scheduler.assert_consistent()
    if scheduler.block_manager.used_block_ids:
        raise AssertionError("KV blocks leaked after drain")
    scheduler.sequences.clear()
