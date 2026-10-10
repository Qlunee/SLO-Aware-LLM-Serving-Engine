"""Opt-in V1 engine. Existing LLMEngine/V0 APIs stay unchanged."""
import atexit
from dataclasses import fields

import torch.multiprocessing as mp
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.mixed_sequence import MixedSequence
from nanovllm.engine.mixed_scheduler import MixedScheduler
from nanovllm.engine.mixed_model_runner import MixedModelRunner


class MixedLLMEngine:
    def __init__(self, model, chunk_size=512, kv_block_cap=None,
                 greedy=True, verify_tp_plan=False, **kwargs):
        unknown = set(kwargs) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"unknown engine options: {unknown}")
        config = Config(model, **kwargs)
        if min(chunk_size, config.max_num_seqs, config.max_num_batched_tokens) <= 0:
            raise ValueError("invalid chunk/budget/max_num_seqs")
        if kv_block_cap is not None and kv_block_cap <= 0:
            raise ValueError("invalid test KV cap")
        if not config.enforce_eager and config.max_num_seqs < 8:
            raise ValueError("frozen graph capture requires max_num_seqs >= 8")
        Sequence.block_size = config.kvcache_block_size
        self.greedy = greedy
        self.verify_tp_plan = verify_tp_plan
        self.closed = False
        self.ps, self.events = [], []
        self.config = config
        self.last_step = None
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, local_files_only=True)
        config.eos = self.tokenizer.eos_token_id
        ctx = mp.get_context("spawn")
        for rank in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=MixedModelRunner, args=(config, rank, event, kv_block_cap))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        try:
            self.model_runner = MixedModelRunner(config, 0, self.events, kv_block_cap)
        except BaseException:
            for process in self.ps:
                if process.is_alive():
                    process.terminate()
                process.join()
            raise
        self.scheduler = MixedScheduler(config, chunk_size)
        atexit.register(self.exit)

    def add_request(self, prompt, sampling_params):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("empty prompt")
        seq = MixedSequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self, return_logits=False):
        if self.is_finished():
            raise RuntimeError("step called on an empty engine")
        batch = self.scheduler.schedule()
        result = self.model_runner.call("run_mixed", batch, return_logits, self.verify_tp_plan, self.greedy)
        self.scheduler.postprocess(batch, result["tokens"])
        self.last_step = {"batch": batch, **result}
        outputs = [(row.seq_id, list(self.scheduler.sequences[row.seq_id].completion_token_ids))
                   for row in batch.requests if self.scheduler.sequences[row.seq_id].is_finished]
        return outputs, {"prefill_tokens": batch.num_prefill_tokens,
                         "decode_tokens": batch.num_decode_tokens}

    def exit(self):
        if self.closed:
            return
        self.closed = True
        atexit.unregister(self.exit)
        self.model_runner.call("exit")
        for process in self.ps:
            process.join()
