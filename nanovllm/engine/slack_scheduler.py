"""Opt-in V3 Slack-only policy for the BF16/eager mixed engine.

Candidate prediction is side-effect free; KV allocation happens after a budget
has been selected. A total Prefill budget may be shared by waiting requests.
"""
from collections import deque
import json
import math
from time import perf_counter_ns

from nanovllm.engine.batch_metadata import ScheduledRequest, build_batch
from nanovllm.engine.mixed_scheduler import MixedScheduler


class SlackScheduler(MixedScheduler):
    CANDIDATES = (0, 256, 512, 1024, 2048, 4096)

    def __init__(self, config, predictor, *, safety_margin_ms=15.0,
                 fallback_chunk=256, residual_window=128,
                 min_residual_samples=8, clock=perf_counter_ns):
        super().__init__(config, fallback_chunk)
        if not config.enforce_eager:
            raise ValueError("V3 Slack-only currently requires eager execution")
        if not math.isfinite(safety_margin_ms) or safety_margin_ms < 0:
            raise ValueError("invalid safety margin")
        if type(fallback_chunk) is not int or fallback_chunk <= 0:
            raise ValueError("invalid fallback chunk")
        if type(residual_window) is not int or residual_window < 1:
            raise ValueError("invalid residual window")
        if type(min_residual_samples) is not int or not 1 <= min_residual_samples <= residual_window:
            raise ValueError("invalid residual sample count")
        self.predictor = predictor
        self.clock = clock
        self.safety_margin_ms = float(safety_margin_ms)
        self.fallback_chunk = fallback_chunk
        self.min_residual_samples = min_residual_samples
        self.residuals = {"decode": deque(maxlen=residual_window),
                          "mixed": deque(maxlen=residual_window),
                          "prefill": deque(maxlen=residual_window)}
        self.itl_slo_ms = {}
        self.ttft_deadline_ns = {}
        self.last_visible_ns = {}
        self.last_decision = {}
        self.last_batch = None
        self.prefill_skip_steps = 0
        self.small_prefill_ms = deque(maxlen=32)

    def register_request(self, seq_id, itl_slo_ms, ttft_slo_ms=None, arrival_ns=None):
        if seq_id not in self.sequences or seq_id in self.itl_slo_ms:
            raise ValueError("unknown or duplicate request SLO")
        if not math.isfinite(itl_slo_ms) or itl_slo_ms <= 0:
            raise ValueError("invalid ITL SLO")
        if (ttft_slo_ms is None) != (arrival_ns is None):
            raise ValueError("TTFT SLO and arrival clock must be provided together")
        if ttft_slo_ms is not None:
            if (not math.isfinite(ttft_slo_ms) or ttft_slo_ms <= 0 or
                    type(arrival_ns) is not int or arrival_ns < 0):
                raise ValueError("invalid TTFT SLO or arrival clock")
            self.ttft_deadline_ns[seq_id] = arrival_ns + math.ceil(ttft_slo_ms * 1e6)
        self.itl_slo_ms[seq_id] = float(itl_slo_ms)

    @staticmethod
    def _features(rows):
        decode = sum(row.phase == "decode" for row in rows)
        prefill = sum(row.num_scheduled_tokens for row in rows if row.phase != "decode")
        prefill_requests = len(rows) - decode
        contexts = [row.end_pos for row in rows]
        return {"decode_count": decode, "prefill_tokens": prefill,
                "prefill_request_count": prefill_requests,
                "sum_context_tokens": sum(contexts),
                "max_context_len": max(contexts),
                "sample_count": sum(row.should_sample for row in rows),
                "prefill_has_history": any(row.phase != "decode" and row.start_pos > 0
                                           for row in rows),
                "graph_mode": "eager"}

    def _margin(self, phase):
        values = self.residuals[phase]
        # Residuals are signed; the fit-table percentile intentionally accepts
        # only nonnegative Step durations and cannot be reused here.
        learned = (max(0.0, sorted(values)[math.ceil(.95 * len(values)) - 1])
                   if len(values) >= self.min_residual_samples else 0.0)
        return self.safety_margin_ms + learned, len(values) >= self.min_residual_samples

    def _decode_rows(self):
        rows, selected = [], set()
        remaining = self.max_num_batched_tokens
        for seq in list(self.running):
            if not remaining or len(rows) == self.max_num_seqs:
                break
            if seq not in self.running:
                continue
            if seq.num_cached_tokens != len(seq) - 1:
                raise RuntimeError("RUNNING request is not decode-ready")
            while not self.block_manager.can_append(seq):
                victim = next((s for s in reversed(self.running)
                               if s.seq_id not in selected and s is not seq), None)
                if victim is None:
                    self.preempt(seq)
                    break
                self.preempt(victim)
            else:
                self.block_manager.may_append(seq)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                rows.append(ScheduledRequest(seq.seq_id, "decode", seq.num_cached_tokens, 1, len(seq)))
                selected.add(seq.seq_id)
                remaining -= 1
        return rows, remaining

    def _allocation_plan(self, seq, allow_free_cached_prefix):
        """Return a stable cached prefix and conservative free-block cost."""
        cached = self.block_manager.can_allocate(seq)
        if cached < 0:
            return None
        used_prefix = stable_prefix = 0
        prefix_hash = -1
        for index in range(cached):
            prefix_hash = self.block_manager.compute_hash(seq.block(index), prefix_hash)
            block_id = self.block_manager.hash_to_block_id[prefix_hash]
            if block_id in self.block_manager.used_block_ids:
                used_prefix += 1
                if stable_prefix == index:
                    stable_prefix += 1
        # An earlier planned allocation can evict an unused cached block. Only
        # the first new request may depend on free cached blocks; later ones
        # reuse the contiguous prefix that is already held by active requests.
        if allow_free_cached_prefix:
            return cached, seq.num_blocks - used_prefix
        return stable_prefix, seq.num_blocks - stable_prefix

    def _prefill_plan(self, decode_rows, remaining, budget, *, grow_chunks=True,
                      waiting_order=None):
        remaining_budget = min(remaining, budget)
        slots = self.max_num_seqs - len(decode_rows)
        free_blocks = len(self.block_manager.free_block_ids)
        plans = []
        new_allocation_planned = False
        for seq in list(self.waiting) if waiting_order is None else waiting_order:
            if not remaining_budget or not slots:
                break
            if seq.block_table:
                start, new_blocks = seq.num_cached_tokens, 0
            else:
                allocation = self._allocation_plan(seq, not new_allocation_planned)
                if allocation is None:
                    continue
                cached_blocks, new_blocks = allocation
                start = cached_blocks * self.block_size
            if start >= len(seq) or new_blocks > free_blocks:
                continue
            count = min(self.fallback_chunk, len(seq) - start, remaining_budget)
            plans.append([seq, start, count])
            remaining_budget -= count
            free_blocks -= new_blocks
            slots -= 1
            new_allocation_planned |= not seq.block_table
        # Fixed fallback retains the V1 one-chunk-per-request packing. Other
        # candidates may spend the remaining budget growing selected chunks.
        if grow_chunks:
            for plan in plans:
                if not remaining_budget:
                    break
                seq, start, count = plan
                extra = min(remaining_budget, len(seq) - start - count)
                plan[2] += extra
                remaining_budget -= extra
        rows = tuple(ScheduledRequest(seq.seq_id,
            "recompute" if seq.num_completion_tokens else "prefill",
            start, count, len(seq)) for seq, start, count in plans)
        return sum(row.num_scheduled_tokens for row in rows), rows

    def schedule(self):
        if self.pending is not None:
            raise RuntimeError("previous batch has not been committed")
        decision_ns = self.clock()
        decode_rows, remaining = self._decode_rows()
        candidates = []
        if decode_rows:
            candidates.append((0, tuple(decode_rows), 0))
        for budget in sorted({*self.CANDIDATES[1:], self.fallback_chunk}):
            count, prefill_rows = self._prefill_plan(decode_rows, remaining, budget)
            if count:
                candidates.append((count, (*decode_rows, *prefill_rows), budget))
        fixed_count, fixed_rows = self._prefill_plan(
            decode_rows, remaining, remaining, grow_chunks=False)
        fixed_rows = (*decode_rows, *fixed_rows)
        if fixed_count and not any(rows == fixed_rows for _, rows, _ in candidates):
            candidates.append((fixed_count, fixed_rows, "fixed"))
        small_step_ms = ttft_urgency_ms = None
        if decode_rows and len(self.small_prefill_ms) >= 3:
            observed = sorted(self.small_prefill_ms)
            small_step_ms = observed[math.ceil(.95 * len(observed)) - 1]
            urgency = []
            urgency_now_ns = self.clock()
            for seq in self.waiting:
                deadline = self.ttft_deadline_ns.get(seq.seq_id)
                if deadline is None or seq.num_completion_tokens:
                    continue
                chunks = math.ceil((len(seq) - seq.num_cached_tokens) / self.fallback_chunk)
                if chunks:
                    slack_ms = ((deadline - urgency_now_ns) / 1e6 -
                                chunks * small_step_ms - self.safety_margin_ms)
                    urgency.append((slack_ms, seq))
            if urgency:
                ttft_urgency_ms = min(slack for slack, _ in urgency)
                if ttft_urgency_ms <= 0:
                    urgent_order = [seq for slack, seq in sorted(urgency, key=lambda x: x[0])
                                    if slack <= 0]
                    urgent_count, urgent_rows = self._prefill_plan(
                        decode_rows, remaining, self.fallback_chunk,
                        grow_chunks=False, waiting_order=urgent_order)
                    if urgent_count:
                        candidates.append((urgent_count, (*decode_rows, *urgent_rows), "ttft"))
        if not candidates:
            raise RuntimeError("nonempty scheduler made no progress; check KV allocation")

        considered = []
        for count, rows, budget in candidates:
            prediction = self.predictor.predict(self._features(rows))
            phase = "mixed" if decode_rows and count else "decode" if decode_rows else "prefill"
            margin, calibrated = self._margin(phase)
            considered.append((count, rows, prediction, margin, calibrated, budget))

        # Candidate evaluation consumes CPU time, so re-read the clock before
        # checking the next-token deadline.  The predictor covers the ordinary
        # Step path; this extra controller cost is charged separately.
        now_ns = self.clock()
        feasible = []
        scored = []
        for count, rows, prediction, margin, calibrated, budget in considered:
            if not decode_rows or not prediction.supported:
                if not decode_rows:
                    feasible.append((count, rows, prediction, margin, calibrated, budget, None))
                continue
            slacks = []
            for row in decode_rows:
                last = self.last_visible_ns.get(row.seq_id)
                slo = self.itl_slo_ms.get(row.seq_id)
                if last is None or slo is None:
                    raise RuntimeError("Decode request has no visible-token clock or ITL SLO")
                age_ms = max(0, now_ns - last) / 1e6
                slacks.append(slo - age_ms - prediction.predicted_ms - margin)
            minimum = min(slacks)
            scored.append((count, minimum))
            if minimum >= 0:
                feasible.append((count, rows, prediction, margin, calibrated, budget, minimum))

        has_prefill = any(c for c, *_ in considered)
        supported_mixed = any(c and p.supported for c, _, p, _, _, _ in considered)
        zero = next((item for item in considered if item[0] == 0), None)
        zero_slack = next((s for c, s in scored if c == 0), None)
        expired = any((now_ns - self.last_visible_ns[row.seq_id]) / 1e6 >=
                      self.itl_slo_ms[row.seq_id] for row in decode_rows)
        cold_mixed = supported_mixed and not any(c and p.supported and calibrated
            for c, _, p, _, calibrated, _ in considered)
        fallback = next((item for item in considered
                         if item[0] and item[1] == fixed_rows), None)
        ttft_choice = None
        urgent = next((item for item in considered if item[-1] == "ttft"), None)
        if (urgent is not None and zero_slack is not None and zero_slack >= 0 and
                not any(item[0] for item in feasible)):
            count, rows, prediction, margin, calibrated, budget = urgent
            empirical_slack = min(
                self.itl_slo_ms[row.seq_id] -
                max(0, now_ns - self.last_visible_ns[row.seq_id]) / 1e6 -
                small_step_ms - self.safety_margin_ms for row in decode_rows)
            if empirical_slack >= 0:
                predicted_slack = (min(
                    self.itl_slo_ms[row.seq_id] -
                    max(0, now_ns - self.last_visible_ns[row.seq_id]) / 1e6 -
                    prediction.predicted_ms - margin for row in decode_rows)
                    if prediction.supported else None)
                ttft_choice = (*urgent, predicted_slack)
        progress_guard = None
        if (self.prefill_skip_steps >= 2 and decode_rows and fixed_count and
                zero_slack is not None and zero_slack >= 0 and
                not any(item[0] for item in feasible)):
            # A phase-wide positive residual can veto every supported Prefill
            # candidate indefinitely. After two such skips, retry one small
            # chunk if it still fits with the fixed safety margin. Keep the
            # calibrated (possibly negative) slack in the decision record.
            for item in considered:
                count, rows, prediction, margin, calibrated, budget = item
                if not 0 < count <= self.fallback_chunk or not prediction.supported:
                    continue
                base_slack = min(
                    self.itl_slo_ms[row.seq_id] -
                    max(0, now_ns - self.last_visible_ns[row.seq_id]) / 1e6 -
                    prediction.predicted_ms - self.safety_margin_ms
                    for row in decode_rows)
                if base_slack >= 0 and (progress_guard is None or count > progress_guard[0]):
                    calibrated_slack = next((s for c, s in scored if c == count), None)
                    progress_guard = (*item, calibrated_slack)
        if expired and zero:
            chosen = (*zero, zero_slack)
            reason = "deadline_expired"
        elif ttft_choice is not None:
            chosen = ttft_choice
            reason = "ttft_urgent_fallback"
        elif has_prefill and decode_rows and not supported_mixed and (zero_slack is None or zero_slack >= 0):
            chosen = (*fallback, None)
            reason = "unsupported_fixed_fallback"
        elif has_prefill and decode_rows and cold_mixed:
            fallback_slack = next((s for c, s in scored if c == fallback[0]), None)
            if fallback[2].supported and fallback_slack is not None and fallback_slack >= 0:
                chosen = (*fallback, fallback_slack)
                reason = "cold_start_fixed_fallback"
            elif not fallback[2].supported and (zero_slack is None or zero_slack >= 0):
                chosen = (*fallback, None)
                reason = "unsupported_fixed_fallback"
            else:
                chosen = (*zero, zero_slack)
                reason = "cold_start_decode_only"
        elif progress_guard is not None:
            chosen = progress_guard
            reason = "prefill_progress_guard"
        elif feasible:
            chosen = max(feasible, key=lambda item: item[0])
            reason = "slack" if decode_rows else "no_decode"
        else:
            if zero and (zero[2].supported or supported_mixed):
                chosen = (*zero, zero_slack)
                reason = "no_feasible_prefill" if supported_mixed else "decode_risk"
            else:
                if fallback is None:
                    fallback = next((item for item in considered if item[0] > 0), None)
                if fallback is None:
                    chosen = (*zero, None)
                    reason = "unsupported_decode_only"
                else:
                    chosen = (*fallback, None)
                    reason = "unsupported_fixed_fallback"

        count, planned_rows, prediction, margin, calibrated, _, minimum = chosen
        self.prefill_skip_steps = (self.prefill_skip_steps + 1
                                   if count == 0 and fixed_count else 0)
        rows = list(decode_rows)
        if count:
            for row in planned_rows[len(decode_rows):]:
                seq = self.sequences[row.seq_id]
                if not seq.block_table:
                    self.block_manager.allocate(seq, row.start_pos // self.block_size)
                seq.num_scheduled_tokens = row.num_scheduled_tokens
                seq.is_prefill = True
                rows.append(row)
        batch = build_batch(rows, self.sequences, self.max_num_batched_tokens, self.block_size)
        self.pending = batch
        self.last_batch = batch
        self.last_decision = {"policy": "slack_only", "decision_start_ns": decision_ns,
                              "selected_prefill_budget": count, "decision_reason": reason,
                              "selected_prefill_requests": len(rows) - len(decode_rows),
                              "predicted_ms": prediction.predicted_ms,
                              "prediction_supported": prediction.supported,
                              "prediction_reason": prediction.reason,
                              "prediction_sample_count": prediction.sample_count,
                              "safety_margin_ms": margin, "residual_calibrated": calibrated,
                              "min_slack_ms": minimum,
                              "unavoidable_risk": bool(decode_rows and count == 0 and
                                                       minimum is not None and minimum < 0),
                              "candidate_count": len(considered),
                              "unsupported_candidate_features": json.dumps([
                                  self._features(candidate_rows)
                                  for _, candidate_rows, candidate_prediction, _, _, _ in considered
                                  if not candidate_prediction.supported], separators=(",", ":")),
                              "actual_ms": None, "underprediction_ms": None,
                              "progress_guard_rearmed": False,
                              "ttft_service_slack_ms": ttft_urgency_ms,
                              "small_prefill_p95_ms": small_step_ms,
                              "decision_cpu_us": (self.clock() - decision_ns) / 1000}
        return batch

    def observe_visible(self, batch, visible_ns):
        if type(visible_ns) is not int:
            raise ValueError("visible timestamp must be an integer")
        decision = self.last_decision
        actual_ms = (visible_ns - decision["decision_start_ns"]) / 1e6
        if batch.num_decode_tokens and 0 < batch.num_prefill_tokens <= self.fallback_chunk:
            self.small_prefill_ms.append(actual_ms)
        if decision.get("decision_reason") == "prefill_progress_guard" and self.waiting:
            # Keep serving small chunks while their observed token intervals
            # retain the fixed safety margin. A slow guarded Step restores the
            # two-step cooldown before another deliberate override.
            intervals = [((visible_ns - self.last_visible_ns[row.seq_id]) / 1e6,
                          self.itl_slo_ms[row.seq_id])
                         for row in batch.requests if row.phase == "decode"]
            if intervals and all(elapsed + self.safety_margin_ms <= slo
                                 for elapsed, slo in intervals):
                self.prefill_skip_steps = 2
                decision["progress_guard_rearmed"] = True
        for row in batch.requests:
            if row.should_sample:
                self.last_visible_ns[row.seq_id] = visible_ns
        if decision.get("prediction_supported"):
            phase = "mixed" if batch.num_decode_tokens and batch.num_prefill_tokens else (
                "decode" if batch.num_decode_tokens else "prefill")
            self.residuals[phase].append(actual_ms - decision["predicted_ms"])
            decision["actual_ms"] = actual_ms
            decision["underprediction_ms"] = max(0.0, actual_ms - decision["predicted_ms"])
