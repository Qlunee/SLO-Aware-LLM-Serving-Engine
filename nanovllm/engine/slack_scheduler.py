"""Opt-in V3 Slack policy with optional Aging, Hysteresis and KV Guard for BF16/eager.

Candidate prediction is side-effect free; KV allocation happens after a budget
has been selected. A total Prefill budget may be shared by waiting requests.

核心目标：Decode 请求不能因为加入 Prefill 而超过 ITL SLO，同时还要尽可能提高 Prefill 吞吐量。
    1. 先安排 Decode 请求
    2. 生成多个不同 Prefill 大小的候选 batch
    3. 用 Step Predictor 预测每个候选的耗时
    4. 加上固定和历史学习得到的安全边际
    5. 计算每个 Decode 请求的 ITL Slack
    6. 丢弃会导致 Slack 为负的候选
    7. 在剩余候选中选择 Prefill 最大的方案
    8. 预测不可靠时使用保守 fallback
    9. Prefill 长期等待时触发进度保护
    10. 根据实际执行时间在线修正安全边际

- self.fallback_chunk：默认 Prefill chunk 大小，通常是 256。
- candidates：所有候选 batch 的列表。如：只 Decode、Decode+256 Prefill、Decode+512 Prefill、...
- feasible：从 candidates 中筛选出的可行方案。

目前 Slack 有残差裕量、未覆盖回退和 Prefill 进展保护
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
                 min_residual_samples=8, hysteresis=False,
                 rise_steps=2, rise_margin_ms=10.0, aging=False,
                 aging_warning_ms=300.0, aging_hard_ms=800.0,
                 kv_guard=False, kv_low_free_blocks=8,
                 kv_high_free_blocks=16,
                 clock=perf_counter_ns):
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
        if type(hysteresis) is not bool or type(rise_steps) is not int or rise_steps < 1:
            raise ValueError("invalid hysteresis configuration")
        if not math.isfinite(rise_margin_ms) or rise_margin_ms < 0:
            raise ValueError("invalid hysteresis rise margin")
        if type(aging) is not bool or not all(math.isfinite(v) for v in
                (aging_warning_ms, aging_hard_ms)) or not 0 <= aging_warning_ms < aging_hard_ms:
            raise ValueError("invalid aging configuration")
        if (type(kv_guard) is not bool or type(kv_low_free_blocks) is not int or
                type(kv_high_free_blocks) is not int or
                not 0 <= kv_low_free_blocks < kv_high_free_blocks or
                (kv_guard and kv_high_free_blocks > config.num_kvcache_blocks)):
            raise ValueError("invalid KV Guard watermarks")
        self.predictor = predictor
        self.clock = clock
        self.safety_margin_ms = float(safety_margin_ms)
        self.fallback_chunk = fallback_chunk
        self.min_residual_samples = min_residual_samples
        self.hysteresis = hysteresis
        # 连续多少次满足条件后才扩大预算
        self.rise_steps = rise_steps
        # 扩大预算时要求至少有多少 Slack
        self.rise_margin_ms = float(rise_margin_ms)
        self.hysteresis_budget = None
        self.hysteresis_streak = 0
        self.aging = aging
        self.aging_warning_ms = float(aging_warning_ms)
        self.aging_hard_ms = float(aging_hard_ms)
        self.waiting_since_ns = {}
        self.kv_guard = kv_guard
        self.kv_low_free_blocks = kv_low_free_blocks
        self.kv_high_free_blocks = kv_high_free_blocks
        self.kv_guard_active = False
        # 保存历史预测误差，用于动态增加安全边际
        self.residuals = {"decode": deque(maxlen=residual_window),
                          "mixed": deque(maxlen=residual_window),
                          "prefill": deque(maxlen=residual_window)}
        self.itl_slo_ms = {}
        # 保存请求的 TTFT 截止时间
        self.ttft_deadline_ns = {}
        self.last_visible_ns = {}
        self.last_decision = {}
        self.last_batch = None
        # 记录连续跳过 Prefill 的次数，防止 Prefill 请求长期饥饿
        self.prefill_skip_steps = 0
        self.small_prefill_ms = deque(maxlen=32)

    def register_request(self, seq_id, itl_slo_ms, ttft_slo_ms=None, arrival_ns=None,
                         waiting_since_ns=None):
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
        if waiting_since_ns is None:
            waiting_since_ns = self.clock()
        # 每个请求都会记录从什么时候开始等待
        if type(waiting_since_ns) is not int or waiting_since_ns < 0:
            raise ValueError("invalid waiting start clock")
        self.waiting_since_ns[seq_id] = waiting_since_ns
        self.itl_slo_ms[seq_id] = float(itl_slo_ms)

    def preempt(self, seq):
        super().preempt(seq)
        self.waiting_since_ns[seq.seq_id] = self.clock()

    def reset_aging(self):
        self.waiting_since_ns.clear()

    def reset_kv_guard(self):
        self.kv_guard_active = False

    def _kv_guard_state(self):
        """Return free blocks and update low/high-watermark admission state."""
        free = len(self.block_manager.free_block_ids)
        was_active = self.kv_guard_active
        if self.kv_guard:
            if free <= self.kv_low_free_blocks:
                self.kv_guard_active = True
            elif free >= self.kv_high_free_blocks:
                self.kv_guard_active = False
        return free, self.kv_guard_active and not was_active, was_active and not self.kv_guard_active

    def _aging_state(self, now_ns):
        """Return oldest waiting segment, its reserve, and an age-prioritized order."""
        if not self.aging or not self.waiting:
            return None, None, 0, None, None
        for seq in self.waiting:
            if seq.seq_id not in self.waiting_since_ns:
                raise RuntimeError("waiting request has no aging clock")
        oldest = min(self.waiting, key=lambda seq: self.waiting_since_ns[seq.seq_id])
        age_ms = max(0, now_ns - self.waiting_since_ns[oldest.seq_id]) / 1e6
        if age_ms < self.aging_warning_ms:
            return oldest, age_ms, 0, None, None
        phase = "hard" if age_ms >= self.aging_hard_ms else "warning"
        reserve = (self.fallback_chunk if phase == "hard" else
                   max(1, math.ceil(self.fallback_chunk *
                       (age_ms - self.aging_warning_ms) /
                       (self.aging_hard_ms - self.aging_warning_ms))))
        # Promote the oldest request progressively. At the hard threshold it
        # reaches the front; before that, existing queue order changes gradually.
        order = list(self.waiting)
        original_index = order.index(oldest)
        fraction = min(1.0, (age_ms - self.aging_warning_ms) /
                       (self.aging_hard_ms - self.aging_warning_ms))
        new_index = original_index - math.ceil(fraction * original_index)
        order.pop(original_index)
        order.insert(new_index, oldest)
        return oldest, age_ms, min(reserve, self.fallback_chunk), phase, order

    @staticmethod
    def _aging_served(item, seq_id, reserve):
        for row in item[1]:
            if row.seq_id == seq_id and row.phase != "decode":
                return row.num_scheduled_tokens >= min(reserve, row.logical_tokens - row.start_pos)
        return False

    @staticmethod
    # 把候选 batch 转换成 StepPredictor 能理解的格式；这些特征会被送入预测器，查找对应 bucket 的 P95 Step 延迟
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

    def reset_hysteresis(self):
        self.hysteresis_budget = None
        self.hysteresis_streak = 0

    def _hysteresis_choice(self, feasible):
        """Lower immediately; raise one feasible tier after repeated headroom."""
        # 先找出当前最大可行 Prefill 预算 desired，再读取上一轮预算 previous。
        desired = max(feasible, key=lambda item: item[0])
        previous = self.hysteresis_budget
        if not self.hysteresis or previous is None or previous <= 0:
            self.hysteresis_streak = 0
            return desired, "slack"
        # 如果当前只能使用更小预算：直接降级
        if desired[0] <= previous:
            self.hysteresis_streak = 0
            return desired, "slack"
        # 如果当前可以使用更大预算，则找到一个不超过上一轮预算的方案
        lower = max((item for item in feasible if 0 < item[0] <= previous),
                    key=lambda item: item[0], default=None)
        if lower is None:
            # An entirely new candidate set must not strand a waiting request.
            self.hysteresis_streak = 0
            return desired, "slack"
        next_tier = min((item for item in feasible if item[0] > previous),
                        key=lambda item: item[0])
        if next_tier[-1] is None or next_tier[-1] < self.rise_margin_ms:
            self.hysteresis_streak = 0
            return lower, "hysteresis_hold"
        self.hysteresis_streak += 1
        if self.hysteresis_streak < self.rise_steps:
            return lower, "hysteresis_hold"
        self.hysteresis_streak = 0
        return next_tier, "hysteresis_raise"

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

    # 估算某个等待中的 Prefill 请求需要多少新的 KV block
    """
    它会考虑：
        已经存在的缓存前缀
        哪些缓存 block 仍然被其他请求使用
        哪些 block 可以复用
        需要分配多少新的 block
    """
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

    # 根据给定的 Prefill 预算生成候选方案
    def _prefill_plan(self, decode_rows, remaining, budget, *, grow_chunks=True,
                      waiting_order=None, min_free_blocks=0,
                      guard_blocked=None, capacity_blocked=None):
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
                    if capacity_blocked is not None:
                        capacity_blocked.add(seq.seq_id)
                    continue
                cached_blocks, new_blocks = allocation
                start = cached_blocks * self.block_size
            if start >= len(seq) or new_blocks > free_blocks:
                continue
            if not seq.block_table and new_blocks > free_blocks - min_free_blocks:
                if guard_blocked is not None:
                    guard_blocked.add(seq.seq_id)
                continue
            # Prefill token 数量
            # self.fallback_chunk：单个请求最多安排的默认 chunk 大小
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
        # 第一个返回值：本轮所有 Prefill 请求总共安排的 token 数；第二个返回值：具体安排了哪些请求及 token 数量
        return sum(row.num_scheduled_tokens for row in rows), rows

    def schedule(self):
        if self.pending is not None:
            raise RuntimeError("previous batch has not been committed")
        decision_ns = self.clock()
        decode_rows, remaining = self._decode_rows()
        kv_free_before, kv_entered, kv_exited = self._kv_guard_state()
        # A held guard admits no new KV owners until the high watermark is
        # restored. Outside the held state, projected admission must leave the
        # low watermark available for already-running Decode requests.
        kv_reserve = (self.kv_high_free_blocks if self.kv_guard_active else
                      self.kv_low_free_blocks) if self.kv_guard and decode_rows else 0
        kv_guard_blocked, kv_capacity_blocked = set(), set()
        aging_seq, aging_age_ms, aging_reserve, aging_phase, aging_order = (
            self._aging_state(self.clock()) if self.aging else
            (None, None, 0, None, None))
        aging_original_position = (list(self.waiting).index(aging_seq)
                                   if aging_phase else None)
        aging_promoted_position = (aging_order.index(aging_seq)
                                   if aging_phase else None)
        candidates = []
        if decode_rows:
            candidates.append((0, tuple(decode_rows), 0))
        for budget in sorted({*self.CANDIDATES[1:], self.fallback_chunk}):
            count, prefill_rows = self._prefill_plan(
                decode_rows, remaining, budget, waiting_order=aging_order,
                min_free_blocks=kv_reserve,
                guard_blocked=kv_guard_blocked,
                capacity_blocked=kv_capacity_blocked)
            if count:
                candidates.append((count, (*decode_rows, *prefill_rows), budget))
        # 遵循“一次给每个请求一个固定小 chunk”的策略，不会把剩余预算全部集中到某个请求上。
        fixed_count, fixed_rows = self._prefill_plan(
            decode_rows, remaining, remaining, grow_chunks=False,
            waiting_order=aging_order, min_free_blocks=kv_reserve,
            guard_blocked=kv_guard_blocked,
            capacity_blocked=kv_capacity_blocked)
        fixed_rows = (*decode_rows, *fixed_rows)
        if fixed_count and not any(rows == fixed_rows for _, rows, _ in candidates):
            candidates.append((fixed_count, fixed_rows, "fixed"))
        small_step_ms = ttft_urgency_ms = None
        #  TTFT 紧急处理
        if decode_rows and len(self.small_prefill_ms) >= 3:
            observed = sorted(self.small_prefill_ms)
            # 计算小 Prefill Step 的 P95
            small_step_ms = observed[math.ceil(.95 * len(observed)) - 1]
            urgency = []
            urgency_now_ns = self.clock()
            for seq in self.waiting:
                deadline = self.ttft_deadline_ns.get(seq.seq_id)
                if deadline is None or seq.num_completion_tokens:
                    continue
                chunks = math.ceil((len(seq) - seq.num_cached_tokens) / self.fallback_chunk)
                if chunks:
                    # 估算等待中的 Prefill 请求还剩多少时间
                    slack_ms = ((deadline - urgency_now_ns) / 1e6 -
                                chunks * small_step_ms - self.safety_margin_ms)
                    urgency.append((slack_ms, seq))
            if urgency:
                ttft_urgency_ms = min(slack for slack, _ in urgency)
                # 如果某些请求已经没有足够 slack：就生成一个 "ttft" 候选方案，优先安排即将违反 TTFT SLO 的 Prefill 请求。
                if ttft_urgency_ms <= 0:
                    urgent_order = [seq for slack, seq in sorted(urgency, key=lambda x: x[0])
                                    if slack <= 0]
                    urgent_count, urgent_rows = self._prefill_plan(
                        decode_rows, remaining, self.fallback_chunk,
                        grow_chunks=False, waiting_order=urgent_order,
                        min_free_blocks=kv_reserve,
                        guard_blocked=kv_guard_blocked,
                        capacity_blocked=kv_capacity_blocked)
                    if urgent_count:
                        candidates.append((urgent_count, (*decode_rows, *urgent_rows), "ttft"))
        if not candidates:
            raise RuntimeError("nonempty scheduler made no progress; check KV allocation")

        considered = []
        for count, rows, budget in candidates:
            prediction = self.predictor.predict(self._features(rows))
            phase = "mixed" if decode_rows and count else "decode" if decode_rows else "prefill"
            # 调度器不会完全相信预测器，而是额外加上安全边际
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
                # 对每个 Decode 请求，调度器计算：
                slacks.append(slo - age_ms - prediction.predicted_ms - margin)
            # 如果一个 batch 中有多个 Decode 请求，程序取其中最小的 Slack
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
        aging_feasible = ([item for item in feasible if self._aging_served(
            item, aging_seq.seq_id, aging_reserve)] if aging_phase else [])
        if expired and zero:
            chosen = (*zero, zero_slack)
            reason = "deadline_expired"
            # Decode 请求已经达到或超过 ITL SLO，这时优先选择 Decode-only，避免继续加入 Prefill
        elif ttft_choice is not None:
            chosen = ttft_choice
            reason = "ttft_urgent_fallback"
            # 某个 Prefill 请求的 TTFT 即将超时，因此优先安排紧急 Prefill
        elif has_prefill and decode_rows and not supported_mixed and (zero_slack is None or zero_slack >= 0):
            chosen = (*fallback, None)
            reason = "unsupported_fixed_fallback"
            # 混合 batch 的预测不受支持，因此使用固定 Prefill chunk
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
            # Prefill 已经连续多轮被跳过，程序强制尝试一个小 Prefill chunk，避免 Prefill 饥饿
        elif feasible:
            # 在所有可行候选中，选择 Prefill token 数最多的方案，从而尽量提高吞吐量
            pool = aging_feasible or feasible
            if decode_rows and urgent is None:
                chosen, reason = self._hysteresis_choice(pool)
            else:
                chosen = max(pool, key=lambda item: item[0])
                reason = "slack" if decode_rows else "no_decode"
            if aging_feasible and chosen[1] != max(feasible, key=lambda item: item[0])[1]:
                reason = f"aging_{aging_phase}"
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

        previous_budget = self.hysteresis_budget
        count, planned_rows, prediction, margin, calibrated, _, minimum = chosen
        # A request blocked in one candidate may still be served by a later
        # urgent candidate with a different order; count only actual deferrals.
        served_prefill_ids = {row.seq_id for row in planned_rows if row.phase != "decode"}
        kv_guard_blocked.difference_update(served_prefill_ids)
        kv_capacity_blocked.difference_update(served_prefill_ids)
        aging_served = (self._aging_served(chosen, aging_seq.seq_id, aging_reserve)
                        if aging_phase else False)
        aging_conflict = None
        if aging_phase == "hard" and not aging_served:
            if expired or (zero_slack is not None and zero_slack < 0):
                aging_conflict = "decode_deadline"
            elif reason == "ttft_urgent_fallback":
                aging_conflict = "ttft_priority"
            elif aging_seq.seq_id in kv_guard_blocked:
                aging_conflict = "kv_guard"
            elif not any(self._aging_served(item, aging_seq.seq_id, aging_reserve)
                         for item in considered):
                aging_conflict = "resource_limit"
            elif not aging_feasible:
                aging_conflict = "prediction_or_decode_slack"
            else:
                aging_conflict = "policy_priority"
        if decode_rows and has_prefill:
            self.hysteresis_budget = count
            if reason not in ("hysteresis_hold", "hysteresis_raise"):
                self.hysteresis_streak = 0
        else:
            self.reset_hysteresis()
        # Prefill 防饥饿机制
        self.prefill_skip_steps = (self.prefill_skip_steps + 1
                                   if count == 0 and fixed_count else 0)
        rows = list(decode_rows)
        if count:
            for row in planned_rows[len(decode_rows):]:
                seq = self.sequences[row.seq_id]
                if not seq.block_table:
                    # 真正分配 KV cache 并构造 batch
                    self.block_manager.allocate(seq, row.start_pos // self.block_size)
                seq.num_scheduled_tokens = row.num_scheduled_tokens
                seq.is_prefill = True
                rows.append(row)
        batch = build_batch(rows, self.sequences, self.max_num_batched_tokens, self.block_size)
        self.pending = batch
        self.last_batch = batch
        kv_conflict = None
        if kv_guard_blocked:
            if expired:
                kv_conflict = "decode_deadline"
            elif ttft_urgency_ms is not None and ttft_urgency_ms <= 0:
                kv_conflict = "ttft_deadline"
            elif aging_phase == "hard" and aging_seq.seq_id in kv_guard_blocked:
                kv_conflict = "aging_hard"
        policy_name = "slack" + ("_hysteresis" if self.hysteresis else "") + (
            "_aging" if self.aging else "") + ("_kv_guard" if self.kv_guard else "")
        self.last_decision = {"policy": policy_name if policy_name != "slack" else "slack_only",
                              "decision_start_ns": decision_ns,
                              "selected_prefill_budget": count, "decision_reason": reason,
                              "hysteresis_enabled": self.hysteresis,
                              "hysteresis_previous_budget": previous_budget,
                              "hysteresis_streak": self.hysteresis_streak,
                              "hysteresis_rise_steps": self.rise_steps,
                              "hysteresis_rise_margin_ms": self.rise_margin_ms,
                              "hysteresis_target_budget": (max(feasible, key=lambda item: item[0])[0]
                                                           if feasible else None),
                              "hysteresis_limited": reason == "hysteresis_hold",
                              "aging_enabled": self.aging,
                              "aging_phase": aging_phase,
                              "aging_oldest_seq_id": aging_seq.seq_id if aging_seq else None,
                              "aging_wait_kind": ("recompute" if aging_seq.num_completion_tokens else
                                                  "inter_chunk" if aging_seq.num_cached_tokens else "initial")
                                                  if aging_seq else None,
                              "aging_oldest_wait_ms": aging_age_ms,
                              "aging_reserved_tokens": aging_reserve,
                              "aging_original_position": aging_original_position,
                              "aging_promoted_position": aging_promoted_position,
                              "aging_served": aging_served,
                              "aging_conflict": aging_conflict,
                              "kv_guard_enabled": self.kv_guard,
                              "kv_guard_active": self.kv_guard_active,
                              "kv_guard_entered": kv_entered,
                              "kv_guard_exited": kv_exited,
                              "kv_guard_low_free_blocks": self.kv_low_free_blocks,
                              "kv_guard_high_free_blocks": self.kv_high_free_blocks,
                              "kv_guard_reserve_blocks": kv_reserve,
                              "kv_guard_free_blocks_before": kv_free_before,
                              "kv_guard_free_blocks_after": len(self.block_manager.free_block_ids),
                              "kv_guard_blocked_seq_ids": sorted(kv_guard_blocked),
                              "kv_capacity_blocked_seq_ids": sorted(kv_capacity_blocked),
                              "kv_guard_conflict": kv_conflict,
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
            if row.phase != "decode":
                seq = self.sequences[row.seq_id]
                if seq in self.waiting:
                    self.waiting_since_ns[row.seq_id] = visible_ns
                else:
                    self.waiting_since_ns.pop(row.seq_id, None)
        if decision.get("prediction_supported"):
            phase = "mixed" if batch.num_decode_tokens and batch.num_prefill_tokens else (
                "decode" if batch.num_decode_tokens else "prefill")
            # 加入对应阶段的残差队列
            self.residuals[phase].append(actual_ms - decision["predicted_ms"])
            decision["actual_ms"] = actual_ms
            decision["underprediction_ms"] = max(0.0, actual_ms - decision["predicted_ms"])
