"""CPU-only, immutable execution plan. No Sequence objects cross the TP boundary."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledRequest:
    seq_id: int
    phase: str  # decode / prefill / recompute
    start_pos: int
    num_scheduled_tokens: int
    logical_tokens: int

    @property
    def end_pos(self):
        return self.start_pos + self.num_scheduled_tokens

    @property
    def should_sample(self):
        return self.end_pos == self.logical_tokens


@dataclass(frozen=True)
class BatchMetadata:
    requests: tuple[ScheduledRequest, ...]
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    slot_mapping: tuple[int, ...]
    block_tables: tuple[tuple[int, ...], ...]
    query_offsets: tuple[int, ...]
    sample_indices: tuple[int, ...]
    sample_seq_ids: tuple[int, ...]
    temperatures: tuple[float, ...]
    num_decode_tokens: int
    num_prefill_tokens: int
    token_budget: int

    @property
    def pure_decode(self):
        return self.num_prefill_tokens == 0

    def validate(self):
        if not self.requests or len({r.seq_id for r in self.requests}) != len(self.requests):
            raise ValueError("empty batch or duplicate request")
        n = len(self.input_ids)
        if not 0 < n <= self.token_budget:
            raise ValueError("global token budget exceeded")
        if n != self.num_decode_tokens + self.num_prefill_tokens:
            raise ValueError("phase token counts disagree")
        if len(self.positions) != n or len(self.slot_mapping) != n:
            raise ValueError("flat tensor lengths disagree")
        if len(self.block_tables) != len(self.requests):
            raise ValueError("block table row count")
        if len(self.query_offsets) != len(self.requests) + 1 or self.query_offsets[0] != 0:
            raise ValueError("query offsets")
        if self.query_offsets[-1] != n:
            raise ValueError("query offsets do not cover batch")
        expected_indices, expected_ids = [], []
        seen_prefill = False
        decode_count = 0
        for i, row in enumerate(self.requests):
            if row.phase not in {"decode", "prefill", "recompute"}:
                raise ValueError("invalid phase")
            if not 0 <= row.start_pos < row.end_pos <= row.logical_tokens:
                raise ValueError("invalid computed span")
            a, b = self.query_offsets[i:i + 2]
            if b - a != row.num_scheduled_tokens:
                raise ValueError("span / offsets mismatch")
            if self.positions[a:b] != tuple(range(row.start_pos, row.end_pos)):
                raise ValueError("position mismatch")
            if not self.block_tables[i] or any(x < 0 for x in self.block_tables[i]):
                raise ValueError("missing physical block")
            if row.phase == "decode":
                if seen_prefill or row.num_scheduled_tokens != 1 or not row.should_sample:
                    raise ValueError("decode rows must be first and one-token")
                decode_count += 1
            else:
                seen_prefill = True
            if row.should_sample:
                expected_indices.append(b - 1)
                expected_ids.append(row.seq_id)
        if decode_count != self.num_decode_tokens:
            raise ValueError("decode count mismatch")
        if tuple(expected_indices) != self.sample_indices or tuple(expected_ids) != self.sample_seq_ids:
            raise ValueError("sampling must exclude intermediate chunks")
        if len(self.temperatures) != len(self.sample_indices) or any(t <= 1e-10 for t in self.temperatures):
            raise ValueError("invalid sampling temperatures")
        if any(slot < 0 for slot in self.slot_mapping):
            raise ValueError("real requests cannot use padding slots")
        # A shared prefix is read-only; newly written slots must never alias.
        if len(set(self.slot_mapping)) != n:
            raise ValueError("two queries write the same physical KV slot")


def build_batch(rows, sequences, token_budget, block_size):
    ids, positions, slots, tables, offsets = [], [], [], [], [0]
    sample_indices, sample_ids, temperatures = [], [], []
    for row in rows:
        seq = sequences[row.seq_id]
        if seq.num_cached_tokens != row.start_pos or len(seq) != row.logical_tokens:
            raise RuntimeError("sequence changed after scheduling")
        ids.extend(seq.token_ids[row.start_pos:row.end_pos])
        positions.extend(range(row.start_pos, row.end_pos))
        tables.append(tuple(seq.block_table))
        for pos in range(row.start_pos, row.end_pos):
            slots.append(seq.block_table[pos // block_size] * block_size + pos % block_size)
        offsets.append(len(ids))
        if row.should_sample:
            sample_indices.append(len(ids) - 1)
            sample_ids.append(row.seq_id)
            temperatures.append(seq.temperature)
    num_decode = sum(r.phase == "decode" for r in rows)
    batch = BatchMetadata(
        tuple(rows), tuple(ids), tuple(positions), tuple(slots), tuple(tables),
        tuple(offsets), tuple(sample_indices), tuple(sample_ids), tuple(temperatures),
        num_decode, len(ids) - num_decode, token_budget,
    )
    batch.validate()
    return batch
