"""CPU contracts; capture metadata has no pickled Python engine objects."""
import hashlib
from pathlib import Path


def validate_plan(plan):
    block, total = plan["block_size"], plan["total_blocks"]
    if block <= 0 or total <= 0 or not plan["rows"]:
        raise ValueError("empty/invalid batch geometry")
    ids, positions, slots = (plan[k] for k in ("input_ids", "positions", "slots"))
    if not len(ids) == len(positions) == len(slots) or not ids:
        raise ValueError("flat tensor lengths differ")
    if len(set(slots)) != len(slots):
        raise ValueError("two queries overwrite the same KV slot")
    pages, offset, sample_ids, sample_indices = set(), 0, [], []
    row_ids = []
    for row in plan["rows"]:
        a, b, start, end = (row[k] for k in ("a", "b", "start", "end"))
        if a != offset or b-a != end-start or not 0 <= start < end <= len(row["history"]):
            raise ValueError("row span/history mismatch")
        if ids[a:b] != row["history"][start:end] or positions[a:b] != list(range(start, end)):
            raise ValueError("input tokens/positions disagree with logical history")
        table = row["table"]
        needed = (end+block-1)//block
        if (len(table) < needed or len(set(table)) != len(table)
                or any(type(p) is not int or not 0 <= p < total for p in table)):
            raise ValueError("invalid physical block table")
        expected = [table[p//block]*block+p % block for p in range(start, end)]
        if slots[a:b] != expected:
            raise ValueError("physical slots disagree with block table")
        if row["phase"] == "decode" and (end-start != 1 or end != len(row["history"])):
            raise ValueError("invalid decode row")
        if row["phase"] not in ("decode", "prefill", "recompute"):
            raise ValueError("unknown row phase")
        pages.update(table[:needed])
        if end == len(row["history"]):
            sample_ids.append(row["seq_id"])
            sample_indices.append(b-1)
        row_ids.append(row["seq_id"])
        offset = b
    if offset != len(ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("query coverage/duplicate sequence")
    if sample_ids != plan["sample_seq_ids"] or sample_indices != plan["sample_indices"]:
        raise ValueError("sampling row mapping differs")
    return sorted(pages)


def tool_hashes():
    folder = Path(__file__).resolve().parent
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".sh", ".md")}
    hashes["../__init__.py"] = hashlib.sha256((folder.parent/"__init__.py").read_bytes()).hexdigest()
    return hashes


def valid_prefix_slots(plan):
    validate_plan(plan)
    block = plan["block_size"]
    return sorted({row["table"][p//block]*block+p % block
                   for row in plan["rows"] for p in range(row["start"])})
