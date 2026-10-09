"""CPU-only accounting shared by observation and regression tests."""


def merge_interval(history, start, end):
    """Count actually re-executed positions and merge their interval union."""
    if start < 0 or end < start:
        raise ValueError("Invalid executed interval")
    if start == end:
        return 0, list(history)
    overlap = sum(max(0, min(end, b) - max(start, a)) for a, b in history)
    merged = []
    for a, b in sorted([*history, (start, end)]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return overlap, merged
