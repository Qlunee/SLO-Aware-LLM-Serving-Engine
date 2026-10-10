"""Explicit provisional BF16 engineering limits, not bitwise equivalence."""
import math


def numerical_pass(row, policy, relative_l2_limit=0.02, cosine_min=0.999,
                   max_abs_limit=0.5):
    if not row["top1_match"]:
        return False
    if policy == "strict":
        return row["allclose"]
    if policy != "bf16":
        raise ValueError("unknown numerical policy")
    values = [row[k] for k in ["relative_l2", "cosine", "max_abs_error"]]
    if not all(math.isfinite(v) for v in values):
        return False
    return (row["allclose"] or (row["relative_l2"] <= relative_l2_limit
            and row["cosine"] >= cosine_min
            and row["max_abs_error"] <= max_abs_limit))
