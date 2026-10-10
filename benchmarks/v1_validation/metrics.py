"""Distribution comparisons on the SAME token history, not two diverged generations."""
import math
import torch


def distribution_metrics(actual, reference, target=None, top_k=10):
    a = torch.as_tensor(actual).detach().double().cpu().flatten()
    r = torch.as_tensor(reference).detach().double().cpu().flatten()
    if a.shape != r.shape or a.numel() < 2:
        raise ValueError("logits must have the same vocabulary with at least two entries")
    if not torch.isfinite(a).all() or not torch.isfinite(r).all():
        raise ValueError("nonfinite logits")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be positive")
    log_a, log_r = a.log_softmax(0), r.log_softmax(0)
    log_m = torch.logaddexp(log_a, log_r) - math.log(2)
    pa, pr = log_a.exp(), log_r.exp()
    ac, rc = a - a.mean(), r - r.mean()
    k = min(top_k, a.numel())
    best = r.topk(2).values
    row = {
        "relative_l2": ((a-r).norm()/r.norm().clamp_min(1e-12)).item(),
        "max_abs_error": (a-r).abs().max().item(),
        "centered_relative_l2": ((ac-rc).norm()/rc.norm().clamp_min(1e-12)).item(),
        "kl_reference_to_actual": max(0., (pr*(log_r-log_a)).sum().item()),
        "js_divergence": max(0., .5*((pa*(log_a-log_m)).sum()+(pr*(log_r-log_m)).sum()).item()),
        "total_variation": .5*(pa-pr).abs().sum().item(),
        "top1_match": a.argmax().item() == r.argmax().item(),
        "reference_argmax_margin": (best[0]-best[1]).item(),
        "top_k": k,
        "top_k_overlap": len(set(a.topk(k).indices.tolist()) & set(r.topk(k).indices.tolist()))/k,
    }
    if target is not None:
        if not isinstance(target, int) or not 0 <= target < a.numel():
            raise ValueError("invalid target token")
        row.update(target_token=target, actual_target_nll=-log_a[target].item(),
                   reference_target_nll=-log_r[target].item(),
                   target_nll_delta=(log_r[target]-log_a[target]).item())
    return row


def sequence_metrics(actual, reference):
    if not actual or not reference:
        raise ValueError("empty completion")
    common = 0
    for a, r in zip(actual, reference):
        if a != r:
            break
        common += 1
    exact = list(actual) == list(reference)
    return {"exact_match": exact, "actual_tokens": len(actual), "reference_tokens": len(reference),
            "common_prefix_tokens": common,
            "first_divergence_index": None if exact else common,
            "position_match_rate": sum(a == r for a, r in zip(actual, reference))/max(len(actual), len(reference))}


def summarize(rows):
    if not rows:
        raise ValueError("no distribution observations")
    result = {"samples": len(rows), "top1_match_rate": sum(r["top1_match"] for r in rows)/len(rows)}
    for name in ("relative_l2", "centered_relative_l2", "js_divergence", "kl_reference_to_actual",
                 "total_variation", "top_k_overlap", "target_nll_delta"):
        values = [float(r[name]) for r in rows if name in r]
        if not values:
            continue
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite summary")
        tensor = torch.tensor(values, dtype=torch.float64)
        result[name] = {"mean": tensor.mean().item(), "p95": tensor.quantile(.95).item(), "max": tensor.max().item()}
    if "target_nll_delta" in result:
        delta = result["target_nll_delta"]["mean"]
        result["sampled_conditional_ppl_ratio"] = math.exp(delta) if delta < 700 else None
    return result
