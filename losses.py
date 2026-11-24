import math
import torch
from torch.nn import functional as F


def grouped_moments(prediction, truth, importance, ids):

    if prediction.shape != truth.shape or truth.shape != importance.shape:
        raise ValueError("Grouped arrays must have matching (batch,horizon) shapes")
    if prediction.ndim != 2 or ids.ndim != 1 or len(ids) != len(prediction):
        raise ValueError("Invalid grouped metadata shapes")
    groups = torch.unique(ids, sorted=True)
    membership = (groups[:, None] == ids[None, :]).to(prediction.dtype)
    valid = (importance > 0) & torch.isfinite(importance) & torch.isfinite(truth)
    weight = torch.where(valid, importance, 0.0)
    p, y = torch.where(valid, prediction, 0.0), torch.where(valid, truth, 0.0)
    total = membership @ weight
    denominator = total.clamp_min(torch.finfo(prediction.dtype).tiny)
    count = membership @ valid.to(prediction.dtype)
    pm, ym = (membership @ (weight * p)) / denominator, (membership @ (weight * y)) / denominator
    pc, yc = p - membership.T @ pm, y - membership.T @ ym
    pv = (membership @ (weight * pc.square())) / denominator
    yv = (membership @ (weight * yc.square())) / denominator
    cov = (membership @ (weight * pc * yc)) / denominator
    return pm, ym, pv, yv, cov, count


def _group_reduce(error, eligible, prediction, horizon_weights):
    weights = tuple(float(v) for v in horizon_weights)
    if len(weights) != prediction.shape[1] or any(not math.isfinite(v) or v < 0 for v in weights):
        raise ValueError("Invalid horizon weights")
    error = torch.where(eligible, error, 0.0)
    count = eligible.sum(0)
    active = prediction.new_tensor(weights) * (count > 0)
    return ((error.sum(0) / count.clamp_min(1)) * active).sum() / active.sum().clamp_min(1e-12)


def level_correlation_loss(prediction, truth, importance, ids, horizon_weights):

    _, _, pv, yv, cov, count = grouped_moments(prediction, truth, importance, ids)
    epsilon2 = 0.01**2
    agreement = cov / ((pv + epsilon2) * (yv + epsilon2)).sqrt()
    eligible = (count >= 3) & (yv > 1e-8)
    return _group_reduce(1.0 - agreement.clamp(-1.0, 1.0), eligible, prediction, horizon_weights)


def masked_huber(prediction, truth, importance):
    valid = (importance > 0) & torch.isfinite(importance) & torch.isfinite(truth)
    p = torch.where(valid, prediction, 0.0)
    y = torch.where(valid, truth, 0.0)
    weight = torch.where(valid, importance, 0.0)
    return F.huber_loss(p, y, delta=1.0, reduction="none") * weight


def objective(output, batch, state):
    weights = (0.45, 0.45, 0.05, 0.025, 0.025)
    fine = masked_huber(output["fine"], batch["fine"], batch["fine_iw"][:, None]).mean(1).mean()
    change = (
        (
            masked_huber(output["delta"], batch["delta"], batch["delta_iw"])
            * output["delta"].new_tensor([0.4, 0.35, 0.25])
        )
        .sum(1)
        .mean()
    )
    short = masked_huber(output["short"], batch["short"], batch["short_iw"]).mean(1).mean()
    pred = output["five"]
    level = (
        (masked_huber(pred, batch["five"], batch["five_iw"]) * pred.new_tensor(weights))
        .sum(1)
        .mean()
    )
    correlation = level_correlation_loss(
        pred, batch["five"], batch["five_iw"], batch["ids"], weights
    )
    far = (pred[:, 4:5] - pred[:, 3:4]) * state["target_std"] / state["far_scale"]
    far_corr = level_correlation_loss(
        far, batch["far"][:, None], batch["far_iw"][:, None], batch["ids"], (1.0,)
    )
    return 0.10 * fine + 0.20 * level + correlation + 0.10 * change + 0.20 * short + 0.01 * far_corr
