import numpy as np

from records import INTERVALS, PAPER_FACTOR


def metrics(prediction, truth, valid, bearings, *, scale=PAPER_FACTOR, common=False):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    bearings = np.asarray(bearings)
    if (
        prediction.shape != truth.shape
        or valid.shape != truth.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 5
        or bearings.shape != (len(prediction),)
    ):
        raise ValueError("Expected matching five-interval arrays and one bearing ID per row")
    if not np.isfinite(prediction).all() or not np.isfinite(truth[valid]).all():
        raise ValueError("Invalid forecast or observed target")
    if common:
        valid = np.broadcast_to(valid.all(1, keepdims=True), valid.shape)
    rows = []
    for i in np.unique(bearings):
        for h, (a, b) in enumerate(INTERVALS):
            keep = (bearings == i) & valid[:, h]
            y, p = truth[keep, h], prediction[keep, h]
            n = len(y)
            eligible = bool(n >= 3 and y.std() > 1e-10 * scale)
            r = float(np.corrcoef(y, p)[0, 1]) if eligible and p.std() > 1e-10 * scale else None
            error = p - y
            rows.append(
                dict(
                    bearing=int(i),
                    start_s=a,
                    end_s=b,
                    n=n,
                    pearson_r=r,
                    mae=float(np.abs(error).mean()) if n else None,
                    mse=float(np.square(error).mean()) if n else None,
                    selection_r=(r if r is not None else 0.0) if eligible else None,
                )
            )
    return rows


def bearing_means(rows):
    output = []
    for a, b in INTERVALS:
        group = [row for row in rows if row["start_s"] == a and row["end_s"] == b]
        stats = {}
        for k in ("pearson_r", "mae", "mse", "selection_r"):
            values = [r[k] for r in group if r[k] is not None]
            stats[k] = float(np.mean(values)) if values else None
        output.append(
            dict(start_s=a, end_s=b, n_bearings=sum(row["n"] > 0 for row in group), **stats)
        )
    return output


def validation_score(pred, truth, valid, ids, sigma):
    weights = [0.30, 0.30, 0.10, 0.15, 0.15]
    groups = [
        bearing_means(metrics(pred, truth, valid, ids, scale=1.0, common=c)) for c in (False, True)
    ]

    def weighted(name):
        if any(row[name] is None for group in groups for row in group):
            raise ValueError(f"Validation lacks an eligible bearing for {name}")
        return np.mean([np.dot(weights, [r[name] for r in group]) for group in groups])

    return float(weighted("selection_r")), float(
        0.5 * (weighted("mae") / sigma + np.sqrt(weighted("mse")) / sigma)
    )
