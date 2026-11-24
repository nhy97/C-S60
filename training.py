import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from losses import objective
from metrics import validation_score
from model import CS60
from prediction import Windows, forecast, specification
from records import fine_targets, labels, load_record, means, KINDS


def probability(ids, mask):
    mask = np.asarray(mask, bool)
    if mask.ndim == 2:
        return np.column_stack([probability(ids, v) for v in mask.T])
    result = np.zeros(len(ids), np.float64)
    for i in np.unique(ids):
        keep = (ids == i) & mask
        if keep.any():
            result[keep] = 1.0 / keep.sum()
    return result / result.sum() if result.any() else result


def training_state(records):
    moments, squares, fine_means, fine_squares = [], [], [], []
    changes, valid_changes, far, valid_far, ids = [], [], [], [], []
    for i, r in records.items():
        x = r["ae"].astype(np.float64)
        moments.append(x.mean(0))
        squares.append(np.square(x).mean(0))
        ends = np.arange(60, len(x) + 1) * 10
        v = np.abs(r["voltage"])
        y, good = fine_targets(v, ends)
        if not good.any():
            raise ValueError(f"Bearing {i} has no eligible fine training target")
        fine_means.append(float(y[good].mean()))
        fine_squares.append(np.square(y[good].astype(np.float64)).mean())
        old, mask = means(v, ends, KINDS["coarse"])
        changes.append(old[:, 1:] - old[:, :1])
        valid_changes.append(mask[:, 1:] & mask[:, :1])
        f, ok = labels(r["voltage"], ends)["far"]
        far.append(f)
        valid_far.append(ok)
        ids.append(np.full(len(ends), i))
    ids = np.concatenate(ids)
    feature_mean = np.mean(moments, axis=0)
    feature_std = np.sqrt(np.maximum(np.mean(squares, axis=0) - feature_mean**2, 0))
    feature_std[feature_std < 1e-6] = 1.0
    target_mean = np.mean(fine_means)
    target_std = max(np.sqrt(max(np.mean(fine_squares) - target_mean**2, 0)), 1e-6)
    diffs = np.concatenate(changes).astype(np.float64)
    valid = np.concatenate(valid_changes)
    if not valid.any(axis=0).all():
        raise ValueError("Training needs complete targets for each original difference")
    scales = np.sqrt((probability(ids, valid) * diffs**2).sum(0))
    far = np.concatenate(far).astype(np.float64)
    far_rms = np.sqrt(np.dot(probability(ids, np.concatenate(valid_far)), far * far))
    return dict(
        features=dict(mean=feature_mean.tolist(), std=feature_std.tolist()),
        target_mean=float(target_mean / 100),
        target_std=float(target_std / 100),
        delta_scale=(np.maximum(scales, 1e-6) / 100).tolist(),
        far_scale=float(max(far_rms, 1e-8)),
    )


class TrainingWindows(Dataset):
    def __init__(self, records, state):
        self.windows = Windows(records, state)
        self.state = state
        items = [
            labels(r["voltage"], self.windows.ends[self.windows.ids == b] * 10)
            for b, r in records.items()
        ]
        raw = {k: np.concatenate([i[k][0] for i in items]) for k in items[0]}
        valid = {k: np.concatenate([i[k][1] for i in items]) for k in items[0]}
        probs = {k: probability(self.windows.ids, v) for k, v in valid.items()}
        self.q = (probs["fine"] + probs["coarse"].mean(1) + probs["delta"].mean(1)) / 3
        if self.q.sum() <= 0 or not np.isclose(self.q.sum(), 1):
            raise ValueError("Training sample probabilities do not sum to one")
        self.fields = {}
        for name in ("fine", "short", "five", "delta", "far"):
            p = probs[name]
            q = self.q if p.ndim == 1 else self.q[:, None]
            if np.any((p > 0) & (q <= 0)):
                raise ValueError(f"Unsupported target probability in {name}")
            iw = np.divide(p, q, out=np.zeros_like(p), where=q > 0)
            self.fields[name + "_iw"] = iw.astype(np.float32)
            if name == "delta":
                y = raw[name] / np.asarray(state["delta_scale"], np.float32)
            elif name == "far":
                y = raw[name] / state["far_scale"]
            else:
                y = (raw[name] - state["target_mean"]) / state["target_std"]
            self.fields[name] = y.astype(np.float32)
        self.fields["ids"] = np.concatenate(
            [
                np.full(np.count_nonzero(self.windows.ids == name), index, dtype=np.int64)
                for index, name in enumerate(self.windows.names)
            ]
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, k):
        return self.windows[k], {name: field[k] for name, field in self.fields.items()}


def choose(candidates):
    peak = max(s for s, e, k in candidates)
    eligible = [row for row in candidates if row[0] >= peak - 0.002 - 1e-12]
    return min(eligible, key=lambda r: (r[1], -r[0], r[2])), peak


def fit_member(
    train, val_records, state, seed, rate, device, epochs, patience, samples, batch_size
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CS60(state).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=0.03)
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(train.q, dtype=torch.double), samples, replacement=True, generator=gen
    )
    val = Windows(val_records, state)
    targets = [
        labels(r["voltage"], val.ends[val.ids == b] * 10)["five"] for b, r in val_records.items()
    ]
    y, valid = [np.concatenate(parts) for parts in zip(*targets)]
    history, retained = [], {}
    previous, bad = None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, batch in DataLoader(
            train, batch_size=batch_size, sampler=sampler, pin_memory=device == "cuda"
        ):
            fields = {k: value.to(device) for k, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x.to(device)), fields, state)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite model loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        predictions = forecast(model, val, state, device, batch_size)
        score, error = validation_score(predictions, y, valid, val.ids, state["target_std"])
        history.append((score, error, epoch))
        best, peak = choose(history)
        if score >= peak - 0.002 - 1e-12:
            retained[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        retained = {
            k: params for k, params in retained.items() if history[k - 1][0] >= peak - 0.002 - 1e-12
        }
        current = (peak, best[2])
        bad = bad + 1 if current == previous else 0
        previous = current
        print(
            f"seed {seed} epoch {epoch} validation r {score:.6f} normalized error {error:.6f}",
            flush=True,
        )
        if bad >= patience:
            break
    best, _ = choose(history)
    return retained[best[2]], best, history


def fit_split(
    record_dir,
    split,
    destination,
    *,
    device="cpu",
    retune=False,
    epochs=160,
    patience=24,
    samples=8192,
    batch_size=256,
):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if min(epochs, patience, samples, batch_size) <= 0:
        raise ValueError("Positive training limits are required")
    spec = specification(split)
    if set(spec["train"]) & set(spec["validation"]):
        raise ValueError("Training and validation overlap")
    names = spec["train"] + spec["validation"]
    records = {i: load_record(Path(record_dir) / f"No{i}.npz", True, i) for i in names}

    def subset(indices):
        return {i: records[i] for i in indices}

    selected = spec["members"][0]["learning_rate"]
    tuning = {}
    if retune:
        rates = (1e-5, 3e-5, 1e-4)
        tuning = {rate: [] for rate in rates}
        for fold in spec["inner"]:
            if set(fold["train"]) & set(fold["validation"]) or set(
                fold["train"] + fold["validation"]
            ) != set(spec["train"]):
                raise ValueError("Invalid inner bearing split")
            inner = subset(fold["train"])
            scale = training_state(inner)
            training = TrainingWindows(inner, scale)
            for rate in rates:
                _, best, _ = fit_member(
                    training,
                    subset(fold["validation"]),
                    scale,
                    2021,
                    rate,
                    device,
                    24,
                    8,
                    8192,
                    256,
                )
                tuning[rate].append(best[:2])
        candidates = [(*np.mean(tuning[rate], axis=0), rate) for rate in rates]
        selected = choose(candidates)[0][2]
    training = subset(spec["train"])
    state = training_state(training)
    windows = TrainingWindows(training, state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        out = Path(scratch) / "ensemble"
        out.mkdir()
        (out / "normalization.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        reports = []
        for i, member in enumerate(spec["members"], 1):
            seed = member["seed"]
            weights, best, history = fit_member(
                windows,
                subset(spec["validation"]),
                state,
                seed,
                selected,
                device,
                epochs,
                patience,
                samples,
                batch_size,
            )
            torch.save(weights, out / f"member_{i}.pt")
            reports.append(dict(seed=seed, selected_epoch=best[2], history=history))
        (out / "training.json").write_text(
            json.dumps(
                dict(
                    split=split,
                    rate=selected,
                    retune=retune,
                    tuning=tuning,
                    epochs=epochs,
                    patience=patience,
                    samples_per_epoch=samples,
                    batch_size=batch_size,
                    members=reports,
                ),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(destination)
        out.rename(destination)
