import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import Ensemble, read_archived_weights
from records import INTERVALS, PAPER_FACTOR, labels, new_output


ROOT = Path(__file__).resolve().parent


def specification(split):
    if split not in range(1, 6):
        raise ValueError("Select a paper split from 1 to 5")
    return json.loads((ROOT / "splits.json").read_text(encoding="utf-8"))[split - 1]


def load_model(split, device="cpu", fitted=None):
    archive = ROOT / "checkpoints" / f"split_{split}"
    folder = Path(fitted) if fitted else archive
    state = json.loads((folder / "normalization.json").read_text(encoding="utf-8"))
    model = Ensemble(state)
    if fitted:
        run = json.loads((folder / "training.json").read_text(encoding="utf-8"))
        if run["split"] != split or len(run["members"]) != 3:
            raise ValueError("The fitted ensemble does not correspond to the requested split")
        for index, member in enumerate(model.members):
            path = folder / f"member_{index+1}.pt"
            member.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )
    else:
        path = folder / "ensemble.pt"
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != specification(split)["sha256"]:
                raise ValueError(f"Unexpected ensemble checkpoint for split {split}")
        read_archived_weights(model, path)
    return model.to(device).eval(), state


class Windows(Dataset):
    def __init__(self, records, state):
        self.names = list(records)
        mean, std = (np.asarray(state["features"][k], np.float32) for k in ("mean", "std"))
        if (
            mean.shape != (52,)
            or std.shape != mean.shape
            or np.any(std <= 0)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
        ):
            raise ValueError("Invalid training-set feature normalization")
        self.arrays, ids, ends, times = [], [], [], []
        for n, record in records.items():
            x = (record["ae"] - mean) / std
            e = np.arange(60, len(x) + 1)
            self.arrays.append(x)
            ends.append(e)
            ids.append(np.full(len(e), n))
            times.append(record["time"][10 * e - 1])
        self.ids, self.ends, self.times = map(np.concatenate, (ids, ends, times))
        self.indices = {name: i for i, name in enumerate(self.names)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        end = self.ends[k]
        return self.arrays[self.indices[self.ids[k]]][end - 60 : end]


@torch.inference_mode()
def forecast(model, windows, state, device, batch_size=128):
    model.eval()
    out = []
    for history in DataLoader(windows, batch_size=batch_size):
        y = model(history.to(device))
        y = y["five"] if isinstance(y, dict) else y
        if not torch.isfinite(y).all():
            raise FloatingPointError("Nonfinite interval forecast")
        out.append(y.cpu().numpy())
    return np.concatenate(out) * state["target_std"] + state["target_mean"]


def evaluate_records(model, records, state, device="cpu", batch_size=128):
    windows = Windows(records, state)
    pred = forecast(model, windows, state, device, batch_size).astype(np.float64) * PAPER_FACTOR
    truth, valid = [], []
    for bearing, rec in records.items():
        ends = windows.ends[windows.ids == bearing] * 10
        if rec["voltage"] is None:
            truth.append(np.zeros((len(ends), 5)))
            valid.append(np.zeros((len(ends), 5), bool))
        else:
            y, v = labels(rec["voltage"], ends)["five"]
            truth.append(y.astype(np.float64) * PAPER_FACTOR)
            valid.append(v)
    return windows, pred, np.concatenate(truth), np.concatenate(valid)


def write_predictions(path, windows, pred, truth, valid):
    with new_output(path) as temp, temp.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "bearing",
                "prediction_time_s",
                "interval_start_s",
                "interval_end_s",
                "predicted_mu",
                "measured_mu",
            ]
        )
        for k, name in enumerate(windows.ids):
            for h, (a, b) in enumerate(INTERVALS):
                out.writerow(
                    [
                        int(name),
                        windows.times[k],
                        a,
                        b,
                        pred[k, h],
                        truth[k, h] if valid[k, h] else "",
                    ]
                )
