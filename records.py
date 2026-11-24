import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from features import STEP, absolute_features, extract


PAPER_FACTOR = 2.0 * 190.0 / 16.75
INTERVALS = ((0, 5), (5, 15), (15, 30), (30, 45), (45, 60))
KINDS = {
    "five": INTERVALS,
    "short": ((0, 5), (5, 10), (10, 15)),
    "coarse": ((0, 5), (5, 15), (15, 30), (30, 60)),
}


@contextmanager
def new_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as tmp:
        temp = Path(tmp.name)
    try:
        yield temp
        if os.name == "nt":
            temp.rename(path)
        else:
            os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def prepare(csv_file, destination, ae_only=False, bearing=None):
    matched = re.fullmatch(r"(?:t|no[._ -]?)([1-9][0-9]*)", Path(csv_file).stem, re.I)
    if matched:
        inferred = int(matched.group(1))
        if bearing is not None and bearing != inferred:
            raise ValueError("The bearing ID differs from the acquisition filename")
        bearing = inferred
    columns = [3] if ae_only else [2, 3]
    pending = np.empty((0, len(columns)), np.float64)
    skipped = 0
    frames = 0
    distributions, voltages = [], []
    with pd.read_csv(
        csv_file,
        header=None,
        usecols=columns,
        encoding="utf-8-sig",
        dtype=np.float64,
        chunksize=STEP * 32,
    ) as reader:
        for block in reader:
            samples = np.concatenate((pending, block.to_numpy(dtype=np.float64)))
            if not np.isfinite(samples).all():
                raise ValueError("Nonfinite acquisition sample")
            stop = len(samples) // STEP * STEP
            pending = samples[stop:]
            batch = samples[:stop].reshape(-1, STEP, len(columns))
            skip = min(2 - skipped, len(batch))
            skipped += skip
            batch = batch[skip:]
            if len(batch):
                frames += len(batch)
                distributions.append(extract(batch[:, :, -1].astype(np.float32)))
                if not ae_only:
                    voltages.append(np.median(batch[:, -250:, 0], axis=1).astype(np.float32))
    if frames < 600:
        raise ValueError("Fewer than 60 complete seconds remain after removing the initial 0.2 s")
    slow = absolute_features(np.concatenate(distributions))
    arrays = {"ae": slow, "time": (np.arange(frames, dtype=np.float64) + 3) / 10}
    if bearing is not None:
        arrays["bearing"] = np.int64(bearing)
    if voltages:
        arrays["voltage"] = np.concatenate(voltages)
    with new_output(destination) as temp, temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_record(path, require_voltage=False, expected_bearing=None):
    with np.load(path, allow_pickle=False) as data:
        if (
            expected_bearing is not None
            and "bearing" in data
            and int(data["bearing"]) != expected_bearing
        ):
            raise ValueError(f"Bearing ID differs from the requested ID in {path}")
        ae = data["ae"].astype(np.float32)
        time = data["time"].astype(np.float64)
        voltage = data["voltage"].astype(np.float32) if "voltage" in data else None
    if ae.ndim != 2 or ae.shape[1] != 52 or time.ndim != 1 or len(time) // 10 != len(ae):
        raise ValueError(f"Misaligned AE frame or timestamp dimensions in {path}")
    if len(ae) < 60 or not np.isfinite(ae).all() or not np.isfinite(time).all():
        raise ValueError(f"Invalid or short record {path}")
    if not np.allclose(np.diff(time), 0.1, atol=1e-6, rtol=0):
        raise ValueError(f"Expected equally spaced frame-end times in {path}")
    if voltage is not None and (voltage.shape != time.shape or not np.isfinite(voltage).all()):
        raise ValueError(f"Invalid voltage in {path}")
    if require_voltage and voltage is None:
        raise ValueError(
            f"Training and labelled evaluation require signed force-gauge voltage in {path}"
        )
    return {"ae": ae, "time": time, "voltage": voltage}


def means(values, ends, intervals):
    prefix = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    result = np.zeros((len(ends), len(intervals)), np.float32)
    mask = np.zeros_like(result, dtype=bool)
    for h, (a, b) in enumerate(intervals):
        start, stop = ends + 10 * a, ends + 10 * b
        keep = stop <= len(values)
        mask[:, h] = keep
        result[keep, h] = (prefix[stop[keep]] - prefix[start[keep]]) / (10 * (b - a))
    return result, mask


def fine_targets(values, ends):
    keep = ends + 50 <= len(values)
    out = np.zeros((len(ends), 5), np.float32)
    out[keep] = values[ends[keep, None] + np.arange(50)].reshape(-1, 5, 10).mean(2)
    return out, keep


def labels(voltage, ends):
    magnitude = np.abs(voltage)
    archived = (magnitude.astype(np.float64) / 100).astype(np.float32)
    result = {k: means(archived, ends, spans) for k, spans in KINDS.items()}
    result["fine"] = fine_targets(archived, ends)
    coarse, valid = result["coarse"]
    result["delta"] = (coarse[:, 1:] - coarse[:, :1], valid[:, 1:] & valid[:, :1])
    five, valid = result["five"]
    result["far"] = (five[:, 4] - five[:, 3], valid.all(1))
    return result
