import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from metrics import bearing_means, metrics
from prediction import evaluate_records, load_model, specification, write_predictions
from records import INTERVALS, load_record, prepare
from training import fit_split


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def paper_evaluation(data, destination, group, device, batch):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    groups = [] if group == "new" else [(i, specification(i)["test"]) for i in range(1, 6)]
    if group != "original":
        groups.append((0, list(range(48, 52))))
    summary, individual = [], []
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        out = Path(staging) / "metrics"
        out.mkdir()
        for index, ids in groups:
            model, state = load_model(index or 1, device)
            records = {i: load_record(Path(data) / f"No{i}.npz", True, i) for i in ids}
            win, p, y, valid = evaluate_records(model, records, state, device, batch)
            scores = metrics(p, y, valid, win.ids)
            group_name = "original" if index else "new"
            for rows, target in ((scores, individual), (bearing_means(scores), summary)):
                for row in rows:
                    fields = {k: v for k, v in row.items() if k != "selection_r"}
                    target.append(dict(group=group_name, split=index or 1, **fields))
        write_rows(out / "per_bearing.csv", individual)
        write_rows(out / "equal_bearing_means.csv", summary)
        if group != "new":
            across = []
            for a, b in INTERVALS:
                subset = [r for r in summary if r["group"] == "original" and r["start_s"] == a]
                stats = {}
                for metric in ("pearson_r", "mae", "mse"):
                    values = [r[metric] for r in subset]
                    if any(v is None for v in values):
                        raise ValueError(
                            f"Five-split summary unavailable for {metric} at {a}-{b} s"
                        )
                    stats[metric + "_mean"] = float(np.mean(values))
                    stats[metric + "_sd"] = float(np.std(values, ddof=1))
                across.append(dict(start_s=a, end_s=b, **stats))
            write_rows(out / "five_split_means_and_sd.csv", across)
        if destination.exists():
            raise FileExistsError(destination)
        out.rename(destination)


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Use a positive integer")
    return n


def main():
    p = argparse.ArgumentParser(description="C-S60 bearing friction-coefficient forecasts")
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="Prepare one integrated acquisition CSV")
    prep.add_argument("--csv", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument(
        "--bearing", type=positive, help="Optional identifier for an unnumbered input filename"
    )
    prep.add_argument(
        "--ae-only", action="store_true", help="Read only the AE column, without force measurements"
    )
    train = sub.add_parser("train", help="Fit three models for one paper split")
    pred = sub.add_parser(
        "predict", help="Forecast one record with a frozen or locally refitted ensemble"
    )
    ev = sub.add_parser("evaluate", help="Recompute paper C-S60 metrics from prepared records")
    for subcommand in (train, pred, ev):
        subcommand.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        subcommand.add_argument("--threads", type=positive, default=4)
        subcommand.add_argument(
            "--batch-size", type=positive, default=256 if subcommand is train else 128
        )
        subcommand.add_argument("--output", type=Path, required=True)
    for subcommand in (train, pred):
        subcommand.add_argument("--split", type=int, choices=range(1, 6), default=1)
    for subcommand in (train, ev):
        subcommand.add_argument(
            "--data", type=Path, required=True, help="Directory containing No1.npz, No2.npz, ..."
        )
    train.add_argument("--retune", action="store_true")
    train.add_argument("--epochs", type=positive, default=160)
    train.add_argument("--samples-per-epoch", type=positive, default=8192)
    train.add_argument("--patience", type=positive, default=24)
    pred.add_argument("--record", type=Path, required=True)
    pred.add_argument("--bearing", type=positive, required=True)
    pred.add_argument("--fitted", type=Path, help="Completed output from the train command")
    ev.add_argument("--group", choices=("all", "original", "new"), default="all")
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.csv, args.output, args.ae_only, args.bearing)
        return
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is unavailable")
    if args.cmd == "train":
        fit_split(
            args.data,
            args.split,
            args.output,
            device=args.device,
            retune=args.retune,
            epochs=args.epochs,
            patience=args.patience,
            samples=args.samples_per_epoch,
            batch_size=args.batch_size,
        )
    elif args.cmd == "predict":
        model, state = load_model(args.split, args.device, args.fitted)
        r = load_record(args.record, expected_bearing=args.bearing)
        win, prediction, truth, valid = evaluate_records(
            model, {args.bearing: r}, state, args.device, args.batch_size
        )
        write_predictions(args.output, win, prediction, truth, valid)
    else:
        paper_evaluation(args.data, args.output, args.group, args.device, args.batch_size)


if __name__ == "__main__":
    main()
