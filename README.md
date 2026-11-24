# C-S60

C-S60 forecasts the mean friction coefficient of a thrust ball bearing from its most recent 60 s of acoustic emission (AE). Each prediction covers five future intervals in seconds: 0–5, 5–15, 15–30, 30–45, and 45–60. The package contains the preprocessing, model, training and evaluation code, the paper's bearing splits, and the five frozen three-model ensembles used in the manuscript. The raw bearing records are supplied separately by the authors according to their data-access arrangements.

## Set up

Use Python 3.11 or later. From this directory, run:

```text
python -m pip install -r requirements.txt
python run.py --help
```

Inference defaults to CPU. Use `--device cuda` with a CUDA-enabled PyTorch installation for GPU inference or training. The archived predictions were checked on CPU with Python 3.11.14, NumPy 2.4.0rc1, pandas 2.3.3 and PyTorch 2.11.0.dev20251216+cu128. Different hardware and library builds can affect the last reported digits. Each command expects a new output file or directory.

## Prepare a record

The input is the study's synchronized, headerless four-column CSV for one complete bearing run. The columns are the acquisition date and time to the second, the microsecond field, signed analog force-gauge voltage in V, and AE voltage in V. Supply a continuous 50 kHz sample sequence already aligned in time. The program constructs elapsed time from sample count. Vendor exports with gaps or separate force and AE clocks must first be aligned and checked. Supply the gauge channel in volts before any force or friction-coefficient conversion.

```text
python run.py prepare --csv raw/t48.csv --output prepared/No48.npz
```

The first two 0.1 s frames and a final incomplete frame are discarded. Each 0.1 s frame supplies AE features and the median of the last 250 entries in the force-voltage column. Ten complete frames produce one 52-feature AE vector. The prepared NumPy archive contains `ae` (seconds × 52), `time` (elapsed frame-end times in seconds) and, for labelled records, `voltage` (signed V at 10 Hz). Bearing identity is read from a filename such as `t48.csv` or `No48.csv`. Use `--bearing 48` for an unnumbered input. For group evaluation, name the prepared files `No1.npz` through `No51.npz` as applicable.

## Predict with the paper weights

The AE-only path reads the fourth CSV column and bypasses the force channel entirely. This is also the path for forecasting without measured targets.

```text
python run.py prepare --csv raw/t48.csv --output prepared/No48_AE.npz --ae-only
python run.py predict --split 1 --bearing 48 --record prepared/No48_AE.npz --output predictions.csv
```

The first decision is at 60.2 s from the start of the acquisition. One decision is issued per subsequent second, using only completed AE observations. Each output row gives the bearing ID, decision time, interval bounds and predicted dimensionless friction coefficient. When a labelled record is supplied, the CSV also includes the measured interval mean wherever the whole interval was observed. With AE alone that column is empty.

For the apparatus described in the paper, the magnitude of the force-gauge reaction is twice the absolute gauge voltage in newtons. The effective lever arm is 190 mm, the mean bearing radius is 16.75 mm, and the axial force is 100 N. The reported friction coefficient is therefore `abs(V) × 2 × 190 / (16.75 × 100)`. The frozen models were fitted on the target scale `abs(V) / 100`. The code reverses target standardization and multiplies by `2 × 190 / 16.75` once before writing measured and predicted coefficients or computing the reported errors.

## Evaluate the paper configuration

Prepare the required labelled records first. For the four additional bearings, Nos. 48–51, the same split-1 ensemble and its original normalization are used throughout.

```text
python run.py evaluate --group new --data prepared --output results-new
python run.py evaluate --group all --data prepared --output results-all
```

Use `--group original` for just the five held-out test sets from the original 47 bearings. The output includes metrics per bearing and equal-bearing means for each split and future interval. The five-split summary also gives the mean and sample standard deviation across the five split-level means. Pearson correlation, mean absolute error (MAE) and mean squared error (MSE) are computed at decision times with complete measured targets for the respective interval. An undefined Pearson correlation has an empty CSV field. MAE and MSE are written in the physical coefficient scale without the display multipliers used in some manuscript tables. Compare `equal_bearing_means.csv` with `reference_metrics.csv`, which contains the corresponding results calculated from the archived predictions.

## Refit an ensemble

Use the original training and validation bearings listed in `splits.json`. The training command reads only the records from these two groups and produces three member weight files, the normalization parameters and a training log. The default reuses the selected learning rate for that split. The optional `--retune` repeats selection over the three recorded inner folds before fitting the three ensemble members.

```text
python run.py train --split 1 --data prepared --device cuda --output fitted-split-1
python run.py predict --split 1 --bearing 48 --record prepared/No48.npz --fitted fitted-split-1 --output refitted-predictions.csv
python run.py train --split 1 --data prepared --device cuda --retune --output retuned-split-1
```

Each final member uses its recorded seed with up to 160 epochs, patience of 24 epochs, 8192 sampled windows per epoch and a batch size of 256. The paper results use the supplied frozen ensembles. Retraining produces a separate set of weights.

## File guide

| File | Contents |
| --- | --- |
| `run.py` | Command-line entry points |
| `records.py`, `features.py` | AE features, input windows' time base and voltage-derived targets |
| `model.py`, `layers.py` | C-S60 model, prediction heads and ensemble |
| `training.py`, `losses.py` | Bearing-balanced sampling, losses, validation and fitting |
| `prediction.py`, `metrics.py` | Frozen prediction, interval masks and bearing-level metrics |
| `splits.json` | Bearing IDs, inner splits, seeds, selected epochs, learning rates and checkpoint hashes |
| `checkpoints/split_1` through `split_5` | Frozen ensemble and corresponding normalization for each split |
| `reference_metrics.csv` | Thirty sets of interval metrics for comparison with a local run |

The independent PatchTST benchmark used for comparison in the manuscript has its own [upstream implementation](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9).

