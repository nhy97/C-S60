import numpy as np

FS = 50_000
STEP = 5_000
ABSOLUTE_DIM = 52
HIST_EDGES = np.r_[-np.inf, np.linspace(-5.0, 5.0, 65)[1:-1], np.inf]
FFT_GROUPS = np.array_split(np.arange(STEP // 2 + 1), 128)
BAND_CENTRES_NYQUIST = np.asarray([(g[0] + g[-1]) / 2.0 / 2500.0 for g in FFT_GROUPS])


def extract(ae):

    x = np.asarray(ae, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != STEP or not np.isfinite(x).all():
        raise ValueError("Expected finite AE frames [n,5000]")
    m = x.mean(1)
    c = x - m[:, None]
    std = np.sqrt(np.mean(c * c, 1))
    den = np.maximum(std, 1e-12)
    stats = np.column_stack(
        (
            m,
            np.sqrt(np.mean(x * x, 1)),
            std,
            np.max(np.abs(x), 1),
            np.ptp(x, axis=1),
            np.max(np.abs(c), 1) / den,
            np.mean((c / den[:, None]) ** 3, 1),
            np.mean((c / den[:, None]) ** 4, 1) - 3,
        )
    )

    u = np.sign(x) * np.log10(1 + np.abs(x) / 0.001)
    hist = np.stack([np.histogram(row, HIST_EDGES)[0] / STEP for row in u])
    w = np.hanning(STEP)
    ps = np.abs(np.fft.rfft(c * w, axis=1)) ** 2 / (FS * np.sum(w * w))
    ps[:, 1:-1] *= 2
    power = np.stack([ps[:, g].sum(1) * (FS / STEP) for g in FFT_GROUPS], axis=1)
    spectrum = 10 * np.log10(np.maximum(power, 1e-24))

    features = np.concatenate((hist, spectrum, stats), axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite dense features")
    return features


def _compact_spectrum(db):

    db = np.asarray(db, np.float64)
    if db.ndim != 2 or db.shape[1] != 128 or not np.isfinite(db).all():
        raise ValueError("Expected finite [frames,128] powers in dB")
    grouped = db.reshape(-1, 16, 8)
    peak = grouped.max(2)
    band_db = peak + 10 * np.log10(np.sum(10 ** ((grouped - peak[:, :, None]) / 10), 2))
    maximum = band_db.max(1, keepdims=True)
    powers = 10 ** ((band_db - maximum) / 10)
    total = powers.sum(1, keepdims=True)
    fractions = powers / total
    total_db = maximum[:, 0] + 10 * np.log10(total[:, 0])

    p128 = 10 ** ((db - db.max(1, keepdims=True)) / 10)
    p128 /= p128.sum(1, keepdims=True)
    centroid = np.sum(p128 * BAND_CENTRES_NYQUIST, 1)
    entropy = -np.sum(fractions * np.log(np.maximum(fractions, 1e-300)), 1) / np.log(16)
    flatness = np.exp(np.mean((band_db - maximum) * np.log(10) / 10, 1) - np.log(total[:, 0] / 16))
    return np.column_stack((band_db, fractions, total_db, centroid, entropy, flatness))


def absolute_features(distribution10hz):

    x = np.asarray(distribution10hz)
    if x.ndim != 2 or x.shape[1] != 200 or not np.isfinite(x).all():
        raise ValueError("Expected a finite array with 200 statistics per frame")
    if np.any(x[:, :64] < -1e-7) or np.any(x[:, 64:192] < -240.001):
        raise ValueError("Invalid histogram/power source values")
    n = len(x) // 10
    slow = np.empty((n, ABSOLUTE_DIM), np.float32)
    scale = np.array([0.001] * 5 + [1.0] * 3)
    for start in range(0, len(x), 40960):
        stop = min(start + 40960, len(x))
        z = np.asarray(x[start:stop], np.float64)
        hist = np.maximum(z[:, :64], 0).reshape(-1, 8, 8).sum(2)
        stats = np.arcsinh(z[:, 192:] / scale)

        count = len(z) // 10
        if count:
            z = z[: count * 10]
            grouped = z[:, 64:192].reshape(count, 10, 128)
            peak = grouped.max(1)
            db = peak + 10 * np.log10(np.mean(10 ** ((grouped - peak[:, None, :]) / 10), 1))
            hist_slow = hist[: count * 10].reshape(count, 10, 8).mean(1)
            stats_slow = stats[: count * 10].reshape(count, 10, 8).mean(1)
            slow[start // 10 : start // 10 + count] = np.column_stack(
                (_compact_spectrum(db), np.sqrt(hist_slow), stats_slow)
            )
    if not np.isfinite(slow).all():
        raise FloatingPointError("Nonfinite compact AE features")
    return slow
