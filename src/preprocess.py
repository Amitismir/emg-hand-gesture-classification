from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Dict
import numpy as np
from scipy import signal
from typing import Dict, Iterable, List, Tuple


# ---------- Fs estimation (optional but useful) ----------

def estimate_fs_from_locs(locs: np.ndarray, seconds_per_move: float = 5.0) -> float:
    locs = np.asarray(locs).astype(float).squeeze()
    diffs = np.diff(locs)
    return float(np.mean(diffs) / seconds_per_move)


# ---------- Spectrum (for before/after plots) ----------

def spectrum_fft(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x).astype(float).squeeze()
    x = x - np.mean(x)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x)) / n
    return freqs, mag


# ---------- Filters ----------

def highpass_butter(x: np.ndarray, fs: float, fc: float = 1.0, order: int = 1) -> np.ndarray:
    """
    High-pass Butterworth filter (suggested: order=1, fc=1 Hz).
    """
    x = np.asarray(x).astype(float).squeeze()
    b, a = signal.butter(order, fc, btype="highpass", fs=fs)
    return signal.filtfilt(b, a, x)


def notch_filter(x: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
    """
    Notch at f0 Hz (Iran powerline = 50 Hz). Uses filtfilt (zero-phase).
    If f0 >= Nyquist, returns unchanged.
    """
    x = np.asarray(x).astype(float).squeeze()
    nyq = fs / 2.0
    if f0 >= nyq:
        return x.copy()

    b, a = signal.iirnotch(w0=f0, Q=q, fs=fs)
    return signal.filtfilt(b, a, x)


def apply_notches(x: np.ndarray, fs: float, freqs: Iterable[float] = (50.0,), q: float = 30.0) -> np.ndarray:
    y = np.asarray(x).astype(float).squeeze().copy()
    for f0 in freqs:
        y = notch_filter(y, fs=fs, f0=float(f0), q=q)
    return y


def preprocess_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Step 1 preprocessing:
      1) high-pass Butterworth (1 Hz, order 1)
      2) notch at 50 Hz
    """
    x_hp = highpass_butter(x, fs, fc=1.0, order=1)
    x_clean = apply_notches(x_hp, fs, freqs=(50.0,), q=30.0)
    return x_clean


# ---------- Save/load preprocessed cache (NPZ -> NPZ) ----------

def preprocess_npz_cache(npz_in: str | Path, npz_out: str | Path, fs: float) -> Path:
    """
    Loads an existing NPZ cache (like data/interim/emg_cache.npz),
    preprocesses every <chXX_bYY__data>, and saves a new NPZ.

    Keeps locs/labels unchanged.
    """
    npz_in = Path(npz_in)
    npz_out = Path(npz_out)
    if not npz_in.exists():
        raise FileNotFoundError(f"Input NPZ not found: {npz_in}")

    with np.load(npz_in, allow_pickle=False) as z:
        out: Dict[str, np.ndarray] = {k: z[k] for k in z.files}

    # Find all data keys and preprocess them
    data_keys = [k for k in out.keys() if k.startswith("ch") and k.endswith("__data")]
    for k in data_keys:
        out[k] = preprocess_signal(out[k], fs=fs).astype(float)

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_out, **out)
    return npz_out

#----------------------Split the signal + Normalization --------------




def split_blocks_80_20(all_blocks: List[int], seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Randomly split blocks into 80/20.
    With 10 blocks -> 8 train, 2 test.
    """
    rng = np.random.default_rng(seed)
    blocks = np.array(all_blocks, dtype=int)
    rng.shuffle(blocks)
    n_test = max(1, int(round(0.2 * len(blocks))))
    test_blocks = sorted(blocks[:n_test].tolist())
    train_blocks = sorted(blocks[n_test:].tolist())
    return train_blocks, test_blocks


def fit_channel_zscore_params_from_cache(
    cache: Dict[str, np.ndarray],
    train_blocks: Iterable[int],
    channels: Iterable[int],
) -> Dict[int, Tuple[float, float]]:
    """
    Fit mean/std per channel using ONLY training blocks (continuous signals).
    Returns dict: channel -> (mean, std)
    """
    params: Dict[int, Tuple[float, float]] = {}
    for ch in channels:
        xs = []
        for b in train_blocks:
            key = f"ch{ch:02d}_b{b:02d}__data"
            xs.append(np.asarray(cache[key], dtype=float).reshape(-1))
        x_all = np.concatenate(xs, axis=0)
        mu = float(np.mean(x_all))
        sd = float(np.std(x_all))
        if sd < 1e-8:
            sd = 1.0
        params[ch] = (mu, sd)
    return params


def apply_channel_zscore_to_cache(
    cache: Dict[str, np.ndarray],
    blocks: Iterable[int],
    channels: Iterable[int],
    params: Dict[int, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """
    Apply per-channel z-score normalization to specified blocks/channels.
    Returns a NEW cache dict with normalized __data arrays (labels/locs unchanged).
    """
    out = dict(cache)  # copy everything
    for ch in channels:
        mu, sd = params[ch]
        for b in blocks:
            key = f"ch{ch:02d}_b{b:02d}__data"
            x = np.asarray(out[key], dtype=float)
            out[key] = (x - mu) / sd
    return out

# --------------- Window Segementation ---------------------



def ms_to_samples(ms: float, fs: float) -> int:
    return int(round((ms / 1000.0) * fs))


def segment_bounds_from_locs(locs: np.ndarray, n: int) -> Tuple[List[int], List[int]]:
    """
    locs length=21 start markers. end is next marker; last end=n.
    Converts to 0-based indices.
    """
    locs = np.asarray(locs).astype(float).squeeze()
    starts = []
    for v in locs.tolist():
        i = int(round(v))
        i0 = i - 1 if i >= 1 else i
        starts.append(max(0, min(i0, n)))
    ends = starts[1:] + [n]
    return starts, ends


def sliding_windows_1d(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """
    Returns array of shape (num_windows, win)
    """
    x = np.asarray(x).astype(float).reshape(-1)
    n = len(x)
    if n < win:
        return np.empty((0, win), dtype=float)

    starts = np.arange(0, n - win + 1, step, dtype=int)
    return np.stack([x[s:s+win] for s in starts], axis=0)


def windows_from_one_record(
    x: np.ndarray,
    locs: np.ndarray,
    labels: List[str],
    fs: float,
    win_ms: float = 200.0,
    step_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment by locs/labels then window each segment.
    Returns:
      X_windows: (num_windows, win_samples)
      y: (num_windows,) labels (strings)
    """
    x = np.asarray(x).astype(float).reshape(-1)
    n = len(x)

    win = ms_to_samples(win_ms, fs)
    step = ms_to_samples(step_ms, fs)
    if step <= 0:
        step = 1

    starts, ends = segment_bounds_from_locs(locs, n)

    X_list = []
    y_list = []

    for lab, s, e in zip(labels, starts, ends):
        seg = x[s:e]
        W = sliding_windows_1d(seg, win=win, step=step)
        if W.shape[0] == 0:
            continue
        X_list.append(W)
        y_list.extend([lab] * W.shape[0])

    if not X_list:
        return np.empty((0, win), dtype=float), np.array([], dtype=str)

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list, dtype=str)
    return X, y


def build_window_dataset_from_cache(
    cache: Dict[str, np.ndarray],
    blocks: Iterable[int],
    channel: int,
    fs: float,
    win_ms: float = 200.0,
    step_ms: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build window dataset for ONE channel across selected blocks.
    Returns:
      X: (total_windows, win_samples)
      y: (total_windows,)
    """
    X_all = []
    y_all = []

    for b in blocks:
        key = f"ch{channel:02d}_b{b:02d}"
        x = cache[f"{key}__data"]
        locs = cache[f"{key}__locs"]
        labels = cache[f"{key}__labels"].tolist()

        Xb, yb = windows_from_one_record(x, locs, labels, fs, win_ms=win_ms, step_ms=step_ms)
        X_all.append(Xb)
        y_all.append(yb)

    X = np.concatenate(X_all, axis=0) if X_all else np.empty((0, ms_to_samples(win_ms, fs)))
    y = np.concatenate(y_all, axis=0) if y_all else np.array([], dtype=str)
    return X, y