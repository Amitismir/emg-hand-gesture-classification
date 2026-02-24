from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def _safe_std(x: np.ndarray, axis=None, eps: float = 1e-12) -> np.ndarray:
    s = np.std(x, axis=axis)
    return np.where(s < eps, eps, s)


def features_single_channel(x: np.ndarray, zc_thresh: float = 0.0, ssc_thresh: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Compute single-channel EMG features for a batch of windows.

    x: shape (N, L) windows for one channel
    returns dict feature_name -> array shape (N,)
    """
    x = np.asarray(x, dtype=float)
    N, L = x.shape

    abs_x = np.abs(x)

    mav = abs_x.mean(axis=1)
    iav = abs_x.sum(axis=1)
    rms = np.sqrt((x * x).mean(axis=1))
    var = x.var(axis=1)
    std = np.sqrt(var)

    # Waveform Length (WL): sum of absolute differences
    wl = np.sum(np.abs(np.diff(x, axis=1)), axis=1)

    # Zero Crossing (ZC): sign changes with optional threshold
    x1 = x[:, :-1]
    x2 = x[:, 1:]
    zc = ((x1 * x2) < 0) & (np.abs(x1 - x2) >= zc_thresh)
    zc = zc.sum(axis=1)

    # Slope Sign Changes (SSC): sign change in first derivative with optional threshold
    d1 = x[:, 1:-1] - x[:, :-2]
    d2 = x[:, 2:] - x[:, 1:-1]
    ssc = ((d1 * d2) < 0) & ((np.abs(d1) >= ssc_thresh) | (np.abs(d2) >= ssc_thresh))
    ssc = ssc.sum(axis=1)

    # NP: Number of Peaks (simple definition)
    # peak if x[i] > x[i-1] and x[i] > x[i+1] and above a threshold
    # threshold based on per-window std to avoid counting noise peaks
    thr = 0.5 * _safe_std(x, axis=1)  # shape (N,)
    mid = x[:, 1:-1]
    left = x[:, :-2]
    right = x[:, 2:]
    peaks = (mid > left) & (mid > right) & (np.abs(mid) >= thr[:, None])
    npk = peaks.sum(axis=1)

    return {
        "MAV": mav,
        "IAV": iav,
        "RMS": rms,
        "VAR": var,
        "STD": std,
        "WL": wl,
        "ZC": zc,
        "SSC": ssc,
        "NP": npk,
    }


def corr_pairs(X: np.ndarray, pairs: List[Tuple[int, int]] | None = None) -> Dict[str, np.ndarray]:
    """
    Correlation features between channels inside each window.

    X: shape (N, C, L)
    returns dict like {"Cor_1_2": (N,), ...}
    """
    X = np.asarray(X, dtype=float)
    N, C, L = X.shape

    if pairs is None:
        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]  # 4 channels default

    out = {}
    for a, b in pairs:
        xa = X[:, a, :]
        xb = X[:, b, :]
        xa0 = xa - xa.mean(axis=1, keepdims=True)
        xb0 = xb - xb.mean(axis=1, keepdims=True)
        denom = _safe_std(xa0, axis=1) * _safe_std(xb0, axis=1)
        cor = (xa0 * xb0).mean(axis=1) / denom
        out[f"Cor_{a+1}_{b+1}"] = cor
    return out


def extract_features_all(
    X: np.ndarray,
    channel_names: List[str] | None = None,
    zc_thresh: float = 0.0,
    ssc_thresh: float = 0.0,
    include_cor: bool = True,
) -> pd.DataFrame:
    """
    Extract ALL features listed:
      MAV, STD, VAR, WL, ZC, RMS, NP, SSC, IAV per channel
      plus Cor between channel pairs

    X: shape (N, C, L)
    returns dataframe with one row per window, feature columns only (no label)
    """
    X = np.asarray(X, dtype=float)
    N, C, L = X.shape

    if channel_names is None:
        channel_names = [f"ch{i+1}" for i in range(C)]

    cols = {}

    # single-channel features
    for ch in range(C):
        feats = features_single_channel(X[:, ch, :], zc_thresh=zc_thresh, ssc_thresh=ssc_thresh)
        for name, arr in feats.items():
            cols[f"{name}_{channel_names[ch]}"] = arr

    # correlation features
    if include_cor and C >= 2:
        cor = corr_pairs(X)
        cols.update(cor)

    return pd.DataFrame(cols)