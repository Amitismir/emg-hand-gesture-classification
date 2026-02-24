from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def features_single_channel_core(x: np.ndarray) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N,L), got shape {x.shape}")

    abs_x = np.abs(x)
    mav = abs_x.mean(axis=1)
    iav = abs_x.sum(axis=1)
    rms = np.sqrt((x * x).mean(axis=1))
    var = x.var(axis=1)
    wl = np.sum(np.abs(np.diff(x, axis=1)), axis=1)

    return {"MAV": mav, "IAV": iav, "RMS": rms, "VAR": var, "WL": wl}


def corr_pairs(
    X: np.ndarray,
    pairs: Optional[List[Tuple[int, int]]] = None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,C,L), got shape {X.shape}")

    N, C, L = X.shape
    if pairs is None:
        pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]

    out: Dict[str, np.ndarray] = {}
    for a, b in pairs:
        xa = X[:, a, :]
        xb = X[:, b, :]

        xa0 = xa - xa.mean(axis=1, keepdims=True)
        xb0 = xb - xb.mean(axis=1, keepdims=True)

        sxa = np.std(xa0, axis=1)
        sxb = np.std(xb0, axis=1)
        sxa = np.where(sxa < eps, eps, sxa)
        sxb = np.where(sxb < eps, eps, sxb)

        cor = (xa0 * xb0).mean(axis=1) / (sxa * sxb)
        cor = np.clip(cor, -1.0, 1.0)

        out[f"Cor_{a+1}_{b+1}"] = cor

    return out


def extract_features_core_plus_cor(
    X: np.ndarray,
    channel_names: Optional[List[str]] = None,
    channels: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Core features (MAV, IAV, RMS, VAR, WL) per channel + correlation features.

    X: (N, C, L)

    channels (optional): select a subset of channel indices (0-based).
      Example: channels=[2,3] selects original ch3 & ch4 from a 4-channel tensor.

    channel_names:
      - if channels is provided and channel_names is None -> names become ch{idx+1}
      - if channels is None -> must match C, or auto default ch1..chC
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,C,L), got shape {X.shape}")

    N, C, L = X.shape

    # select subset if requested
    if channels is not None:
        X = X[:, channels, :]
        C = X.shape[1]
        if channel_names is None:
            channel_names = [f"ch{i+1}" for i in channels]  # original numbering
        elif len(channel_names) != C:
            raise ValueError(f"channel_names length {len(channel_names)} must equal selected C={C}")
    else:
        if channel_names is None:
            channel_names = [f"ch{i+1}" for i in range(C)]
        elif len(channel_names) != C:
            raise ValueError(f"channel_names length {len(channel_names)} must equal C={C}")

    cols: Dict[str, np.ndarray] = {}

    # per-channel core features
    for ch in range(C):
        feats = features_single_channel_core(X[:, ch, :])
        for name, arr in feats.items():
            cols[f"{name}_{channel_names[ch]}"] = arr

    # correlations (auto pairs)
    if C >= 2:
        cor = corr_pairs(X)
        # Rename Cor_1_2 -> Cor_ch3_ch4 style if channel_names reflect original
        renamed = {}
        for k, v in cor.items():
            # k like Cor_1_2 for selected tensor indexing
            i, j = k.split("_")[1:]
            i = int(i) - 1
            j = int(j) - 1
            renamed[f"Cor_{channel_names[i]}_{channel_names[j]}"] = v
        cols.update(renamed)

    return pd.DataFrame(cols)