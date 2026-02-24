from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat


@dataclass
class CellRecord:
    channel: int
    block: int
    data: np.ndarray               # 1D signal
    locs: Optional[np.ndarray]     # usually (21,)
    labels: Optional[List[str]]    # usually length 21


def load_raw_mat(mat_path: str | Path) -> Dict[str, Any]:
    """
    Load MAT file and return the raw dict from scipy.io.loadmat
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    return loadmat(mat_path, squeeze_me=True, struct_as_record=False)


def list_user_keys(raw: Dict[str, Any]) -> List[str]:
    """
    List non-metadata keys in .mat (ignoring __header__, __version__, __globals__)
    """
    return [k for k in raw.keys() if not k.startswith("__")]


def find_channels_data(raw: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """
    Find the 4x10 cell-like object array (channels_data).
    Returns (array, varname)
    """
    keys = list_user_keys(raw)

    #Finding channels_data
    if "channels_data" in keys:
        arr = raw["channels_data"]
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim == 2:
            return arr, "channels_data"

    # Otherwise: heuristic search for (4,10) object array
    for k in keys:
        v = raw[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.ndim == 2 and v.shape == (4, 10):
            return v, k

    raise ValueError(f"Could not find a 4x10 object array. Keys found: {keys}")


def _as_1d_float(x: Any) -> np.ndarray:
    a = np.asarray(x).squeeze()
    if a.ndim != 1:
        a = a.reshape(-1)
    return a.astype(float, copy=False)


def _normalize_labels(labels_any: Any) -> List[str]:
    # MATLAB cellstr often becomes ndarray of dtype object
    if labels_any is None:
        return []
    if isinstance(labels_any, (list, tuple)):
        return [str(z) for z in labels_any]
    arr = np.asarray(labels_any).squeeze()
    if arr.ndim == 0:
        return [str(arr.item())]
    if arr.dtype == object:
        return [str(z) for z in arr.tolist()]
    return [str(z) for z in arr.tolist()]


def parse_one_cell(cell: Any) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    cell is expected to be a struct with fields:
      - data
      - comments (struct) -> locs, labels
    """
    data = getattr(cell, "data", None)
    comments = getattr(cell, "comments", None)

    if data is None:
        raise ValueError("Cell missing 'data' field")

    locs = None
    labels = []

    if comments is not None:
        locs_raw = getattr(comments, "locs", None)
        labels_raw = getattr(comments, "lables", None)

        if locs_raw is not None:
            locs = _as_1d_float(locs_raw)

        if labels_raw is not None:
            labels = _normalize_labels(labels_raw)

    return _as_1d_float(data), locs, labels


def build_records(channels_data: np.ndarray) -> List[CellRecord]:
    """
    Convert the full 4x10 into a flat list of CellRecord.
    """
    records: List[CellRecord] = []
    n_ch, n_blk = channels_data.shape
    for ch in range(n_ch):
        for blk in range(n_blk):
            cell = channels_data[ch, blk]
            data, locs, labels = parse_one_cell(cell)
            records.append(
                CellRecord(
                    channel=ch + 1,
                    block=blk + 1,
                    data=data,
                    locs=locs,
                    labels=labels if labels else None,
                )
            )
    return records