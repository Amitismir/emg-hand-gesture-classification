from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data_loader import load_raw_mat, find_channels_data, build_records


def _pack_key(channel: int, block: int) -> str:
    return f"ch{channel:02d}_b{block:02d}"


def build_cache_from_mat(mat_path: str | Path) -> Dict[str, np.ndarray]:
    """
    Reads the .mat and returns a dict ready to be saved into an NPZ.
    Keys:
      - meta__shape
      - meta__labels_order (unique label order from first record)
      - per record:
          <key>__data
          <key>__locs
          <key>__labels  (stored as np.array of dtype '<U')
    """
    mat_path = Path(mat_path)
    raw = load_raw_mat(mat_path)
    channels_data, varname = find_channels_data(raw)
    records = build_records(channels_data)

    out: Dict[str, np.ndarray] = {}
    out["meta__shape"] = np.array(channels_data.shape, dtype=int)
    out["meta__main_var"] = np.array([varname])

    # Use first record as canonical label order (should be same everywhere)
    first_labels = records[0].labels if records and records[0].labels else []
    out["meta__labels_order"] = np.array(first_labels, dtype=str)

    for r in records:
        k = _pack_key(r.channel, r.block)
        out[f"{k}__data"] = np.asarray(r.data, dtype=float)
        out[f"{k}__locs"] = np.asarray(r.locs, dtype=float) if r.locs is not None else np.array([])
        out[f"{k}__labels"] = np.asarray(r.labels, dtype=str) if r.labels is not None else np.array([], dtype=str)

    return out


def save_npz_cache(cache: Dict[str, np.ndarray], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)
    return out_path


def load_npz_cache(npz_path: str | Path) -> Dict[str, np.ndarray]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ cache not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def list_records_in_cache(cache: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
    """
    Returns list of (channel, block) found in cache.
    """
    pairs = []
    for k in cache.keys():
        if k.endswith("__data") and k.startswith("ch"):
            prefix = k.split("__")[0]  # ch01_b01
            ch = int(prefix.split("_")[0].replace("ch", ""))
            b = int(prefix.split("_")[1].replace("b", ""))
            pairs.append((ch, b))
    return sorted(set(pairs))


def get_record_from_cache(cache: Dict[str, np.ndarray], channel: int, block: int):
    """
    Returns (data, locs, labels) from cache for the given channel/block.
    """
    key = _pack_key(channel, block)
    data = cache[f"{key}__data"]
    locs = cache.get(f"{key}__locs", np.array([]))
    labels = cache.get(f"{key}__labels", np.array([], dtype=str)).tolist()
    return data, locs, labels