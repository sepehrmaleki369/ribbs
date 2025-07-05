#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import numpy as np
import torch

def compute_lut(file_list, sdf_min, sdf_max, n_bins, eps):
    """
    Compute LIF weights from a collection of SDFs.

    Args:
        file_list (List[str]): paths to .npy files containing SDF arrays
        sdf_min (float): lower clamp
        sdf_max (float): upper clamp
        n_bins (int): number of histogram bins
        eps (float): small constant inside log

    Returns:
        torch.Tensor: shape (n_bins,), dtype float32
    """
    counts = np.zeros(n_bins, dtype=np.int64)
    total = 0

    for pth in file_list:
        arr = np.load(pth)  # load SDF array
        # clamp into [sdf_min, sdf_max]
        arr = np.clip(arr, sdf_min, sdf_max)
        # normalize to [0,1]
        unit = (arr - sdf_min) / (sdf_max - sdf_min)
        # bin indices in [0, n_bins-1]
        idx = np.round(unit * (n_bins - 1)).astype(np.int64)
        flat = idx.ravel()
        # accumulate histogram
        bc = np.bincount(flat, minlength=n_bins)
        counts += bc
        total += flat.size

    if total == 0:
        raise RuntimeError("No SDF data found in the provided files.")

    # relative frequency per bin
    freq = counts.astype(np.float64) / total
    # LIF weight = 1 / log(1 + eps + freq)
    weights = 1.0 / np.log1p(eps + freq)

    return torch.from_numpy(weights.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(
        description="Compute Log-Inverse-Frequency LUT from SDF .npy files"
    )
    parser.add_argument(
        "json_path", type=Path,
        help="Path to JSON index file mapping keys to lists of .npy file paths"
    )
    parser.add_argument(
        "key", type=str,
        help="Key in the JSON whose list of .npy files to use (e.g. 'distance')"
    )
    parser.add_argument(
        "--sdf_min", type=float, default=-7.0,
        help="Minimum SDF value for clamping"
    )
    parser.add_argument(
        "--sdf_max", type=float, default=7.0,
        help="Maximum SDF value for clamping"
    )
    parser.add_argument(
        "--n_bins", type=int, default=256,
        help="Number of histogram bins"
    )
    parser.add_argument(
        "--eps", type=float, default=0.02,
        help="Small epsilon inside log"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("sdf_lut.pt"),
        help="Where to save the resulting .pt file"
    )
    args = parser.parse_args()

    # load JSON index
    with open(args.json_path, "r") as f:
        idx = json.load(f)

    if args.key not in idx:
        raise KeyError(f"Key '{args.key}' not found in {args.json_path}")

    file_list = idx[args.key]
    if not isinstance(file_list, list) or not file_list:
        raise ValueError(f"No files listed under key '{args.key}'")

    # compute LUT
    lut = compute_lut(
        file_list,
        sdf_min=args.sdf_min,
        sdf_max=args.sdf_max,
        n_bins=args.n_bins,
        eps=args.eps
    )
    print(lut)
    # save as .pt
    import os
    os.makedirs(args.output.parent, exist_ok=True)
    torch.save(lut, args.output)
    print(f"Saved LUT (shape={tuple(lut.shape)}) to {args.output}")


if __name__ == "__main__":
    main()



'''
python scripts/compute_lif_weights.py \
  ../Sinergia2/AL175-inds/train_index.json \
  distance \
  --sdf_min 0.0 \
  --sdf_max 15.0 \
  --n_bins 15 \
  --eps 0.02 \
  --output weights/al175_15_lif_weights.pt
'''
