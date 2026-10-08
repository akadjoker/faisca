#!/usr/bin/env python3
"""Compares two pockettts.cpp style dump directories (int32 ndims, int32 shape, float32 data)."""
import struct
import sys
from pathlib import Path

import numpy as np


def read(path):
    data = Path(path).read_bytes()
    nd = struct.unpack("<i", data[:4])[0]
    shape = struct.unpack("<%di" % nd, data[4:4 + 4 * nd])
    return np.frombuffer(data[4 + 4 * nd:], dtype="<f4").reshape(shape)


def main():
    if len(sys.argv) != 3:
        print("usage: compare_pocket.py <ours dir> <reference dir>")
        return 1
    ours, ref = Path(sys.argv[1]), Path(sys.argv[2])
    worst = 0.0
    for path in sorted(ref.glob("*.bin")):
        mine = ours / path.name
        if not mine.exists():
            print(f"{path.name:16s} missing on our side")
            continue
        a, b = read(mine), read(path)
        if a.shape != b.shape:
            print(f"{path.name:16s} shape {a.shape} vs {b.shape}")
            n = min(a.size, b.size)
            a, b = a.ravel()[:n], b.ravel()[:n]
        rel = float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)) / max(np.linalg.norm(b.astype(np.float64)), 1e-30))
        max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        worst = max(worst, rel)
        print(f"{path.name:16s} {str(b.shape):14s} rel L2 {rel:.3e}  max abs {max_abs:.3e}")
    print(f"worst rel L2 {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
