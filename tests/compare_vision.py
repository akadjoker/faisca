#!/usr/bin/env python3
"""Layer by layer comparison of the faisca vision encoder against llama.cpp's mtmd.

    python3 tests/compare_vision.py <ours_trace.bin> <ref_trace.bin> [--rel 1e-5] [--abs 1e-4]
    python3 tests/compare_vision.py --embeddings <ours.bin> <ref.bin> --n-embd 2560

The traces come from `vision_dump --trace` and `llama_vision_dump --trace` (same
format as the text model dumps). A faisca tensor with more columns than the
reference (the FFN padded to the Q8_0 block) is compared on the reference's
columns. --embeddings compares the raw embedding files of the two tools.
"""

import argparse
import sys

import numpy as np

from compare_layers import read_dump


def error_of(mine: np.ndarray, theirs: np.ndarray) -> tuple[float, float]:
    a = mine.reshape(-1).astype(np.float64)
    b = theirs.reshape(-1).astype(np.float64)
    scale = float(np.abs(b).max()) if b.size else 0.0
    max_abs = float(np.abs(a - b).max()) if b.size else 0.0
    norm = float(np.linalg.norm(b))
    rel = float(np.linalg.norm(a - b) / norm) if norm > 0 else float(np.linalg.norm(a - b))
    return (max_abs / scale if scale > 0 else max_abs), rel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ours")
    parser.add_argument("reference")
    parser.add_argument("--rel", type=float, default=1e-5)
    parser.add_argument("--abs", type=float, default=1e-4)
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--n-embd", type=int, default=2560)
    args = parser.parse_args()

    if args.embeddings:
        a = np.fromfile(args.ours, dtype="<f4").reshape(-1, args.n_embd)
        b = np.fromfile(args.reference, dtype="<f4").reshape(-1, args.n_embd)
        max_abs, rel = error_of(a, b)
        rows = np.linalg.norm(a.astype(np.float64) - b, axis=1) / np.linalg.norm(b.astype(np.float64), axis=1)
        bad = rel > args.rel or max_abs > args.abs
        print(f"embeddings {a.shape}: max abs/scale {max_abs:.3e}  rel L2 {rel:.3e}  worst token {int(rows.argmax())} "
              f"{rows.max():.3e}{'  FAIL' if bad else ''}")
        return 1 if bad else 0

    ours = read_dump(args.ours)
    reference = read_dump(args.reference)
    failures = 0
    compared = 0
    worst = (0.0, "")
    for name, mine in ours.items():
        variants = [(name, reference[name])] if name in reference else []
        variants += [(key, value) for key, value in reference.items() if key.startswith(name + "#")]
        if not variants:
            print(f"  {name:20s} only in faisca")
            continue
        best = None
        for key, theirs in variants:
            candidate = mine
            if mine.shape != theirs.shape:
                if mine.ndim == 2 and theirs.reshape(-1).size % mine.shape[0] == 0:
                    theirs_2d = theirs.reshape(mine.shape[0], -1)
                    if theirs_2d.shape[1] <= mine.shape[1]:
                        candidate = mine[:, :theirs_2d.shape[1]]
                        theirs = theirs_2d
                if candidate.size != theirs.size:
                    continue
            max_abs, rel = error_of(candidate, theirs)
            if best is None or rel < best[2]:
                best = (key, max_abs, rel, theirs.shape)
        if best is None:
            print(f"  {name:20s} shape {mine.shape} vs {[v.shape for _, v in variants]}")
            failures += 1
            continue
        key, max_abs, rel, shape = best
        compared += 1
        bad = rel > args.rel or max_abs > args.abs
        failures += 1 if bad else 0
        if rel > worst[0]:
            worst = (rel, key)
        print(f"  {key:22s} shape {tuple(shape)!s:18s} max abs/scale {max_abs:10.3e}  rel L2 {rel:10.3e}{'  FAIL' if bad else ''}")
    for name in reference:
        if name not in ours and "#" not in name:
            print(f"  {name:20s} only in llama.cpp")
    status = "ok  " if failures == 0 else "FAIL"
    print(f"{status} {compared} tensors compared, worst rel L2 {worst[0]:.3e} ({worst[1]}), {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
