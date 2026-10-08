#!/usr/bin/env python3
"""Compare a faisca whisper_dump with a whisper_layer_dump of the reference.

    python3 tests/compare_whisper.py ours.bin reference.bin [--rel 1e-4] [--abs 1e-3]

Both files hold {name, rows, cols, float32 data} records; decoder tensors carry @step.
Prints the worst tensors, checks every tensor against the limits and that the token
lines (passed through --ours-tokens / --reference-tokens as text files) are identical.
"""

import argparse
import struct
import sys

import numpy as np


def read(path):
    out = {}
    order = []
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            n = struct.unpack("<I", head)[0]
            name = f.read(n).decode()
            f.read(4)
            ne = struct.unpack("<2Q", f.read(16))
            data = np.frombuffer(f.read(4 * ne[0] * ne[1]), dtype=np.float32).reshape(ne)
            out[name] = data
            order.append(name)
    return order, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ours")
    parser.add_argument("reference")
    parser.add_argument("--rel", type=float, default=1e-4)
    parser.add_argument("--abs", type=float, default=1e-3)
    parser.add_argument("--ours-tokens")
    parser.add_argument("--reference-tokens")
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    order, ours = read(args.ours)
    _, ref = read(args.reference)
    failures = 0
    missing = [name for name in order if name not in ref]
    results = []
    for name in order:
        if name not in ref:
            continue
        a = ours[name]
        b = ref[name]
        if a.shape != b.shape:
            if a.size == b.size:
                b = b.reshape(a.shape)
            else:
                print("%-24s shape %s vs %s" % (name, a.shape, b.shape))
                failures += 1
                continue
        scale = float(np.abs(b).max()) or 1.0
        max_abs = float(np.abs(a - b).max()) / scale
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0))
        bad = max_abs > args.abs or rel > args.rel
        failures += bad
        results.append((rel, max_abs, name, a.shape, bad))
    worst = sorted(results, key=lambda r: -r[0])[: args.show]
    for rel, max_abs, name, shape, bad in sorted(results, key=lambda r: order.index(r[2])):
        if bad or (rel, max_abs, name, shape, bad) in worst:
            print("  %-24s shape %-14s max abs/scale %.3e  rel L2 %.3e%s" % (name, shape, max_abs, rel, "  FAIL" if bad else ""))
    if missing:
        print("not in the reference: %s" % ", ".join(missing[:10]))
    extra = [name for name in ref if name not in ours]
    if extra:
        print("not in ours: %s" % ", ".join(extra[:10]))
    tokens_ok = True
    if args.ours_tokens and args.reference_tokens:
        a = open(args.ours_tokens).read().split("tokens:")[1].split("\n")[0].split()
        b = open(args.reference_tokens).read().split("tokens:")[1].split("\n")[0].split()
        tokens_ok = a == b
        print("tokens: %d ours, %d reference, %s" % (len(a), len(b), "identical" if tokens_ok else "DIFFERENT"))
    worst_rel = max(results, key=lambda r: r[0]) if results else (0, 0, "", (), False)
    print("%s %d tensors compared, worst rel L2 %.3e (%s), %d failures%s" % (
        "ok  " if failures == 0 and tokens_ok and not missing else "FAIL", len(results), worst_rel[0], worst_rel[2], failures,
        "" if tokens_ok else ", tokens differ"))
    sys.exit(0 if failures == 0 and tokens_ok and not missing else 1)


if __name__ == "__main__":
    main()
