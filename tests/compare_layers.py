#!/usr/bin/env python3
"""Layer by layer comparison of the faisca runner against llama.cpp.

    python3 tests/compare_layers.py <runner_dump> <llama_layer_dump> <model.gguf>
                                    --tokens 1 2 3 ... [--generate N] [--rel 1e-4] [--abs 1e-3]
                                    [--dequantize] [--ours-flags=--kv-f32] [--reference-flags=--kv-f16]

Both tools run the same prompt and write every intermediate tensor. For each
tensor this prints the largest absolute difference (relative to the largest
magnitude in the tensor) and the relative L2 error, then checks that the
greedily generated tokens are identical. llama.cpp gives some names to more
than one tensor (Qcur before and after RoPE, ffn_out before and after the
residual add); those variants are written as name#2, name#3 and the faisca
tensor is compared with the variant it matches best.
"""

import argparse
import struct
import subprocess
import sys
import tempfile

import numpy as np


def read_dump(path: str) -> dict[str, np.ndarray]:
    tensors = {}
    with open(path, "rb") as handle:
        data = handle.read()
    at = 0
    while at < len(data):
        (name_length,) = struct.unpack_from("<I", data, at)
        at += 4
        name = data[at:at + name_length].decode()
        at += name_length
        (dims,) = struct.unpack_from("<I", data, at)
        at += 4
        shape = struct.unpack_from("<" + "Q" * dims, data, at)
        at += 8 * dims
        count = int(np.prod(shape))
        values = np.frombuffer(data, dtype="<f4", count=count, offset=at).reshape(tuple(reversed(shape)))
        at += 4 * count
        tensors[name] = values
    return tensors


def run(tool: str, model: str, out: str, generate: int, tokens: list[int], flags: list[str]) -> list[int]:
    command = [tool] + flags + [model, out, str(generate)] + [str(t) for t in tokens]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"{tool}: {result.stderr.decode('utf-8', 'replace').strip()}")
    for line in result.stdout.decode().splitlines():
        if line.startswith("generated:"):
            return [int(t) for t in line.split()[1:]]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ours")
    parser.add_argument("reference")
    parser.add_argument("model")
    parser.add_argument("--tokens", nargs="+", type=int, required=True)
    parser.add_argument("--generate", type=int, default=16)
    parser.add_argument("--rel", type=float, default=1e-5, help="largest relative L2 error")
    parser.add_argument("--abs", type=float, default=1e-4, help="largest difference, relative to the tensor's largest magnitude")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dequantize", action="store_true", help="faisca converts every weight to float32 at load")
    parser.add_argument("--ours-flags", default="", help="more options for the faisca tool, written as --ours-flags=--kv-f32")
    parser.add_argument("--reference-flags", default="", help="options for the llama.cpp tool, written as --reference-flags=--kv-f16")
    parser.add_argument("--reference-model", default=None, help="a different model file for the reference side")
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".bin") as ours_file, tempfile.NamedTemporaryFile(suffix=".bin") as ref_file:
        ours_generated = run(args.ours, args.model, ours_file.name, args.generate, args.tokens,
                             (["--dequantize"] if args.dequantize else []) + args.ours_flags.split())
        ref_generated = run(args.reference, args.reference_model or args.model, ref_file.name, args.generate, args.tokens,
                            args.reference_flags.split())
        ours = read_dump(ours_file.name)
        reference = read_dump(ref_file.name)

    failures = 0
    compared = 0
    worst_rel = 0.0
    worst_name = ""
    def error_of(mine: np.ndarray, theirs: np.ndarray) -> tuple[float, float]:
        a = mine.reshape(-1).astype(np.float64)
        b = theirs.reshape(-1).astype(np.float64)
        scale = float(np.abs(b).max()) if b.size else 0.0
        max_abs = float(np.abs(a - b).max()) if b.size else 0.0
        norm = float(np.linalg.norm(b))
        rel = float(np.linalg.norm(a - b) / norm) if norm > 0 else float(np.linalg.norm(a - b))
        return (max_abs / scale if scale > 0 else max_abs), rel

    for name, mine in ours.items():
        variants = [(name, reference[name])] if name in reference else []
        variants += [(key, value) for key, value in reference.items() if key.startswith(name + "#")]
        variants = [(key, value) for key, value in variants if value.size == mine.size]
        if not variants:
            if name in reference:
                print(f"  {name:20s} shape {mine.shape} vs {reference[name].shape}")
                failures += 1
            else:
                print(f"  {name:20s} only in faisca")
            continue

        best = None
        for key, theirs in variants:
            max_abs, rel = error_of(mine, theirs)
            if best is None or rel < best[2]:
                best = (key, max_abs, rel, theirs.shape)
        key, max_abs, rel, shape = best
        compared += 1
        bad = rel > args.rel or max_abs > args.abs
        failures += 1 if bad else 0
        if rel > worst_rel:
            worst_rel = rel
            worst_name = key
        if bad or not args.quiet:
            print(f"  {key:22s} shape {tuple(shape)!s:16s} max abs/scale {max_abs:10.3e}  rel L2 {rel:10.3e}"
                  f"{'  FAIL' if bad else ''}")

    missing = [name for name in reference if name not in ours and "#" not in name]
    for name in missing:
        print(f"  {name:20s} only in llama.cpp")

    same_generation = ours_generated == ref_generated
    if not same_generation:
        failures += 1
        print(f"  generated tokens differ\n    llama.cpp: {ref_generated}\n    faisca:    {ours_generated}")

    status = "ok  " if failures == 0 else "FAIL"
    print(f"{status} {args.model}: {compared} tensors compared, worst rel L2 {worst_rel:.3e} ({worst_name}), "
          f"{len(ours_generated)} generated tokens {'identical' if same_generation else 'DIFFERENT'}, "
          f"{failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
