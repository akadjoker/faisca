#!/usr/bin/env python3
"""Compile CUDA kernels to embedded PTX; normal faisca builds need no toolkit."""

import argparse
import pathlib
import subprocess
import tempfile


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="compute_89")
    parser.add_argument("--source-dir", type=pathlib.Path, default=root / "src" / "cuda" / "kernels")
    parser.add_argument("--output-dir", type=pathlib.Path, default=root / "src" / "cuda")
    args = parser.parse_args()
    kernels = sorted(args.source_dir.glob("*.cu"))
    if not kernels:
        parser.error(f"no CUDA kernels in {args.source_dir}")

    header = ["#pragma once", "", "namespace faisca", "{", ""]
    source = ['#include "kernels_ptx.h"', "", "namespace faisca", "{", ""]
    with tempfile.TemporaryDirectory() as temp:
        for kernel in kernels:
            if not kernel.stem.isascii() or not kernel.stem.isidentifier():
                parser.error(f"kernel filename is not an ASCII identifier: {kernel.name}")
            output = pathlib.Path(temp) / (kernel.stem + ".ptx")
            subprocess.run([args.nvcc, "--ptx", "--std=c++17", "--gpu-architecture=" + args.arch,
                            "--fmad=false", str(kernel), "-o", str(output)], check=True)
            ptx = output.read_text(encoding="ascii")
            if ')faisca_ptx"' in ptx:
                parser.error(f"PTX contains the embedding delimiter: {kernel.name}")
            name = "ptx_" + kernel.stem
            header.append(f"extern const char {name}[];")
            source.append(f'const char {name}[] = R"faisca_ptx({ptx})faisca_ptx";')
            source.append("")
            print(f"{kernel.name}: {len(ptx)} PTX characters, {args.arch}")

    header += ["", "}", ""]
    source += ["}", ""]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "kernels_ptx.h").write_text("\n".join(header), encoding="ascii")
    (args.output_dir / "kernels_ptx.cpp").write_text("\n".join(source), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())