#!/usr/bin/env python3
"""Compiles every GLSL compute shader in src/vulkan/shaders with glslc and writes
the SPIR-V words as C arrays into src/vulkan/shaders_spv.h and .cpp, so the
library needs neither glslc nor shader files at run time.

    python3 tools/embed_spirv.py [--glslc path]
"""

import argparse
import pathlib
import struct
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glslc", default="glslc")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    shaders = sorted((root / "src" / "vulkan" / "shaders").glob("*.comp"))
    header = ["#pragma once", "", "#include <cstddef>", "#include <cstdint>", "", "namespace faisca", "{", ""]
    source = ['#include "shaders_spv.h"', "", "namespace faisca", "{", ""]
    with tempfile.TemporaryDirectory() as temp:
        for shader in shaders:
            out = pathlib.Path(temp) / (shader.stem + ".spv")
            subprocess.run([args.glslc, "-O", "--target-env=vulkan1.1", "-fshader-stage=compute", str(shader),
                            "-o", str(out)], check=True)
            data = out.read_bytes()
            words = struct.unpack("<" + "I" * (len(data) // 4), data)
            name = "spv_" + shader.stem
            header.append(f"extern const uint32_t {name}[];")
            header.append(f"constexpr size_t {name}_words = {len(words)};")
            source.append(f"const uint32_t {name}[] = {{")
            for i in range(0, len(words), 8):
                source.append("    " + ", ".join(f"0x{w:08x}" for w in words[i:i + 8]) + ",")
            source.append("};")
            source.append("")
            print(f"{shader.name}: {len(words)} words")
    header += ["", "}", ""]
    source += ["}", ""]
    (root / "src" / "vulkan" / "shaders_spv.h").write_text("\n".join(header))
    (root / "src" / "vulkan" / "shaders_spv.cpp").write_text("\n".join(source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
