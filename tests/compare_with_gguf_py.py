#!/usr/bin/env python3
"""Compares faisca's GGUF reader with gguf-py, the reference implementation.

Both sides produce the same canonical dump (see tests/gguf_dump.cpp): every
metadata key with its type and value, every tensor with its type, shape,
offset, size and a checksum of its bytes. The dumps must be identical.

Works on any GGUF file, including real models:

    pip install gguf numpy
    python3 tests/compare_with_gguf_py.py build/gguf_dump model.gguf [more.gguf ...]

Test only. The library itself does not use Python.
"""

import struct
import subprocess
import sys

import numpy as np
from gguf import GGUFReader, GGUFValueType

MASK = (1 << 64) - 1
CHUNK_WORDS = 1 << 20

def checksum64(data) -> int:
    """Same function as faisca::checksum64."""
    view = np.frombuffer(data, dtype=np.uint8)
    size = int(view.size)
    full = size // 8
    total = 0

    for start in range(0, full, CHUNK_WORDS):
        stop = min(start + CHUNK_WORDS, full)
        words = np.frombuffer(view[start * 8:stop * 8].tobytes(), dtype="<u8")
        index = np.arange(start, stop, dtype=np.uint64)
        weights = index * np.uint64(2) + np.uint64(1)
        total = (total + int(np.sum(words * weights, dtype=np.uint64))) & MASK

    tail = view[full * 8:].tobytes()
    if tail:
        word = int.from_bytes(tail + b"\0" * (8 - len(tail)), "little")
        total = (total + word * (2 * full + 1)) & MASK

    return (total + size) & MASK

def double_bits(value) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]

def format_value(field) -> str:
    kind = field.types[0]

    if kind == GGUFValueType.ARRAY:
        elem_type = int(field.parts[3][0])
        count = int(field.parts[4][0])
        raw = b"".join(part.tobytes() for part in field.parts[5:])
        return f"{elem_type}:{count}:{len(raw)}:{checksum64(raw)}"

    part = field.parts[field.data[0]]
    if kind == GGUFValueType.STRING:
        raw = part.tobytes()
        return f"{len(raw)}:{checksum64(raw)}"
    if kind in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        return str(double_bits(part[0]))
    return str(int(part[0]))

def reference_dump(path: str) -> list[str]:
    reader = GGUFReader(path)
    raw_file = np.memmap(path, dtype=np.uint8, mode="r")
    lines = []

    lines.append(f"version\t{int(reader.fields['GGUF.version'].parts[0][0])}")
    lines.append(f"alignment\t{int(reader.alignment)}")
    lines.append(f"data_offset\t{int(reader.data_offset)}")

    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        lines.append(f"kv\t{name}\t{int(field.types[0])}\t{format_value(field)}")

    for tensor in reader.tensors:
        dims = [int(d) for d in tensor.field.parts[3]]
        offset = int(tensor.data_offset) - int(reader.data_offset)
        n_bytes = int(tensor.n_bytes)
        begin = int(tensor.data_offset)
        checksum = checksum64(raw_file[begin:begin + n_bytes])
        lines.append(
            f"tensor\t{tensor.name}\t{int(tensor.tensor_type)}\t{len(dims)}\t"
            f"{','.join(str(d) for d in dims)}\t{offset}\t{int(tensor.n_elements)}\t{n_bytes}\t{checksum}"
        )

    return lines

def faisca_dump(dump_binary: str, path: str) -> list[str]:
    result = subprocess.run([dump_binary, path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout.decode("utf-8").splitlines()

def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    dump_binary = sys.argv[1]
    failures = 0

    for path in sys.argv[2:]:
        expected = reference_dump(path)
        try:
            actual = faisca_dump(dump_binary, path)
        except RuntimeError as error:
            print(f"FAIL {path}: faisca could not read the file: {error}")
            failures += 1
            continue

        if expected == actual:
            kvs = sum(1 for line in expected if line.startswith("kv\t"))
            tensors = sum(1 for line in expected if line.startswith("tensor\t"))
            print(f"ok   {path}: {kvs} keys, {tensors} tensors")
            continue

        failures += 1
        print(f"FAIL {path}: {len(expected)} reference lines, {len(actual)} faisca lines")
        shown = 0
        for index in range(max(len(expected), len(actual))):
            left = expected[index] if index < len(expected) else "<missing>"
            right = actual[index] if index < len(actual) else "<missing>"
            if left != right:
                print(f"  line {index + 1}")
                print(f"    gguf-py: {left}")
                print(f"    faisca:  {right}")
                shown += 1
                if shown == 10:
                    break

    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
