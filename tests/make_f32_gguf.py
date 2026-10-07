#!/usr/bin/env python3
"""Writes a copy of a GGUF with every tensor dequantized to F32.

    python3 tests/make_f32_gguf.py <in.gguf> <out.gguf>

Both llama.cpp and faisca then compute from the same float32 weights, so the
layer by layer comparison only sees rounding differences.
"""

import sys

import numpy as np
import gguf
from gguf import quants


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    reader = gguf.GGUFReader(sys.argv[1])
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(sys.argv[2], arch)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.FILE_TYPE:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    arrays = []
    for tensor in reader.tensors:
        if tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                                  gguf.GGMLQuantizationType.BF16):
            data = tensor.data.astype(np.float32)
        else:
            data = quants.dequantize(tensor.data, tensor.tensor_type)
        shape = tuple(int(d) for d in reversed(tensor.shape.tolist()))
        data = np.ascontiguousarray(data.reshape(shape), dtype=np.float32)
        arrays.append(data)
        writer.add_tensor(tensor.name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{len(arrays)} tensors written to {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
