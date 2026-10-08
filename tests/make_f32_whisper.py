#!/usr/bin/env python3
"""Rewrite a whisper.cpp ggml model with every tensor in float32.

    python3 tests/make_f32_whisper.py ggml-tiny.bin ggml-tiny-f32.bin

The layout is the one written by whisper.cpp's convert-pt-to-ggml.py: magic,
11 int32 hparams (the last one the ftype), the mel filters, the vocabulary,
then tensors as {n_dims, name_length, ftype, dims[n_dims], name, data}.
"""

import struct
import sys

import numpy as np


def main():
    src = open(sys.argv[1], "rb").read()
    out = bytearray()
    pos = 0

    def read(fmt):
        nonlocal pos
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, src, pos)
        pos += size
        return values

    magic, = read("<i")
    if magic != 0x67676D6C:
        raise SystemExit("not a whisper ggml file")
    hparams = list(read("<11i"))
    hparams[10] = 0
    out += struct.pack("<i", magic) + struct.pack("<11i", *hparams)

    n_mel, n_fft = read("<2i")
    out += struct.pack("<2i", n_mel, n_fft)
    out += src[pos:pos + 4 * n_mel * n_fft]
    pos += 4 * n_mel * n_fft

    n_vocab, = read("<i")
    out += struct.pack("<i", n_vocab)
    for _ in range(n_vocab):
        length, = read("<i")
        out += struct.pack("<i", length) + src[pos:pos + length]
        pos += length

    count = 0
    while pos < len(src):
        n_dims, name_length, ftype = read("<3i")
        dims = list(read("<%di" % n_dims))
        name = src[pos:pos + name_length]
        pos += name_length
        n = int(np.prod(dims))
        if ftype == 1:
            data = np.frombuffer(src[pos:pos + 2 * n], dtype=np.float16).astype(np.float32)
            pos += 2 * n
        elif ftype == 0:
            data = np.frombuffer(src[pos:pos + 4 * n], dtype=np.float32)
            pos += 4 * n
        else:
            raise SystemExit("tensor %s has ftype %d" % (name, ftype))
        out += struct.pack("<3i", n_dims, name_length, 0) + struct.pack("<%di" % n_dims, *dims) + name
        out += data.tobytes()
        count += 1
    open(sys.argv[2], "wb").write(out)
    print("%s: %d tensors, hparams %s" % (sys.argv[2], count, hparams))


if __name__ == "__main__":
    main()
