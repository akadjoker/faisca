#!/usr/bin/env python3
"""Writes small GGUF files with gguf-py, the reference implementation.

    pip install gguf numpy
    python3 tests/make_test_gguf.py <output directory>

The files cover every metadata type, every tensor type gguf-py knows, a custom
alignment, a file without tensors and a tiny file shaped like a Llama model.
Tensor contents are random: these files test the container, not a model.

Test only. The library itself does not use Python.
"""

import os
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES

RNG = np.random.default_rng(1234)

def finish(writer: GGUFWriter) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

def write_basic(path: str) -> None:
    writer = GGUFWriter(path, "testarch")

    writer.add_uint8("test.u8", 255)
    writer.add_int8("test.i8", -128)
    writer.add_uint16("test.u16", 65535)
    writer.add_int16("test.i16", -32768)
    writer.add_uint32("test.u32", 4294967295)
    writer.add_int32("test.i32", -2147483648)
    writer.add_uint64("test.u64", 18446744073709551615)
    writer.add_int64("test.i64", -9223372036854775808)
    writer.add_float32("test.f32", 0.1)
    writer.add_float32("test.f32_negative", -1234.5678)
    writer.add_float64("test.f64", 2.718281828459045)
    writer.add_bool("test.true", True)
    writer.add_bool("test.false", False)
    writer.add_string("test.string", "hello GGUF")
    writer.add_string("test.string_empty", "")
    writer.add_string("test.string_utf8", "açúcar, 日本語, emoji 🙂")
    writer.add_string("test.string_multiline", "line one\nline two\n\ttabbed")

    writer.add_array("test.array_i32", [1, -2, 3, -4, 5])
    writer.add_array("test.array_f32", [0.5, -1.25, 3.0])
    writer.add_array("test.array_bool", [True, False, True, True])
    writer.add_array("test.array_str", ["a", "", "bc", "çã", "a longer string with spaces"])
    writer.add_array("test.array_large", list(range(-5000, 5000)))
    writer.add_array("test.array_bytes", b"\x00\x01\x02\xfe\xff")

    writer.add_tensor("scalar_like.f32", RNG.standard_normal((1,)).astype(np.float32))
    writer.add_tensor("matrix.f32", RNG.standard_normal((7, 13)).astype(np.float32))
    writer.add_tensor("cube.f16", RNG.standard_normal((3, 5, 7)).astype(np.float16))
    writer.add_tensor("hyper.f32", RNG.standard_normal((2, 3, 4, 5)).astype(np.float32))
    writer.add_tensor("vector.f64", RNG.standard_normal((11,)).astype(np.float64))
    writer.add_tensor("vector.i8", RNG.integers(-128, 127, (33,), dtype=np.int8))
    writer.add_tensor("vector.i16", RNG.integers(-30000, 30000, (17,), dtype=np.int16))
    writer.add_tensor("vector.i32", RNG.integers(-1000000, 1000000, (9,), dtype=np.int32))
    writer.add_tensor("vector.i64", RNG.integers(-(1 << 40), 1 << 40, (5,), dtype=np.int64))

    finish(writer)

def write_all_tensor_types(path: str) -> None:
    writer = GGUFWriter(path, "testarch")

    for quant_type, (block_elements, block_bytes) in GGML_QUANT_SIZES.items():
        raw = RNG.integers(0, 256, (3, 2 * block_bytes), dtype=np.uint8)
        writer.add_tensor(f"raw.{quant_type.name.lower()}", raw, raw_dtype=quant_type)

    finish(writer)

def write_custom_alignment(path: str) -> None:
    writer = GGUFWriter(path, "testarch")
    writer.add_custom_alignment(64)
    writer.add_string("test.note", "tensor data aligned to 64 bytes")
    writer.add_tensor("a.f32", RNG.standard_normal((5, 3)).astype(np.float32))
    writer.add_tensor("b.i8", RNG.integers(-128, 127, (7,), dtype=np.int8))
    writer.add_tensor("c.f16", RNG.standard_normal((2, 9)).astype(np.float16))
    finish(writer)

def write_no_tensors(path: str) -> None:
    writer = GGUFWriter(path, "testarch")
    writer.add_string("general.name", "metadata only")
    writer.add_array("test.values", [1.0, 2.0, 3.0])
    finish(writer)

def write_llama_like(path: str) -> None:
    """Same keys and tensor names as a real Llama style model, tiny sizes."""
    n_vocab = 1000
    n_embd = 64
    n_ff = 128
    n_blocks = 2
    n_head = 4
    n_head_kv = 2
    head_dim = n_embd // n_head

    writer = GGUFWriter(path, "llama")
    writer.add_name("tiny llama-like test model")
    writer.add_context_length(256)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_blocks)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_file_type(1)

    tokens = [f"<tok{i}>".encode("utf-8") for i in range(n_vocab)]
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores([float(-i) for i in range(n_vocab)])
    writer.add_token_types([1] * n_vocab)
    writer.add_token_merges([f"a{i} b{i}" for i in range(500)])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_chat_template("{% for message in messages %}{{ message['content'] }}{% endfor %}")

    def f16(*shape):
        return RNG.standard_normal(shape).astype(np.float16)

    def f32(*shape):
        return RNG.standard_normal(shape).astype(np.float32)

    writer.add_tensor("token_embd.weight", f16(n_vocab, n_embd))
    for block in range(n_blocks):
        prefix = f"blk.{block}"
        writer.add_tensor(f"{prefix}.attn_norm.weight", f32(n_embd))
        writer.add_tensor(f"{prefix}.attn_q.weight", f16(n_embd, n_embd))
        writer.add_tensor(f"{prefix}.attn_k.weight", f16(n_head_kv * head_dim, n_embd))
        writer.add_tensor(f"{prefix}.attn_v.weight", f16(n_head_kv * head_dim, n_embd))
        writer.add_tensor(f"{prefix}.attn_output.weight", f16(n_embd, n_embd))
        writer.add_tensor(f"{prefix}.ffn_norm.weight", f32(n_embd))
        writer.add_tensor(f"{prefix}.ffn_gate.weight", f16(n_ff, n_embd))
        writer.add_tensor(f"{prefix}.ffn_up.weight", f16(n_ff, n_embd))
        writer.add_tensor(f"{prefix}.ffn_down.weight", f16(n_embd, n_ff))
    writer.add_tensor("output_norm.weight", f32(n_embd))

    finish(writer)

def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    files = {
        "basic.gguf": write_basic,
        "all_tensor_types.gguf": write_all_tensor_types,
        "alignment64.gguf": write_custom_alignment,
        "no_tensors.gguf": write_no_tensors,
        "llama_like.gguf": write_llama_like,
    }
    for name, build in files.items():
        path = os.path.join(out_dir, name)
        build(path)
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
