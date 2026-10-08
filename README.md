# faisca

A small LLM runner written from scratch in C++17. A library first, with a
command line tool on top. No third party dependencies, plus the GPU driver once
the Vulkan backend lands.

The library does not use standard library containers, strings, smart pointers
or exceptions. It has its own containers (`faisca::Array`, `faisca::StringMap`,
`faisca::IntMap`, `faisca::Arena`), chosen by measurement: see
[bench/README.md](bench/README.md).

`faisca` is a working name.

## Status

| Step | State |
| --- | --- |
| GGUF loader (memory mapped, zero-copy) | done |
| Tokenizer (byte level BPE, validated against llama.cpp) | done |
| CPU runner, float32, validated layer by layer against llama.cpp | done |
| CPU optimization: threads, VNNI kernels for Q8_0, Q4_0, Q5_0, Q4_K and Q6_K, batched prompt, f16 KV cache | done, AMX pending |
| Vulkan compute backend | planned |
| Own cache format and KV cache on disk | planned |

## Build

```
cmake -S . -B build
cmake --build build -j
```

This produces the `faisca` static library and the `faisca` tool.

| Platform | State |
| --- | --- |
| Linux, GCC 13 and Clang 18 | built and tested, including ASan and UBSan |
| Windows, MinGW-w64 GCC 13 | cross-compiled and tested under Wine |
| Windows, MSVC | not built yet |

## Command line

```
faisca info model.gguf              # summary
faisca info model.gguf --meta       # every metadata key
faisca info model.gguf --tensors    # every tensor
faisca generate model.gguf -p "What is the capital of France?" -n 64 -t 4
```

`generate` wraps the prompt in the SmolLM2 chat template (`--raw` to skip
that), decodes greedily and prints the tokens as they come, then the timings.
`-t` sets the threads, `--f32` converts the weights to float32 at load (the
validation path), `--kv-f32` keeps the KV cache in float32 instead of f16.
SmolLM2-360M on a 2.1 GHz Xeon, 4 threads: 98 tokens per second from Q8_0,
about 150 from Q4_0, 118 from Q4_K_M, prompts at 550 to 600 tokens per
second (bench/README.md has the table against llama.cpp).

## Library

```cpp
#include "faisca/gguf.h"

#include <cstdio>

int main()
{
    faisca::Error error;
    faisca::GgufFile file;
    if (!file.open("model.gguf", error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    uint64_t n_embd = 0;
    file.get_uint("llama.embedding_length", n_embd);

    const faisca::GgufTensor* weights = file.find_tensor("token_embd.weight");
    // weights->data points straight into the mapped file.
    return 0;
}
```

```cpp
#include "faisca/tokenizer.h"

faisca::Tokenizer tokenizer;
if (!tokenizer.load(file, error))
{
    // unsupported tokenizer or damaged vocabulary
}

faisca::Array<int32_t> tokens;
tokenizer.encode("<|im_start|>user\nHello<|im_end|>\n", /*add_special*/ true, /*parse_special*/ true, tokens);

faisca::Array<char> text;
tokenizer.decode(tokens.data(), tokens.size(), /*remove_special*/ false, /*unparse_special*/ true, text);
```

The tokenizer is byte level BPE (`tokenizer.ggml.model` = `gpt2`) with the
`gpt-2`, `phi-2` and `smollm` pre-tokenizers, and produces the same tokens as
llama.cpp, special tokens and malformed UTF-8 included. Token texts are views
into the GGUF file, which must stay open. One `Tokenizer` per thread: it keeps
scratch buffers and a cache of the tokens of every word it has seen.

```cpp
#include "faisca/model.h"
#include "faisca/runner.h"

faisca::LlamaModel model;
faisca::Runner runner;
if (!model.load(file, error) || !runner.init(model, /*n_ctx*/ 2048, error))
{
    // unsupported architecture or tensor type, or out of memory
}

faisca::Array<float> logits;
runner.forward(tokens.data(), tokens.size(), logits, error); // logits of the last token
// pick the next token, then runner.forward(&next, 1, logits, error) and so on
```

`LlamaModel` reads the `llama` architecture (SmolLM2, Llama 2 and 3 style
models without rope scaling). F32 tensors are used in place in the mapped
file, F16 tensors are converted to float32 at load. Q8_0, Q4_0, Q5_0, Q4_K
and Q6_K tensors (the types a Q4_K_M file holds) are repacked at load into
groups of 16 interleaved rows in anonymous memory (huge pages where the
system allows), the layout the VNNI kernels want: one `vpdpbusd` advances 16
output rows, and the weights are read once per batch of 32 tokens.
Activations are quantized the way llama.cpp does on x86: Q8_0 per block of 32
for the Q8_0, Q4_0 and Q5_0 kernels, Q8_K per block of 256 (float32 scale,
sums of 16) for the K-quants, whose kernels sum every super block exactly in
integers with the sub block scales before two fused multiply adds in float32.
Builds with AVX-VNNI but no AVX-512 have kernels for Q8_0 and Q4_0 only and
convert the other types to float32; builds without VNNI keep Q8_0 in place
with an AVX2 kernel and convert the rest.

`Runner` holds the KV cache (`n_ctx` positions, keys and values in f16 unless
`init` is told otherwise, keys in tiles of 16 positions) and a thread pool.
The f16 halves are converted with F16C as they are read and the arithmetic
stays float32; q and the softmax weights are never rounded, unlike llama.cpp
which rounds both to f16 for its f16 cache. Every matrix row group, attention head and
token step is computed whole by one thread, so the output is bit identical
for any thread count. The prompt is processed in batches; generation reuses
the same kernels one token at a time.

`GgufFile` is a plain value object: no smart pointers, no heap allocation for
the object itself. The reader never copies or decodes anything up front.
Strings, arrays and tensors are views into the mapping and stay valid until the
object is closed or destroyed. Paths are UTF-8.

Errors come back as a message in a fixed buffer (`faisca::Error`), so reporting
one never allocates. The library does not throw, and running out of memory is
reported like any other error.

## Tests

```
pip install gguf numpy      # tests only, the library does not use Python
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

With a Visual Studio generator add `-C Release` to the `ctest` line. On Linux
`tests/run_all.sh` also runs everything under ASan and UBSan.

To check a real model against the reference reader:

```
python3 tests/compare_with_gguf_py.py build/gguf_dump model.gguf
```

- `containers_unit`: `Array`, `StringMap` and `Error`. The map is checked
  against `std::unordered_map` on random operations, including two different
  keys built to have the same 64-bit hash.
- `gguf_unit`: hand written images for edge cases and every rejection path.
- `ops_unit`: the float32 kernels (F16 conversion over every finite half,
  dot product, matrix times vector, RMS norm, RoPE, softmax, SwiGLU) against
  double precision references.
- `tokenizer_unit`: a hand written vocabulary with merges, special, byte,
  unused and empty tokens; load errors, byte fallback, special token
  partition, BOS/EOS, both pre-tokenizers, decode variants and the
  clean_spaces passes. `tokenizer_unit --write-gguf out.gguf` writes the same
  vocabulary for the differential test below.
- `compare_with_gguf_py.py`: reads the same file with faisca and with gguf-py,
  the reference implementation, and compares every key, value, tensor shape,
  offset and tensor byte checksum. Works on real model files too.
- `gguf_malformed`: truncated and randomly damaged files under
  AddressSanitizer and UndefinedBehaviorSanitizer. A damaged file must be
  rejected with an error or stay fully inside its bounds.
- `compare_tokenizer.py`: tokenizes the same texts with faisca and with
  llama.cpp (`tests/reference/llama_tokenizer_dump.cpp`, built against a
  llama.cpp checkout) and compares every token id and both detokenizations.
  Cases are whole files, paragraphs, lines and random strings mixing scripts,
  digits, every kind of whitespace, apostrophes, special tokens and malformed
  UTF-8. Results with llama.cpp at commit b86d2f0, 0 mismatches everywhere:

  | Vocabulary | Cases | Tokens |
  | --- | --- | --- |
  | SmolLM2-360M-Instruct (`smollm`, 17 special tokens) | 59,152 | 4.2 M |
  | GPT-2, six variants: `gpt-2` and `smollm` pre-tokenizers, with and without extra special tokens and BOS/EOS flags | 294,912 | 26.1 M |
  | the `tokenizer_unit` vocabulary | 49,152 | 6.6 M |

  The one known difference: a malformed 4 byte sequence that decodes above
  U+10FFFF makes llama.cpp throw, faisca encodes it as U+FFFD. The script
  counts those cases separately.
- `compare_layers.py`: runs the same prompt through `runner_dump` and through
  llama.cpp (`tests/reference/llama_layer_dump.cpp`, built with
  `-DFAISCA_BUILD_BENCH=ON -DFAISCA_LLAMA_DIR=...`), which captures every
  intermediate tensor of every block through `cb_eval`, with the KV cache in
  float32, flash attention off and one thread. Both read the same float32 GGUF
  made by `tests/make_f32_gguf.py`, so the only differences left are rounding
  and summation order. Then both generate greedily and the tokens must match.

  Results on SmolLM2-360M-Instruct, llama.cpp b86d2f0, float32 weights,
  faisca with `--dequantize`:

  | Prompt | Tensors compared | Worst relative L2 error | Generated tokens |
  | --- | --- | --- | --- |
  | chat prompt, 16 tokens | 386 | 4.5e-6 (`l_out-31`) | 16, identical |
  | chat prompt, 1045 tokens | 386 | 3.9e-6 (`ffn_out-13`) | 8, identical |

  Those rows keep both caches in float32 (`--ours-flags=--kv-f32`). With the
  default f16 cache on the faisca side and the same float32 reference, the
  error is the rounding of the keys and values: 1.9e-3 (16 tokens) and
  2.49e-3 (1045 tokens), tokens still identical. llama.cpp with its own f16
  cache (`--reference-flags=--kv-f16`) sits at 2.2e-3 and 2.50e-3 from its
  float32 run.

  The Q8_0 and Q4_K_M models converted to float32 by faisca give bit
  identical tensors and tokens to the float32 copies made by gguf-py, so the
  dequantization of Q8_0, Q5_0, Q4_K and Q6_K matches it. The Q4_K_M model
  computed in float32 on both sides: 386 tensors, worst 3.3e-6, 16 tokens
  identical.
- `matmul_check`: the quantized kernels on their own. It feeds every matrix of
  every block with the exact input llama.cpp computed (from a
  `llama_layer_dump` of the quantized model) and compares the products. On
  SmolLM2 Q8_0 and Q4_0, all 193 matrices match llama.cpp's outputs with zero
  error: the integer block sums are exact and the float accumulation has the
  same structure as llama.cpp's VNNI kernels. On the Q4_K_M file the 16 Q4_K
  and 16 Q6_K matrices match with zero error too (the sub block scales are
  applied in integers and the super block is accumulated with the same two
  fused multiply adds as llama.cpp's AMX and VNNI kernels); the 144 Q5_0
  matrices are within 1.8e-7, because llama.cpp runs Q5_0 through its generic
  AVX kernel, which accumulates eight float partial sums across blocks, while
  faisca sums every block exactly in integers. A full pass with quantized
  activations cannot be compared that tightly: differences of 1e-7 before the
  quantization round to different integers, and both runners sit about 4e-2
  from the float32 logits (llama.cpp 4.43e-2, faisca 4.42e-2) and 2e-2 from
  each other, with the same argmax at every position and the same 16 greedy
  tokens; on Q4_K_M the two runners are 6.7e-2 apart with the same 16 tokens.

## Layout

```
include/faisca/   public headers
src/              library
cli/              command line tool
tests/            tests and test tools
bench/            benchmarks against the alternatives
tools/            generator of the Unicode tables
```
