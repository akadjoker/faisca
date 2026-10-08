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
| Tokenizer (byte level BPE with the gpt-2, smollm and llama-bpe pre-tokenizers, validated against llama.cpp) | done |
| CPU runner, float32, validated layer by layer against llama.cpp | done |
| CPU optimization: threads, VNNI kernels for Q8_0, Q4_0, Q5_0, Q4_K and Q6_K, batched prompt, f16 KV cache | done, AMX pending |
| Vulkan compute backend | runs the whole model, validated layer by layer on llvmpipe; speed on a real GPU not measured yet |
| Own cache format and KV cache on disk | done: sessions with prefix reuse, snapshots and files with an offset table |
| Whisper (speech to text) | done: tiny, base and small validated layer by layer against whisper.cpp, `transcribe` command, no timestamps yet |
| Pocket TTS (text to speech) | done: validated against pockettts.cpp, `speak` command, 27 voices, no cloning from a recording yet |
| Qwen2.5, Qwen3, Gemma 3 | done: validated layer by layer against llama.cpp, SentencePiece tokenizer, sliding window attention; Gemma 3 4B and up need rope scaling |

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
faisca chat model.gguf -t 4 --session talk.session
faisca transcribe ggml-small.bin recording.wav --language pt -t 4
```

`chat` reads one user message per line and keeps the conversation in the KV
cache between turns, so a new turn only processes its own tokens;
`--session` loads the conversation from the file if it exists and saves it
after every reply, so a chat can be resumed days later without recomputing
anything.

`generate` wraps the prompt in the SmolLM2 chat template (`--raw` to skip
that), decodes greedily and prints the tokens as they come, then the timings.
`-t` sets the threads, `--f32` converts the weights to float32 at load (the
validation path), `--kv-f32` keeps the KV cache in float32 instead of f16.
`--draft <small.gguf>` turns on speculative decoding with a smaller model of
the same vocabulary (`--draft-n` tokens per round, default 3); the output is
the same, only the speed changes, see bench/README.md for when it helps.
SmolLM2-360M on a 2.1 GHz Xeon, 4 threads: 98 tokens per second from Q8_0,
about 150 from Q4_0, 118 from Q4_K_M, prompts at 550 to 600 tokens per
second; SmolLM2-1.7B Q4_K_M: 28 tokens per second, prompts at 125;
Llama 3.2 1B Q8_0: 31 to 37 tokens per second, prompts at 250
(bench/README.md has the tables against llama.cpp). The quantized weights
are packed into the kernels' layout by all threads at load. The chat template follows the
vocabulary: ChatML for SmolLM2, the header tokens for Llama 3.

`generate` and `chat` take `llama` files (SmolLM2, Llama 3.2), `qwen2`
(Qwen2.5), `qwen3` and `gemma3` ones; the chat template follows the
architecture (ChatML, Qwen2.5's default system prompt, Llama 3 headers,
Gemma's turns). On the same Xeon with 4 threads, 512 token prompt then 32
tokens: Qwen2.5-0.5B Q8_0 458 and 55 tokens per second, Qwen3-0.6B 279 and
38, Gemma 3 1B 192 and 24 (llama.cpp without its AMX path: 328 / 42, 245 /
32, 135 / 20; bench/README.md).

`transcribe` runs a Whisper model from a whisper.cpp `ggml-*.bin` file on a
16 kHz 16 bit WAV (`ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav` makes one) and
prints the text: greedy, without timestamps, one 30 s window at a time,
`--language` the two letter code (`pt` for Portuguese). The log mel
spectrogram is bit identical to whisper.cpp's; the encoder and decoder
matrices are quantized to Q8_0 at load and run on the VNNI kernels of the
LLM runner (`--f32` keeps the file's values, the validation path). On the
same Xeon with 4 threads, 11 s of audio: tiny 0.48 s, base 1.1 s, small
3.3 s, against 0.85 / 1.5 / 4.0 s for whisper-cli with its f16 files and
0.72 / 1.1 / 2.9 s with its Q8_0 files (bench/README.md).

`speak` turns text into a 24 kHz WAV with a Pocket TTS pack (Kyutai's 100M
model in pockettts.cpp's GGUF: `pocket-tts-<pack>-Q8_0.gguf` or `-F32`,
plus `pocket-tts-<pack>-voices.gguf` with 27 voices such as alba, jane,
george):

```
faisca speak pocket-tts-english_2026-09-Q8_0.gguf pocket-tts-english_2026-09-voices.gguf alba \
    -p "The quick brown fox jumps over the lazy dog." -o fox.wav -t 4
```

The text is cut into chunks of at most 50 tokens at sentence and clause
boundaries, each chunk runs the flow LM one step per 80 ms frame after the
voice prompt until it predicts the end, and the Mimi decoder turns the
latents into audio. `--seed` picks the noise (default 0, the same seed
gives the same audio as pocket-tts). With 4 threads on the same Xeon
13.9 s of audio takes 2.2 s (pocket-tts 6.9 s on its 2 threads).

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
models, including the Llama 3 rope frequency factors of `rope_freqs.weight`;
linear and YaRN rope scaling are still rejected). F32 tensors are used in place in the mapped
file, F16 tensors are converted to float32 at load. Q8_0, Q4_0, Q5_0, Q4_K
and Q6_K tensors (the types a Q4_K_M file holds) are repacked at load into
groups of 16 interleaved rows in anonymous memory (huge pages where the
system allows), the layout the VNNI kernels want: one `vpdpbusd` advances 16
output rows, and the weights are read once per batch of 32 tokens.
Activations are quantized the way llama.cpp does on x86: Q8_0 per block of 32
for the Q8_0, Q4_0 and Q5_0 kernels, Q8_K per block of 256 (float32 scale,
sums of 16) for the K-quants, whose kernels sum every super block exactly in
integers with the sub block scales before two fused multiply adds in float32.
Builds with AVX-VNNI but no AVX-512 run the same five types with 256-bit
kernels (each 64 byte chunk of a group as two halves of 8 rows, unpacked
once and reused for 4 inputs of the batch) and give bit identical output to
the AVX-512 build; builds without VNNI keep Q8_0 in place
with an AVX2 kernel and convert the rest to float32.

`Runner` holds the KV cache (`n_ctx` positions, keys and values in f16 unless
`init` is told otherwise, keys in tiles of 16 positions) and a thread pool.
The f16 halves are converted with F16C as they are read and the arithmetic
stays float32; q and the softmax weights are never rounded, unlike llama.cpp
which rounds both to f16 for its f16 cache. Every matrix row group, attention head and
token step is computed whole by one thread, so the output is bit identical
for any thread count. The prompt is processed in batches; generation reuses
the same kernels one token at a time.

`Session` (include/faisca/session.h) keeps the tokens a runner's cache holds
by position. `feed` keeps the longest prefix the new tokens share with them
(all but the last token at most) and forwards only the rest; `snapshot` and
`restore` serialize the tokens and the cache of the positions in use
(`Runner::export_state` and `import_state`); `save` and `load` write that
snapshot as a file: a header with the model name, the hyperparameters the
cache depends on and the cache type, then an offset table with one entry for
the tokens and one per layer for the keys and for the values, so a reader can
find any part without parsing the rest. A session of another model, another
cache type or another version is refused. `tests/session_check` verifies
that prefix reuse, a cut at a divergence, restore from memory and load from
a file in a fresh runner all continue a generation with logits bit identical
to an uninterrupted run.

`Speculative` (include/faisca/speculative.h) runs greedy speculative decoding
over two runners sharing one thread pool (`Runner::init` with a
`ThreadPool&`): the draft proposes `n_draft` tokens one at a time, the target
checks them in one batched pass (`Runner::forward` with all logits) and keeps
the longest agreeing prefix plus one token of its own, then both caches are
cut back with `Runner::truncate`. The tokens are exactly those of greedy
decoding with the target alone, which `tests/speculative_check` verifies.

`VulkanDevice` (include/faisca/vulkan.h) opens the system's Vulkan loader at
run time, so the library has no link time dependency and a machine without
Vulkan just gets an error from `init`; it exposes buffers, compute pipelines
and recorded dispatches with push constants. `VulkanKernels`
(vulkan_ops.h) holds the shaders, GLSL under src/vulkan/shaders compiled to
SPIR-V by `tools/embed_spirv.py` and committed as arrays in
src/vulkan/shaders_spv.cpp, so building needs neither glslc nor the shader
files: `matvec_q8_0` (Q8_0 rows split into packed f16 scales and int8
quants, float32 activations and arithmetic), `matvec_f32`, the embedding
gather, `rms_norm`, `rope`, the KV cache store, `attention`, `swiglu` and
`add`. `VulkanRunner` (vulkan_runner.h) runs the whole model with them,
one command buffer per batch of 32 tokens, weights on the device in their
quantized form (Q8_0, Q4_0, Q5_0, Q4_K and Q6_K each have a matvec shader
that dequantizes as it reads; F16 is converted to float32), activations and
KV cache in float32, and the same trace callback as `Runner`; `faisca generate --vulkan` and
`runner_dump --vulkan` use it. Validation on llvmpipe, Mesa's CPU device
(there is no GPU on the development machine, so no speed numbers yet):
`tests/vulkan_check` compares the matvec kernels over block 0, the output
matrix and (`--all-down`) every `ffn_down` with a double precision product:
1.2e-7 to 1.8e-7 for Q8_0, Q4_0, Q5_0, Q6_K and F32, 3e-7 to 5e-7 for Q4_K,
with 1 to 16 inputs; `compare_layers.py --ours-flags=--vulkan` puts the
float32 model at 1.8e-6 from llama.cpp and 3.4e-6 from the CPU runner, and
the Q8_0, Q4_0 and Q4_K_M models at 2.0e-6, 1.9e-6 and 2.3e-6 from llama.cpp
on their float32 copies (`--reference-model`), 16 greedy tokens identical in
every case. `cmake
-DFAISCA_VULKAN=OFF` leaves the backend out.

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
  | chat prompt, 1045 tokens | 386 | 5.3e-6 (`ffn_out-13`) | 16, identical |

  Those rows keep both caches in float32 (`--ours-flags=--kv-f32`). With the
  default f16 cache on the faisca side and the same float32 reference, the
  error is the rounding of the keys and values: 1.9e-3 (16 tokens) and
  2.59e-3 (1045 tokens), tokens still identical. llama.cpp with its own f16
  cache (`--reference-flags=--kv-f16`) sits at 2.2e-3 and 2.50e-3 from its
  float32 run.

  The Q8_0 and Q4_K_M models converted to float32 by faisca give bit
  identical tensors and tokens to the float32 copies made by gguf-py, so the
  dequantization of Q8_0, Q5_0, Q4_K and Q6_K matches it. The Q4_K_M model
  computed in float32 on both sides: 386 tensors, worst 3.3e-6, 16 tokens
  identical.
- `speculative_check`: generates N tokens greedily with the target alone and
  then with a draft, checks that the sequences are identical and prints the
  acceptance rate and both speeds. Identical in every run so far (SmolLM2-360M
  Q8_0 with SmolLM2-135M in Q8_0 and Q4_0, prompts of 16 and 512 tokens, 32
  to 128 generated, 1 and 4 threads, under ASan too).
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

- **Whisper against whisper.cpp.** `tests/make_f32_whisper.py` rewrites a
  `ggml-*.bin` with every tensor in float32; `whisper_layer_dump`
  (`-DFAISCA_WHISPER_DIR`, `-DFAISCA_WHISPER_BUILD`, against a copy of
  whisper.cpp 1.9.5 patched with an eval callback, named tensors and a
  `WHISPER_F32_REFERENCE` build that keeps the intermediates in float32)
  writes every tensor of a run, `whisper_dump` does the same for faisca and
  `tests/compare_whisper.py` compares them. tiny, base and small on
  jfk.wav and on Portuguese clips from FLEURS: worst rel L2 1.3e-5 / 1.6e-5
  / 1.6e-5 over 1545 / 2475 / 4893 tensors on the float32 path, tokens
  identical in every run, 1 and 4 threads bit identical, ASan and UBSan
  clean. With the Q8_0 matrices the noise reaches 1.6e-1 at the last
  encoder GELU and 8e-2 at the logits, small and base keep the same
  tokens, tiny flips one near tie ("you, ask"), the same flip whisper.cpp
  makes with its own Q8_0 file.

- **Qwen and Gemma against llama.cpp.** The same `compare_layers.py` on the
  float32 copies: Qwen2.5-0.5B-Instruct 266 tensors at worst 5.5e-6,
  Qwen3-0.6B 366 at 1.9e-6, Gemma 3 1B 367 at 4.1e-6 on a chat prompt and
  3.6e-6 on a 600 token prompt, where Gemma's local layers see only the
  last 512 positions; 16 greedy tokens identical, the Q8_0 files give the
  same tokens, 1 and 4 threads bit identical, ASan and UBSan clean. The
  reference copy has ggml's GELU f16 table turned off, so the comparison
  is against the exact GELU. The qwen2 pre-tokenizer and the SentencePiece
  tokenizer give 0 mismatches against llama.cpp over 40062 cases each
  (2.9 M and 3.3 M tokens).

- **Pocket TTS against pockettts.cpp.** `pocket_dump` writes what
  pockettts.cpp writes with `--dump` (the prompt ids, text embeddings, the
  hidden state and latent of step 0, every EOS logit and latent, the audio)
  and `tests/compare_pocket.py` compares the two directories. Against a
  float32 build of the reference (no GELU table, float32 im2col) on the
  English pack, voice alba, seed 1: ids and embeddings identical, step 0
  hidden 5.9e-7, EOS 2.9e-7, latents 5.1e-6, audio 4.0e-5 rel L2, the
  same number of frames, also on a three chunk text; 1 and 4 threads bit
  identical, ASan and UBSan clean. On the Q8_0 pack the two runners are
  1.2e-2 apart at step 0 while each is 1.3e-2 to 1.5e-2 from float32: one
  Q8_0 activation byte of the first layer rounds differently from a 1 ulp
  difference and the autoregressive loop keeps the runs apart; the
  reference's own stock and float32 builds are as far apart from each
  other on that pack.

## Layout

```
include/faisca/   public headers
src/              library
cli/              command line tool
tests/            tests and test tools
bench/            benchmarks against the alternatives
tools/            generator of the Unicode tables
```
