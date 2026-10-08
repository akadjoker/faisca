# faisca

LLM runner for small models, written from scratch in C++17. Library first, CLI on
top. No third party dependencies. Owner: Luís (akadjoker). "faisca" is a working name.

## Rules (set by the owner, follow them without being asked again)

- Opening brace always on its own line.
- Minimal or no comments. None at the top of files. A comment only when it
  explains something specific that the code does not show.
- No smart pointers. No std containers, std::string or exceptions in the library
  and CLI. Use `faisca::Array`, `faisca::StringMap`, `faisca::IntMap`,
  `faisca::Arena`, `faisca::Error`. Tests and benchmarks may use the standard library.
- Speed matters: write own data structures when they are faster, and prove it
  with a real benchmark on real data against the alternatives (see bench/).
- When investigating, never fill gaps with assumptions. Say "precisamos de mais
  dados" or "precisamos de mais investigação" instead.
- When there is a work plan, follow it. No shortcuts to finish a task sooner.
- Windows support is required (MinGW tested under Wine; MSVC not built yet).
- Talk to the owner in European Portuguese, without em dashes. Code, comments
  and README in English.
- Every piece is validated against a reference before moving on.

## Plan

1. GGUF loader: done (validated against gguf-py and ggml, 16x faster than ggml).
2. Tokenizer: done, see below.
3. CPU runner in float32: done, validated layer by layer against llama.cpp,
   see below.
4. CPU optimization: in progress. Done: Q8_0, Q4_0, Q5_0, Q4_K and Q6_K
   packed VNNI kernels (AVX-512), thread pool, batched prompt, tiled
   attention, KV cache in f16. Pending: AMX for the prompt, a faster Q4_0 and
   Q5_0 unpack (compute bound on one thread), AVX-VNNI (256-bit) kernels for
   Q5_0 and the K-quants (those types convert to float32 on such builds). Then speculative decoding with SmolLM2-135M as draft;
   then Vulkan; then the cache format.
5. Vulkan compute backend (the owner chose Vulkan over CUDA and OpenGL).
6. Own cache format with offset tables, KV cache kept between turns and saved to
   disk, longest common prefix reuse, snapshots.

Target model: SmolLM2-360M-Instruct GGUF, then 1B to 3B models.

## Tokenizer status

Byte level BPE (`tokenizer.ggml.model` = gpt2) with the `gpt-2`, `phi-2` and
`smollm` pre-tokenizers, matching llama.cpp token for token.

- Done: load from GGUF, special token partition, both splitters, BPE merges,
  decode with llama.cpp's clean_spaces passes, unit tests
  (`tests/tokenizer_unit.cpp`), ASan/UBSan clean, word cache, piece table for
  decode, linear scan merges for short words, benchmark against llama.cpp
  (`bench/bench_tokenizer.cpp`, results in bench/README.md: encode 16x to
  27x faster, decode 1.3x, load 5.8x).
- Validated against llama.cpp (commit b86d2f0), 0 mismatches, on the final
  code: the real SmolLM2-360M-Instruct vocabulary (59k cases, 4.2 M tokens,
  chat template tokens and words longer than 64 bytes included), GPT-2 in six
  variants (both pre-tokenizers, with and without extra special tokens and
  BOS/EOS flags, 295k cases, 26.1 M tokens), the unit test vocabulary (49k
  cases, 6.6 M tokens). Every optimization step was re-validated the same way.
- Not done: Windows build (MinGW not available in the last sessions), CLI
  `tokenize` command, a run with Clang.
- Known difference: code points above U+10FFFF from malformed UTF-8 make
  llama.cpp throw; faisca encodes them as U+FFFD.
- Reference setup: clone llama.cpp at b86d2f0, build `libllama` only
  (`-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF`), compile
  `tests/reference/llama_tokenizer_dump.cpp` against it. The GPT-2 files come
  from huggingface.co/openai-community/gpt2 (vocab.json, merges.txt), the model
  from huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF (q8_0).

## Runner status

`LlamaModel` + `Runner` (include/faisca/model.h, runner.h, ops.h, thread.h,
memory.h): the `llama` architecture. F32 in place, F16 converted to float32,
Q8_0, Q4_0, Q5_0, Q4_K and Q6_K packed at load into groups of 16 interleaved
rows (ops.h, "Packed Q8_0" and the K-quant layouts) in anonymous huge page
memory; `--f32` / `--dequantize` convert everything to float32 for the
reference comparison. Activations quantized like llama.cpp on x86: Q8_0
(scale amax/127 as f16, 127/amax, round to even) for the 32-block types, Q8_K
(blocks of 256, float32 scale max/-127 with max the signed extreme, ggml's
nearest_int, sums of 16) for Q4_K and Q6_K; `Weight::activation()` says
which. K-quant kernels: per super block the sub block scales are applied in
int32 (Q6_K subtracts 32 x sum16 first), then fmadd(sumi, d*dx) and, for
Q4_K, fnmadd(summ, dmin*dx), the structure of llama.cpp's AMX tile and VNNI
kernels, so matmul_check gives zero error; the scalar kernels use std::fma
and are bit identical to the AVX-512 ones (ops_unit checks it). Kernels:
AVX-512 VNNI, AVX-VNNI (Q8_0 and Q4_0 only), scalar reference; without the
kernel for a type the loader converts it to float32 (Q8_0 without VNNI stays
in the file layout with an AVX2 kernel). Prompt
in batches of 32 (Runner::BATCH), weights read once per batch. Attention over
keys stored in tiles of 16 positions (key_index), own vectorized exp (1 ulp of
expf). KV cache in f16 by default (`Runner::init(..., kv_f16)`, `--kv-f32` in
the CLI and runner_dump): halves converted with F16C on read, arithmetic in
float32, q and softmax weights never rounded; the kernels are templates
shared with the float32 cache, so attention_head_f16 is bit identical to
attention_head on the rounded values (ops_unit checks it). Thread pool over pthreads/Win32; every row group, head and token step
whole on one thread, so output is bit identical for 1, 2 or 4 threads (cmp of
runner_dump files is the test). -ffp-contract=off keeps Debug and Release
identical. No rope scaling, no biases, no MoE yet: rejected with an error.

- Validation: float32 path against llama.cpp with compare_layers.py: 386
  tensors, worst rel L2 4.5e-6 (16 tokens) and 3.9e-6 (1045 tokens), greedy
  tokens identical (both caches float32, `--ours-flags=--kv-f32`). f16 cache
  against the same float32 reference: 1.9e-3 and 2.49e-3, tokens identical;
  llama.cpp's own f16 cache (`--reference-flags=--kv-f16`) is at 2.2e-3 and
  2.50e-3, so ours is never further than theirs. On Q8_0 the 8th greedy token
  of the chat prompt is a near tie (gap 0.03 to 0.04 logits between 7042 and
  EOS) that flips with any change of rounding, on both runners. Quantized kernels with matmul_check against llama.cpp's
  own inputs: 193 matrices, zero error, Q8_0 and Q4_0. Full quantized pass:
  inherent noise, both runners 4.4e-2 from float32 logits, same argmax at
  every position, 16 greedy tokens identical. ASan, UBSan, TSan clean with 4
  threads.
- Speed, SmolLM2-360M, 2.1 GHz Sapphire Rapids VM, 1 / 4 threads. Generation:
  Q8_0 24.6 / 83 tok/s (llama.cpp 24.9 / 71), Q4_0 31 / 66 (llama.cpp 38.8 /
  70). At a context of 1045 positions, Q8_0: f16 cache 21.5 / 65 tok/s
  against 20.4 / 56 with the float32 cache, prompt 132 / 417 against 122 /
  393, cache 80 MiB instead of 160 for 2048 positions. Prompt of 633 tokens:
  Q8_0 142 / 443 tok/s, Q4_0 165 / 480 (llama.cpp pp512 255 / 143 with AMX;
  its pp128 with 4 threads was 893). Q4_K_M with a 512 token prompt then 32
  tokens (bench/README.md): generation 23.6 / 78 (llama.cpp tg32 19.2 / 54),
  prompt 164 / 492 (llama.cpp pp512 83 / 266); Q8_0 under the same harness
  24.6 / 76.5 and 166 / 494. The 633 token prompt text was not kept; the 512
  token file is the first 512 ids of /home/user/ref/long_tokens.txt. Q5_0
  experiment (scratch bench_q5): expanding Q5_0 to packed Q8_0 at load is 6%
  faster for one token on one thread (compute bound unpack, 7.0 vs 11.5 GB/s)
  but 1.55x the bytes and slower with 16 tokens; packed Q5_0 kept. float32
  path: about 5 tok/s on one thread.
- What explained the speed (the investigation is in the commit messages): the
  row by row kernel was compute bound at 6 GB/s of weights, ggml's own
  vec_dot too when called from our harness; llama.cpp reaches 10 GB/s because
  on this CPU it repacks Q8_0/Q4_0 into an AMX buffer and runs 512-bit VNNI
  over 16 interleaved rows. Attention was 12 GMAC per 633 token prompt
  (forgetting the 32 layers once cost an hour): tiles, broadcast FMAs and the
  vectorized exp fixed it. Huge pages: +25% on the row kernel, so the packed
  copy asks for them; load takes 0.5 to 1.2 s from synchronous compaction.
- Reference tools: `tests/reference/llama_layer_dump.cpp` and
  `tests/matmul_check.cpp` (see README). Q4_0 and Q4_K_M files of SmolLM2 were
  made with llama-quantize from the float32 copy; the Q4_K_M one is mostly
  Q5_0 with Q4_K/Q6_K only where n_in is a multiple of 256 (SmolLM2 has 960).
- Reference setup: `python3 tests/make_f32_gguf.py q8.gguf f32.gguf` (both
  sides read the same float32 weights), `llama_layer_dump` is built by the bench
  configuration (`-DFAISCA_BUILD_BENCH=ON -DFAISCA_LLAMA_DIR=...`). The tools
  take token ids; get them with `tokenizer_dump` on a cases file
  (`<u32 flags=3><u32 length><text>`).
- Pitfalls met: llama.cpp names both the projected and the rotated Q and K
  "Qcur"/"Kcur" and both the FFN down output and the residual sum "ffn_out";
  the dump tool suffixes repeats with #2, #3 and the script matches the best
  variant. llama.cpp defaults to an f16 KV cache and flash attention; the
  reference tool forces f32 and disables it. `llama_decode` takes
  `llama_batch`, `llama_process` takes `llama_batch_ext`.

## Commands

```
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
tests/run_all.sh                    # release + ASan/UBSan
python3 tests/compare_tokenizer.py build/tokenizer_dump <llama_tokenizer_dump> model.gguf --files text.txt
```

`tests/reference/llama_tokenizer_dump.cpp` links against a built llama.cpp.
