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
   attention, KV cache in f16, software prefetch in the generation kernels.
   AMX for the prompt was measured (scratch bench_amx) and set aside: the
   per block float epilogue bounds it at 1.25x over VNNI on the GEMM, about
   10% of the prompt (bench/README.md). AVX-VNNI (256-bit) kernels for
   Q5_0, Q4_K and Q6_K: done, the AVX-VNNI build is bit identical to the
   AVX-512 one on every model (the softmax denominator of the attention is
   16 partial sums reduced by one fixed tree in both; before, the two
   builds differed by ulps there, invisible under f16 scales and visible
   under the float32 scale of Q8_K).
   Speculative decoding with SmolLM2-135M as draft: done (`Speculative`,
   `--draft`), output identical to plain greedy, gain 1.3 to 1.4x on one
   thread with a Q4_0 draft and about 1.1x with four, 0.9x at a 512 token
   context; the draft costs 0.3 to 0.4 of the target per token, which caps
   it (bench/README.md). Then Vulkan; then the cache format.
5. Vulkan compute backend (the owner chose Vulkan over CUDA and OpenGL): in
   progress. Done: VulkanDevice (loader opened at run time with dlopen /
   LoadLibrary, instance, compute queue, buffers with staging, pipelines,
   recorded dispatches with a memory barrier after each), VulkanKernels
   with matvec_q8_0 and matvec_f32 (GLSL in src/vulkan/shaders, SPIR-V
   embedded by tools/embed_spirv.py, generated file committed),
   tests/vulkan_check (1.3e-7 to 1.8e-7 against double precision, 1 and 16
   inputs), the other ops (embed, rms_norm, rope, kv_store, attention,
   swiglu, add) and VulkanRunner (include/faisca/vulkan_runner.h: weights
   uploaded from the GgufFile: Q8_0, Q4_0 and Q5_0 split into packed f16
   scales and uints of quants, Q4_K as the file's 36 uint blocks, Q6_K as
   53 uint blocks (2 bytes of padding), each with its own matvec shader
   that dequantizes as it reads, F16 as float32; the embedding gather
   reads Q8_0 or float32 only, so other embedding types convert; float32
   activations and KV cache; one command buffer per batch, submitted and
   downloaded after every traced step when a trace is set; `--vulkan` in
   the CLI and runner_dump). Validated with compare_layers
   (`--ours-flags=--vulkan`, `--reference-model` for a different file on
   the reference side): float32 model 1.8e-6 from llama.cpp and 3.4e-6
   from the CPU runner, Q8_0 / Q4_0 / Q4_K_M models 2.0e-6 / 1.9e-6 /
   2.3e-6 from llama.cpp on their float32 copies (the GPU does not quantize
   activations, so it is closer to float32 than the CPU quantized paths;
   against llama.cpp Q8_0 it is 8e-2 and the near tie token flips), 16
   tokens identical; vulkan_check per matrix 1.2e-7 to 5e-7 for every
   type (`--all-down` covers the Q4_K layers). Only llvmpipe (Mesa CPU, Vulkan
   1.4, subgroup 8) is available here, so correctness only; speed needs the
   owner's GPU. Next: speed on a real GPU (the kernels are simple: one
   workgroup per row, scalar loads; expected first optimizations: several
   rows per workgroup, vectorized loads, f16 cache, fewer dispatches),
   Windows loading path untested.
6. Own cache format with offset tables, KV cache kept between turns and saved to
   disk, longest common prefix reuse, snapshots: done. `Runner::state_bytes /
   export_state / import_state` (positions 0..pos: per layer the key tiles
   covering them, then the value rows), `Session` (include/faisca/session.h:
   tokens by position, `feed` with longest common prefix reuse capped at
   count - 1 so the last token always runs, `push`, `truncate`, `snapshot /
   restore` in memory, `save / load`: "FSCA" header, version 1, n_layer,
   n_head_kv, head_dim, kv_f16, n_vocab, n_tokens, model name, offset table
   of {kind, layer, offset, size} with the tokens then keys and values per
   layer, sections 8 byte aligned; another model, cache type or version is
   refused), `faisca chat` with `--session`. tests/session_check: prefix
   reuse, cut at a divergence, restore and load in a fresh runner all give
   logits bit identical to an uninterrupted run (Q8_0 and Q4_K_M, ASan
   clean). A 19 token snapshot of SmolLM2-360M in f16 is 1 MiB (whole key
   tiles of 16 positions are stored). The Vulkan runner has no state
   export yet.

Target model: SmolLM2-360M-Instruct GGUF, then 1B to 3B models. SmolLM2-1.7B
Q4_K_M (/home/user/models/1.7b/, official file, its float32 copy and the
llama dump /home/user/ref/llama_17b_dump.bin) validated: matmul_check zero
error on 145 matrices, float32 path 1.8e-5 from llama.cpp (block 22 is
sensitive: the quantized runs of both runners are 0.9 to 1.0 from float32
there, tokens identical), 16 tokens identical; 9.0 / 28.2 tok/s generation
and 45 / 125 prompt with 1 / 4 threads (llama.cpp 5.4 / 16.5 and 55 /
169). Load takes 10 s (packing 1 GB into huge pages). Llama 3.x files
need rope scaling, which the loader still rejects.

## Tokenizer status

Byte level BPE (`tokenizer.ggml.model` = gpt2) with the `gpt-2`, `phi-2`,
`smollm` and `llama-bpe` pre-tokenizers, matching llama.cpp token for token.
`llama-bpe` (Llama 3): `split_llama3` mirrors llama.cpp's hand written
`unicode_regex_split_custom_llama3`; `ignore_merges` (a word whose byte level
form is a vocabulary token is that token, looked up as the mapped string,
not the raw bytes); `clean_spaces` on; `add_bos` on unless the file says
otherwise. Validated on Llama 3.2 1B: 0 mismatches over 39941 cases, 2.69 M
tokens, the text corpus in /home/user/ref/texts.

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
AVX-512 VNNI, AVX-VNNI (256-bit, all five types, runner output bit
identical to the AVX-512 build: ops_unit on build-avxvnni, runner_dump cmp
on Q8_0 / Q4_0 / Q4_K_M / 1.7B, matmul_check zero error on the K-quants),
scalar reference; without the kernel for a type the loader converts it to
float32 (Q8_0 without VNNI stays in the file layout with an AVX2 kernel).
Prompt
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
- Speed, SmolLM2-360M, 2.1 GHz Sapphire Rapids VM, 1 / 4 threads, 512 token
  prompt then 32 tokens (bench/README.md). Generation: Q8_0 33.8 / 98 tok/s
  (llama.cpp 24.9 / 71), Q4_0 45 / 144 to 183 (llama.cpp 38.8 / 70), Q4_K_M
  40.3 / 118 (llama.cpp 19.2 / 54). Prompt: Q8_0 212 / 603, Q4_0 207 / 570
  to 599, Q4_K_M 204 / 550 (llama.cpp pp512 255 / 143, 249 / 147, 83 /
  266). Four thread numbers move up to 25% between runs on this VM. Before
  the software prefetch generation was Q8_0 24.6 / 76.5, Q4_0 30.8 / 89.7,
  Q4_K_M 23.6 / 78; the vectorized SwiGLU and quantizers then took the Q8_0
  prompt from 178 / 494 to 212 / 603. At a context of 1045 positions, Q8_0: f16 cache 21.5 / 65 tok/s
  against 20.4 / 56 with the float32 cache, prompt 132 / 417 against 122 /
  393, cache 80 MiB instead of 160 for 2048 positions (measured before the
  prefetch). The 512 token file is the first 512 ids of
  /home/user/ref/long_tokens.txt; the older 633 token prompt text was not
  kept. Q5_0 experiment (scratch bench_q5, before the prefetch): expanding
  Q5_0 to packed Q8_0 at load was 6% faster for one token on one thread but
  1.55x the bytes and slower with 16 tokens; packed Q5_0 kept. float32 path:
  about 5 tok/s on one thread.
- What explained the speed (the investigation is in the commit messages): the
  row by row kernel was compute bound at 6 GB/s of weights, ggml's own
  vec_dot too when called from our harness; llama.cpp reaches 10 GB/s because
  on this CPU it repacks Q8_0/Q4_0 into an AMX buffer and runs 512-bit VNNI
  over 16 interleaved rows. Attention was 12 GMAC per 633 token prompt
  (forgetting the 32 layers once cost an hour): tiles, broadcast FMAs and the
  vectorized exp fixed it. Huge pages: +25% on the row kernel, so the packed
  copy asks for them; load takes 0.5 to 1.2 s from synchronous compaction.
  Generation kernels one token at a time were bound by the misses in flight,
  not by the unpack (in cache Q4_0 runs at 39 GB/s, streaming at 7.9): a
  `prefetcht0` per line 8 KiB ahead of the group block took Q4_0 from 7.9 to
  13.8 GB/s on a 400 MiB stream (the L3 is 260 MiB, so streaming tests need
  more than that) and Q8_0 from 12.3 to 16.0; 2 or 4 integer accumulators
  changed nothing. Measure micro benchmarks at -O3: at -O2 the compiler
  spills the unpacked chunks and the in-cache numbers are 3x off. Prompt
  phases, Q8_0, 512 tokens, 1 thread, 3.14 s: matmul 1.86 s, attention
  0.65, token steps 0.37, quantize 0.25 (instrumented copy of runner.cpp in
  the scratchpad, runner_phases.cpp). Token steps were std::exp in swiglu
  (82k calls per token): now exp_values, 0.18 s; quantizers vectorized with
  bit identical bytes (x * id rounded to nearest even, as nearbyint), 0.11
  s. The VNNI GEMM is bound by one zmm vpdpbusd per cycle (64 MAC/cycle,
  131 GMAC/s at 2.1 GHz). swiglu now differs from std::exp by a few ulp, so
  dumps changed; the float32 path against llama.cpp is 4.1e-6 (was 4.5e-6)
  and Q8_0 keeps the same 16 tokens as llama.cpp.
- Speculative decoding: `Runner::truncate(pos)`, `forward(..., all_logits)`
  (count rows of logits) and `init(model, n_ctx, ThreadPool&, ...)` so two
  runners share one pool (two pools spinning on four cores cut the speed to
  a tenth). `Speculative::round(last, out)`: the draft catches up with the
  tokens the target kept (m_tokens by position), proposes n_draft tokens,
  the target checks them in one pass, kept = longest agreeing prefix, out
  gets the kept proposals plus the target's token at the first disagreement.
  Draft files: /home/user/models/draft/ (bartowski Q8_0 and f16, the f32 copy,
  and q4_0-pure made with llama-quantize; the community Q4_0 mixes Q4_1).
  The draft on our runner is at 1.6e-6 from llama.cpp in float32 and 1.7e-2
  in Q8_0, 16 tokens identical. Costs behind the gain: target token 25 ms
  (1 thread), K + 1 token pass 29 to 52 ms, draft token 10.8 (Q8_0) / 7
  (Q4_0) ms; with 4 threads the draft (86 to 136 MiB, in L3) is bound by
  the ~450 fork/join tasks per token.
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
