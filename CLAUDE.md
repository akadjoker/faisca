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
4. CPU optimization: threads, SIMD, quantized weights. Next. Order agreed
   with the owner: quantized matvec with SIMD (Q8_0, Q4_0, Q4_K), threads for
   the prompt, KV cache in f16 or Q8; then speculative decoding with
   SmolLM2-135M as draft; then Vulkan; then the cache format.
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

`LlamaModel` + `Runner` (include/faisca/model.h, runner.h, ops.h): the `llama`
architecture in float32, one token at a time, one thread. F32 tensors in place,
F16 and Q8_0 dequantized at load (bit identical to gguf-py). No rope scaling,
no biases, no MoE yet: those GGUF keys are rejected with an error.

- Validated against llama.cpp b86d2f0 on SmolLM2-360M-Instruct with
  `tests/compare_layers.py`: 386 intermediate tensors per prompt, worst
  relative L2 error 4.5e-6 on a 16 token prompt and 2.1e-6 on a 1045 token
  prompt, greedy generation identical for 16 and 32 tokens. ASan/UBSan clean.
- Speed, one thread, float32: about 5 tokens per second on the 360M model on a
  2.1 GHz Xeon, 1.35 GB of dequantized weights. That is step 4's job.
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
