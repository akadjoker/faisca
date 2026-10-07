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
2. Tokenizer: in progress, see below.
3. CPU runner in float32, validated layer by layer against llama.cpp.
4. CPU optimization: threads, SIMD, quantized weights.
5. Vulkan compute backend (the owner chose Vulkan over CUDA and OpenGL).
6. Own cache format with offset tables, KV cache kept between turns and saved to
   disk, longest common prefix reuse, snapshots.

Target model: SmolLM2-360M-Instruct GGUF, then 1B to 3B models.

## Tokenizer status

Byte level BPE (`tokenizer.ggml.model` = gpt2) with the `gpt-2`, `phi-2` and
`smollm` pre-tokenizers, written to match llama.cpp token for token.

- Done: load from GGUF, special token partition, both splitters, BPE merges,
  decode with llama.cpp's clean_spaces passes.
- Validated against llama.cpp (commit b86d2f0) with a GPT-2 vocabulary:
  43633 cases and 1.6 M tokens per pre-tokenizer, 0 mismatches.
- Not done: the differential test with extra special tokens
  (`tests/make_tokenizer_gguf.py --special` still adds a text that already
  exists and llama.cpp rejects duplicates), unit tests, sanitizers, Windows
  build, speed comparison with llama.cpp, word cache, CLI command, README.
- Not validated yet: the real SmolLM2 vocabulary. Needs the model file.
- Known difference: code points above U+10FFFF from malformed UTF-8 make
  llama.cpp throw; faisca encodes them as U+FFFD.

## Commands

```
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
tests/run_all.sh                    # release + ASan/UBSan
python3 tests/compare_tokenizer.py build/tokenizer_dump <llama_tokenizer_dump> model.gguf --files text.txt
```

`tests/reference/llama_tokenizer_dump.cpp` links against a built llama.cpp.
