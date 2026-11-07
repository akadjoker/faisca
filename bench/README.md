# Benchmarks

These exist to answer one question with measurements instead of opinions:
which containers should the library use?

Three candidates run the same work on the same data:

- the C++ standard library,
- ct ([akadjoker/containers](https://github.com/akadjoker/containers)),
- the two containers written for faisca (`faisca::Array`, `faisca::StringMap`).

## Running them

```
cmake -S . -B build-bench -DFAISCA_BUILD_BENCH=ON -DFAISCA_CT_DIR=/path/to/containers
cmake --build build-bench -j

build-bench/bench_containers model.gguf some_text.txt
build-bench/bench_loader_std    model.gguf
build-bench/bench_loader_ct     model.gguf
build-bench/bench_loader_faisca model.gguf
```

Use a real model: the data comes from its vocabulary, merges and tensor names.
The optional text file is a stream of real words to look up.

Without `FAISCA_CT_DIR` only `bench_loader_std` and `bench_loader_faisca` are built.

## What is measured

`bench_containers` times the containers alone:

- arrays: `push_back` of the record the loader stores, with and without
  `reserve`, and copying a block into a fresh buffer;
- string maps: building the map, lookups that hit in random order, lookups
  that miss, and the words of a real text in their original order.

`bench_loader_*` is the real loader (`src/gguf.cpp`) compiled once per
candidate. `bench/adapters/` implements `faisca::Array` and `faisca::StringMap`
on top of each library, so nothing else differs between the executables.

## How it is kept honest

- Every candidate runs the same workload on the same data and must produce
  the same checksum, or the program stops.
- Candidates are interleaved round by round, so drift and noise hit all of
  them alike. The first round is a warm-up and is not counted.
- Both the best and the median time are printed.
- ct builds its map with a `find` before each `put`, because the loader has
  to detect duplicate names and ct has no insert-if-absent.

## Results on one machine

Intel Xeon 2.1 GHz (2 cores of a cloud VM), Linux, `-O3`, real GPT-2 vocabulary
(50257 tokens, 50000 merges) inside a 694 MiB synthetic model file with 290
tensors. Best time in nanoseconds per operation, GCC 13 / Clang 18.

| Arrays | std | ct | faisca |
| --- | --- | --- | --- |
| push 290 tensor records, no reserve | 8.8 / 7.9 | 4.9 / 4.5 | 3.7 / 2.8 |
| push 290 tensor records, reserved | 4.1 / 3.9 | 4.6 / 5.0 | 3.3 / 2.9 |
| push 1,000,000 `uint32_t`, no reserve | 1.04 / 0.64 | 0.75 / 0.76 | 0.36 / 0.36 |
| copy 50,000 `uint32_t` into a new buffer | 0.17 / 0.17 | 0.17 / 0.17 | 0.09 / 0.09 |

| String maps | std | ct | ct + 8-byte hash | faisca |
| --- | --- | --- | --- | --- |
| tokens: build | 98 / 96 | 31 / 29 | 26 / 25 | 12 / 15 |
| tokens: hit, random order | 67 / 58 | 33 / 28 | 29 / 26 | 27 / 23 |
| tokens: miss | 64 / 56 | 30 / 29 | 25 / 25 | 20 / 20 |
| tokens: words of a real text | 33 / 35 | 14.7 / 14.4 | 9.8 / 11.1 | 7.3 / 9.2 |
| tensor names: hit | 9.6 / 10.9 | 13.8 / 13.8 | 6.7 / 7.8 | 5.8 / 7.8 |
| tensor names: miss | 8.2 / 7.6 | 12.2 / 10.1 | 5.1 / 4.9 | 4.9 / 3.9 |

"ct + 8-byte hash" is `ct::HashMap` with a hasher that reads 8 bytes at a time
instead of ct's byte-at-a-time FNV-1a. On long keys that alone closes most of
the gap.

| Real loader | std | ct | faisca |
| --- | --- | --- | --- |
| parse header, metadata, tensor list (microseconds) | 213 / 175 | 195 / 170 | 192 / 166 |
| `find_tensor` by name (ns) | 10.1 / 11.2 | 14.3 / 12.5 | 7.9 / 7.3 |
| find metadata key (ns) | 6.7 / 8.2 | 14.2 / 12.9 | 7.4 / 7.1 |

## Against llama.cpp

`compare_ggml` reads the same file with ggml's GGUF loader (the one llama.cpp
uses) and with faisca, checks that both report the same keys, types, tensors,
offsets and sizes, and times them.

```
cmake -S . -B build-bench -DFAISCA_BUILD_BENCH=ON -DFAISCA_LLAMA_DIR=/path/to/llama.cpp
cmake --build build-bench -j --target compare_ggml
build-bench/compare_ggml model.gguf
```

Same machine and file as above, GCC 13, llama.cpp at commit b86d2f0, best of 300 rounds:

| | ggml | faisca | |
| --- | --- | --- | --- |
| parse from memory (microseconds) | 4575 | 253 | 18x |
| open from disk, page cache warm (microseconds) | 5871 | 367 | 16x |
| find a tensor by name (ns) | 752 | 7.8 | 96x |

ggml copies every string into its own heap allocation, faisca keeps views into
the mapped file. `gguf_find_tensor` is a linear search; llama.cpp's model
loader builds its own map on top of it, which this does not measure.

## Tokenizer against llama.cpp

`bench_tokenizer` loads the same vocabulary with llama.cpp and with faisca and
times encode, decode and load on real text. Every round runs both libraries on
the same input, their outputs are hashed, and the hashes must match.

```
cmake -S . -B build-bench -DFAISCA_BUILD_BENCH=ON -DFAISCA_LLAMA_DIR=/path/to/llama.cpp
cmake --build build-bench -j --target bench_tokenizer
build-bench/bench_tokenizer model.gguf text1.md text2.cpp ... [--rounds N]
```

Same machine as above (4 cores), GCC 13, `-O3`, llama.cpp at commit b86d2f0
built with `GGML_NATIVE`, SmolLM2-360M-Instruct vocabulary (49152 tokens),
six text files of 298 KB in total (llama.cpp's README and build guide,
`llama-vocab.cpp`, the tokenizer test inputs, a Python script and a chat
transcript), 99,067 tokens, best of 100 rounds.

| | llama.cpp | faisca | |
| --- | --- | --- | --- |
| encode whole files, word cache warm (ns/token) | 1108 | 40 | 27x |
| encode whole files, word cache cleared before each run (ns/token) | 1097 | 69 | 16x |
| encode line by line, 6691 calls (ns/token) | 2322 | 41 | 57x |
| decode whole files (ns/token) | 18.4 | 14.2 | 1.3x |
| `token_to_piece`, one token at a time (ns/token) | 19.5 | 14.6 | 1.3x |
| load the vocabulary from the GGUF (ms) | 79 | 13.7 | 5.8x |

Encode throughput on this text: 2.7 MB/s for llama.cpp, 75 MB/s for faisca.

Where the time went, and what changed, measured with callgrind on the same
text (instructions for the encode path alone, the load excluded):

| Step | Cold cache | Warm cache |
| --- | --- | --- |
| starting point: heap based BPE on every word | 147 M | 147 M |
| word cache: the tokens of every word seen, up to 65,536 words of at most 64 bytes | 147 M | 144 M |
| Unicode classes looked up once per code point, ASCII fast path in the UTF-8 decode | 128 M | 110 M |
| merges of words up to 64 bytes found by scanning a small array instead of a heap | 115 M | 110 M |

The word cache changes little in instruction counts within one pass over a
text, but most of its wins show up in the timings: the pass over a text goes
from 161 to 69 ns/token cold and 40 ns/token warm. One attempt was reverted:
pointing the words at the input text instead of re-encoding their code points
saved a copy but needed two more arrays per code point, and cost 10% more
instructions.

The decode side got a piece table built at load: the decoded bytes of every
token back to back, so a token is a `memcpy`. That is what llama.cpp does too,
and it took faisca from 21.5 to 14.2 ns/token, for 1 ms more at load.

## Runner against llama.cpp

Measured with `faisca generate` (prompt of 633 tokens from llama.cpp's README,
then greedy tokens) and `llama-bench` (pp512 and tg32) on the same machine:
2.1 GHz Sapphire Rapids VM with 4 vCPUs, AVX-512 VNNI and AMX, llama.cpp
b86d2f0 built with GGML_NATIVE (its CPU backend uses the AMX buffer here).
SmolLM2-360M-Instruct.

| | threads | llama.cpp | faisca |
| --- | --- | --- | --- |
| Q8_0 generation (tok/s) | 1 | 24.9 | 24.6 |
| Q8_0 generation (tok/s) | 4 | 71 | 83 |
| Q8_0 prompt (tok/s) | 1 | 255 | 142 |
| Q8_0 prompt (tok/s) | 4 | 143 (pp512), 893 (pp128) | 443 |
| Q4_0 generation (tok/s) | 1 | 38.8 | 31 |
| Q4_0 generation (tok/s) | 4 | 70 | 66 |
| Q4_0 prompt (tok/s) | 1 | 249 | 165 |
| Q4_0 prompt (tok/s) | 4 | 147 | 480 |

Laptop, 2026-11-07: Intel Core Ultra 7 165H (6 P cores, AVX2 and VNNI, no
AVX-512 or AMX), SmolLM2-360M-Instruct Q8_0, f16 KV on both. llama.cpp
(unsloth build, CPU backend libggml-cpu-alderlake) with `-dev none -ngl 0
-nopo 1`, so nothing goes to the GPU: pp512 and tg64 at depth 512, mean of 3.
faisca: `bench_runner --threads N --runs 3 --generate 64` on 512 tokens, then
64 greedy tokens, best of 3. Two alternated rounds per thread count, ranges:

| | threads | llama.cpp | faisca |
| --- | --- | --- | --- |
| Q8_0 generation at depth 512 (tok/s) | 1 | 36.0-36.8 | 48.9-49.1 |
| Q8_0 generation at depth 512 (tok/s) | 4 | 63.3-63.9 | 118.6-118.8 |
| Q8_0 generation at depth 512 (tok/s) | 6 | 66.6-68.9 | 137.0-138.1 |
| Q8_0 prompt 512 (tok/s) | 1 | 136-141 | 245-252 |
| Q8_0 prompt 512 (tok/s) | 4 | 412-459 | 831-844 |
| Q8_0 prompt 512 (tok/s) | 6 | 525-561 | 1025-1112 |

Q4_K_M (Q5_0 for the 960 wide matrices, Q4_K and Q6_K for `ffn_down`, Q8_0
for the embeddings and half of `attn_v`), Q4_0 and Q8_0 measured with a
harness over `Runner`: the first 512 tokens of the 1045 token prompt, then
32 greedy tokens, best of 3; llama-bench pp512 and tg32. The "before" column
is the same code without the software prefetch described below.

| | threads | llama.cpp | faisca before prefetch | faisca |
| --- | --- | --- | --- | --- |
| Q8_0 generation (tok/s) | 1 | 24.9 | 24.6 | 33.8 |
| Q8_0 generation (tok/s) | 4 | 71 | 76.5 | 98 |
| Q8_0 prompt (tok/s) | 1 | 255 | 166 | 212 |
| Q8_0 prompt (tok/s) | 4 | 143 | 494 | 603 |
| Q4_0 generation (tok/s) | 1 | 38.8 | 30.8 | 45 |
| Q4_0 generation (tok/s) | 4 | 70 | 89.7 | 144 to 183 |
| Q4_0 prompt (tok/s) | 1 | 249 | 177 | 207 |
| Q4_0 prompt (tok/s) | 4 | 147 | 558 | 570 to 599 |
| Q4_K_M generation (tok/s) | 1 | 19.2 | 23.6 | 40.3 |
| Q4_K_M generation (tok/s) | 4 | 54 | 78 | 118 |
| Q4_K_M prompt (tok/s) | 1 | 83 | 164 | 204 |
| Q4_K_M prompt (tok/s) | 4 | 266 | 492 | 550 |

The last column has the software prefetch, the vectorized SwiGLU and the
vectorized activation quantizers. Four thread numbers on this VM move by up
to 25% between runs (the Q4_0 rows show the spread of three runs), one
thread numbers by a few percent.

llama.cpp is slower on Q4_K_M than on Q8_0 here because Q5_0 has no AMX
kernel in its CPU backend.

**Why the generation kernels needed a software prefetch.** One input at a
time, the Q4_0 kernel streamed weights at 7 GB/s and Q8_0 at 11.5 while the
same kernels on data in cache run at 39 and 67 GB/s (-O3; at -O2 the
compiler spills the unpacked chunks and the in-cache numbers are 3x worse,
which sent the first measurements astray). So the unpack was not the cost.
Q4_0 spends 36 instructions per block of 4.5 cache lines where Q8_0 spends
24 per 8.5 lines, so the out of order window holds fewer outstanding lines
for Q4_0 and the memory pipeline runs below the 10 GB/s one core can pull
with the hardware prefetchers alone. Asking for the group block 8 KiB ahead
(one `prefetcht0` per line, measured on a 400 MiB stream, the L3 here is
260 MiB) fixes it for every type:

| 400 MiB stream, one token | no prefetch | 4 KiB ahead | 8 KiB | 16 KiB | 32 KiB |
| --- | --- | --- | --- | --- | --- |
| Q8_0 (GB/s) | 12.3 | 14.0 | 16.0 | 13.0 | 12.2 |
| Q4_0 (GB/s) | 7.9 | 13.3 | 13.8 | 13.5 | 11.4 |
| Q5_0 (GB/s) | 7.0 | 12.9 | 13.2 | 12.9 | 10.3 |

Splitting the `vpdpbusd` chain over 2 or 4 accumulators changed nothing, in
cache or streaming. The outputs are bit identical with and without the
prefetch.

**Where the prompt time goes**, Q8_0, 512 tokens, one thread, from an
instrumented copy of the runner: matrix products 1.86 s, attention 0.65,
per token steps 0.37, activation quantization 0.25 (3.14 s in all). The
per token steps were mostly `std::exp` in SwiGLU, 82 thousand calls per
token; with the vectorized exp they take 0.18 s, and the quantizers in
AVX-512 (bit identical bytes: the same product rounded to nearest even)
take 0.11 s. The products are bound by one 512-bit `vpdpbusd` per cycle, 64
multiply adds, 131 GMAC/s at 2.1 GHz.

**SmolLM2-1.7B-Instruct Q4_K_M** (HuggingFaceTB's file: 24 blocks, 2048 wide,
32 heads, Q4_K and Q6_K everywhere, 1005 MiB), same harness and machine:

| | threads | llama.cpp | faisca |
| --- | --- | --- | --- |
| generation (tok/s) | 1 | 5.4 | 9.0 |
| generation (tok/s) | 4 | 16.5 | 28.2 |
| prompt (tok/s) | 1 | 55 | 45 |
| prompt (tok/s) | 4 | 169 | 125 |

Validation of the 1.7B: matmul_check gives zero error on all 145 matrices
checked (120 Q4_K, 25 Q6_K); the float32 path is 1.8e-5 from llama.cpp
(block 22 of this model is numerically sensitive: the quantized runs of both
faisca and llama.cpp sit at 0.9 to 1.0 relative L2 from the float32 run
there, with 16 greedy tokens identical on every side, so the activation
quantization is the cause, not the kernels); the Vulkan kernels on llvmpipe
are within 4e-7 per matrix.

**Llama 3.2 1B Instruct Q8_0** (bartowski's file, 16 blocks, 2048 wide, 32
heads over 8 kv heads, a vocabulary of 128256, rope frequency factors,
1252 MiB), same harness: generation 10.0 tok/s with 1 thread and 30.7 to
36.6 with 4 (three runs alternated with llama-bench tg32 at 29.2 to 31.3),
prompt 68 / 250 (llama-bench pp512 with 4 threads 297, AMX). Generation
phases per token with 4 threads: layer products 21.6 ms (1.03 GB, 48 GB/s),
the 266 MiB output matrix 5.5 ms, attention 1.4, quantization 0.4, token
steps 0.6. A first measurement of 24.9 tok/s was VM noise, not a gap.

**Load time.** Packing the weights into the interleaved layout is now done
by every thread of the pool, in chunks of 1024 rows; the Llama 3.2 file
goes from 6.7 s on one thread to 0.38 s on four. The single thread figure
moves between 1.5 and 20 s from run to run: the packing itself is about 1.3
s per GB, the rest is the kernel compacting memory for the huge pages the
packed copy asks for, synchronous at the first touch of each page, so it
depends on how fragmented memory is; packing in parallel also spreads that
compaction over the threads.
Validation: matmul_check zero error on 97 matrices, float32 path 2.6e-6 from
llama.cpp with 16 tokens and 2.4e-6 with 512, Vulkan 2.6e-6, tokens
identical; the llama-bpe tokenizer at 0 mismatches over 39941 cases (2.69 M
tokens) of the text corpus.

**Speculative decoding with SmolLM2-135M as draft.** Measured with
`speculative_check` (SmolLM2-360M Q8_0 target, chat prompt of 16 tokens, 64
generated unless said otherwise; the sequences are identical to plain greedy
decoding in every run):

| draft | threads | tokens per round | plain tok/s | speculative tok/s | gain |
| --- | --- | --- | --- | --- | --- |
| 135M Q8_0 | 1 | 1 | 34.1 | 41.0 | 1.20x |
| 135M Q8_0 | 1 | 4 | 32.1 | 37.9 | 1.18x |
| 135M Q8_0 | 1 | 8 | 34.5 | 25.6 | 0.74x |
| 135M Q8_0 | 4 | 2 | 102 | 107 | 1.05x |
| 135M Q4_0 | 1 | 2 | 35.6 | 45.7 | 1.28x |
| 135M Q4_0 | 1 | 3 | 36.3 | 50.7 | 1.39x |
| 135M Q4_0 | 1 | 4 | 34.8 | 47.2 | 1.36x |
| 135M Q4_0 | 4 | 1 | 109 | 127 | 1.16x |
| 135M Q4_0 | 4 | 3 | 112 | 127 | 1.13x |
| 135M Q4_0 | 4 | 3, 128 tokens | 114 | 117 | 1.02x |
| 135M Q4_0 | 1 | 3, prompt of 512 | 29.2 | 32.0 | 1.09x |
| 135M Q4_0 | 4 | 3, prompt of 512 | 101 | 94 | 0.92x |

Acceptance runs from 83% with one proposal per round to 48% with eight; 59
to 65% at the longer prompt and generation. The costs behind the table, one
thread: a target token 25 ms, a target pass over K + 1 tokens with all the
logits 29, 34, 43, 52 ms for K = 1, 2, 4, 6 (about 4.5 ms per extra token,
compute), a draft token 10.8 ms in Q8_0 (136 MiB, 0.43 of the target) and
about 7 in Q4_0. With four threads the target token takes 9 ms and the
draft 2.5, but the draft is then bound by the fork and join of its 450
small tasks per token, not by its 86 MiB, so the gain shrinks. The draft is
too close to the target in cost for a large gain: the method pays off with
a draft ten times smaller, or when the target is bound by memory and the
draft fits in cache. The Q4_0 draft was made with llama-quantize from the
float32 copy of the f16 file; the community Q4_0 file mixes Q4_1, which the
loader does not read. Note for whoever measures this: two runners with two
thread pools of four threads on four cores spin against each other and the
numbers collapse to a tenth; the runners must share one pool.

**AMX for the prompt, measured and set aside.** The tile unit itself is fast
(`tdpbsud` 16 x 16 x 32 in 6 ns, 1.3 TMAC/s), but Q8_0 has a scale per block
of 32, so every block needs its int32 tile stored and scaled in float32 for
16 tokens (about 110 vector instructions), and that epilogue is where the
time goes: on a 400 MiB stream of weights with 16 tokens the VNNI GEMM does
124 GMAC/s, the AMX kernel 145, and 150 with two tile sets so the next
product overlaps the epilogue, 1.25x and bit identical outputs. With four
processes both run memory bound at about 10 GB/s each. The matrix products
are 59% of the prompt time, so AMX would buy about 10% end to end for a per
thread tile configuration, a permission syscall and a Windows path; not
taken.

Expanding Q5_0 to packed Q8_0 at load (exact values, 1.55x the bytes) was
measured before the prefetch: 6% faster for one token on one thread, slower
with 16 tokens; packed Q5_0 is kept.

AVX-VNNI build (`-mavx2 -mfma -mf16c -mavxvnni`, no AVX-512) on the same
machine, the harness over `Runner` as above (512 tokens then 32, best of 3,
two runs each, the binaries interleaved in the same minutes). Three states
of that build: Q5_0 and the K-quants converted to float32 at load, the
first 256-bit kernels that walked the weights once per token, and the GEMM
templates that unpack each chunk once for 4 inputs. The AVX-512 column is
from the same session.

| | threads | float32 conversion | per token kernels | GEMM of 4 | AVX-512 |
| --- | --- | --- | --- | --- | --- |
| Q4_K_M 360M prompt (tok/s) | 1 | 17.2 to 17.5 | 80 to 82 | 126 to 132 | 176 to 204 |
| Q4_K_M 360M prompt (tok/s) | 4 | 64 | 239 to 247 | 338 to 344 | 573 to 628 |
| Q4_K_M 360M generation (tok/s) | 1 | 6.5 to 6.8 | 33.5 to 34 | 35.5 to 35.7 | 39.8 to 40.0 |
| Q4_K_M 360M generation (tok/s) | 4 | 21.5 to 21.8 | 94 to 114 | 94 to 99 | 111 to 134 |
| Q8_0 360M prompt (tok/s) | 1 | | 81 to 85 | 146 to 147 | 212 |
| Q8_0 360M prompt (tok/s) | 4 | | 355 to 422 | 419 to 442 | 603 |
| Q8_0 360M generation (tok/s) | 1 | | 27 to 29 | 30 to 31 | 33.8 |
| Q8_0 360M generation (tok/s) | 4 | | 85 to 92 | 80 to 83 | 98 |
| Q4_0 360M prompt (tok/s) | 1 | | 125 to 128 | 159 to 163 | 207 |
| Q4_0 360M prompt (tok/s) | 4 | | 402 to 421 | 427 to 545 | 570 to 599 |
| Q4_0 360M generation (tok/s) | 1 | | 43 to 45 | 42 to 43 | 45 |
| Q4_0 360M generation (tok/s) | 4 | | 150 to 153 | 133 to 167 | 144 to 183 |
| 1.7B Q4_K_M prompt (tok/s) | 1 | | 17.2 | 33.8 | 43.5 |
| 1.7B Q4_K_M prompt (tok/s) | 4 | | 64.6 | 114 | 134 |
| 1.7B Q4_K_M generation (tok/s) | 1 | | 8.2 | 8.4 | 9.2 |
| 1.7B Q4_K_M generation (tok/s) | 4 | | 27.0 | 27.8 | 24.9 |

The Q8_0 and Q4_0 AVX-512 numbers are the ones of the table above. The
GEMM takes the prompt 1.2x (Q4_0, whose unpack is cheap) to 2x (1.7B)
over the per token kernels and generation is unchanged, since one input
goes through the same template. A GEMM of 8 inputs was measured in the
same runs and lost 10 to 25% of the prompt to register spills (16 ymm hold
the two halves of the sums for 4 inputs plus the chunk), so 4 is kept.
The 1.7B float32 conversion column is missing: that file converted to
float32 does not fit the memory of this VM.

How the generation speed got there, one thread, Q8_0: float32 weights 4.7
tok/s; Q8_0 rows with an AVX2 kernel 7.8; block scales converted with F16C
13.4; packed groups of 16 rows with VNNI 24.6. The kernels are checked
against llama.cpp's own outputs on its own inputs (`matmul_check`, zero
error) and the thread splits never change a sum, so these numbers come with
bit identical results across thread counts.

## Qwen and Gemma against llama.cpp

`bench_runner` (bench/bench_runner.cpp: the first 512 tokens of a text
through `Runner::forward`, then 32 greedy tokens, best of 3) against
`llama-bench` pp512 and tg32, same machine, the Q8_0 files of Qwen2.5-0.5B-
Instruct, Qwen3-0.6B and gemma-3-1b-it. llama.cpp b86d2f0's CPU backend
repacks Q8_0 into an AMX buffer and that path aborts on this VM ("AMX is
not ready to be used"), so it was built with GGML_AMX_INT8, GGML_AMX_TILE
and GGML_AMX_BF16 off; the SmolLM2 tables above used its AMX build.

| | threads | llama.cpp | faisca |
| --- | --- | --- | --- |
| Qwen2.5-0.5B generation (tok/s) | 1 | 13.7 | 17.9 |
| Qwen2.5-0.5B generation (tok/s) | 4 | 41.7 | 54.5 |
| Qwen2.5-0.5B prompt (tok/s) | 1 | 95 | 165 |
| Qwen2.5-0.5B prompt (tok/s) | 4 | 328 | 458 |
| Qwen3-0.6B generation (tok/s) | 1 | 11.5 | 12.5 |
| Qwen3-0.6B generation (tok/s) | 4 | 32.1 | 38.2 |
| Qwen3-0.6B prompt (tok/s) | 1 | 72 | 96 |
| Qwen3-0.6B prompt (tok/s) | 4 | 245 | 279 |
| Gemma 3 270M generation (tok/s) | 1 | 23.5 | 34.2 |
| Gemma 3 270M generation (tok/s) | 4 | 67 | 77 |
| Gemma 3 270M prompt (tok/s) | 1 | 221 | 262 |
| Gemma 3 270M prompt (tok/s) | 4 | 643 | 641 |
| Gemma 3 1B generation (tok/s) | 1 | 6.5 | 8.7 |
| Gemma 3 1B generation (tok/s) | 4 | 20.2 | 24.1 |
| Gemma 3 1B prompt (tok/s) | 1 | 43 | 61 |
| Gemma 3 1B prompt (tok/s) | 4 | 135 | 192 |
| Gemma 3 4B generation (tok/s) | 1 | 1.7 | 2.3 |
| Gemma 3 4B generation (tok/s) | 4 | 6.0 | 7.2 |
| Gemma 3 4B prompt (tok/s) | 1 | 10.3 | 18.1 |
| Gemma 3 4B prompt (tok/s) | 4 | 38 | 59 |

Qwen2.5-0.5B reads 500 MiB of weights per token, 136 of them the
151936 x 896 output matrix; Gemma 3 1B reads 1013 MiB, 302 of them the
262144 x 1152 output matrix, which bounds its generation; Gemma 3 4B reads
3.9 GiB (best of 2 runs there). The 270M (271 MiB) sits in the L3 of this
machine, so its generation is bound by the fork/join of the pool, not by
the memory, and the prompt ties llama-bench at four threads. The load of
the Qwen2.5 file takes 4 s on one thread and 0.2 s with four, the 4B 25 s
and 1.3 s (packing into huge pages).

## Gemma 3 vision against llama.cpp's mtmd

The SigLIP encoder of gemma-3-4b-it (27 layers over 4096 patches, 400M
parameters) on test-1.png, `vision_dump` (Q8_0 matrices, quantized
activations) against `llama_vision_dump` (the F16 mmproj with mtmd's
defaults: f16 weights, flash attention, the same no AMX build), encode
time of the image in seconds.

| | threads | llama.cpp | faisca |
| --- | --- | --- | --- |
| encoder (s) | 1 | 115 | 74 |
| encoder (s) | 4 | 31 | 20.5 |

Where the time went: the first version took 374 s on one thread.
`attention_head` (one query against the tiled keys) is bound by the L3
on 4096 keys: each query streams the head's 1.2 MB of key tiles and, with
the values strided by n_embd, 20k cache lines of V, 2.5 GMAC/s in a micro
benchmark (scratch bench_attn, 5.0 with V packed per head).
`attention_block` runs 16 queries per pass over the keys and the values
(each query's FMAs in the order of `attention_head`, so bit identical):
24.5 GMAC/s; 8 queries give 13.7, 12 give 23.6, 20 and 24 spill and fall
to 10; the AVX2 version with 4 queries and 16 wide value chunks is 2.6x
over its single query kernel. The FFN's bias and GELU ran on the main
thread (476 M tanhf per image, 21 s): they are row jobs now and the GELU
is vectorized on the exp of exp_values (x / (1 + exp(-2y)), within 1.2e-7
of tanhf relative to the input). Phases at 4 threads before the GELU work,
with another job on the machine: attention 26 s, the six matmuls 18, GELU
21, norms 0.5, tiling 0.5. The block kernel is bound by the broadcast
loads of the queries (one per FMA); a key tiled kernel that keeps K and V
of a block in L2 is the next step, and the matmuls (1.7 TMAC per image
against 1.0 TMAC of attention) are at the VNNI rate of the text prompt.

`faisca describe` with 4 threads on the 4B Q8_0: load 9.5 s, encode
20.9 s, the 271 token prompt 4.1 s (66 tok/s), generation 7.7 tok/s.

## Whisper against whisper.cpp

`faisca transcribe` against `whisper-cli` 1.9.5 (stock build, CPU, no flash
attention) on jfk.wav (11 s, 24 to 26 tokens), the same machine, best of
3, wall time of the whole process including the model load. faisca runs
the matrices on Q8_0 quantized at load; whisper-cli with its f16 files and
with its Q8_0 files.

| | threads | faisca Q8_0 | whisper-cli f16 | whisper-cli Q8_0 |
| --- | --- | --- | --- | --- |
| tiny (s) | 1 | 1.74 (load 0.26) | 2.11 | 1.67 |
| tiny (s) | 4 | 0.74 (load 0.24) | 0.81 | 0.71 |
| base (s) | 1 | 4.74 (load 0.47) | 4.17 | 3.22 |
| base (s) | 4 | 1.65 (load 0.48) | 1.48 | 1.14 |
| small (s) | 1 | 11.1 (load 1.58) | 12.9 | 8.74 |
| small (s) | 4 | 4.53 (load 1.54) | 4.09 | 2.94 |

Without the load (the time the CLI reports for the audio): tiny 1.47 /
0.48 s, base 4.35 / 1.14 s, small 9.45 / 3.25 s with 1 / 4 threads, that
is 7.5 / 23x, 2.5 / 9.7x and 1.2 / 3.4x real time. The load converts the
f16 file to float32 and quantizes and packs the matrices on one thread,
0.26 to 1.6 s; whisper-cli reads its files in 0.08 to 0.4 s.

Where the time goes (trace timestamps, jfk.wav, Q8_0, 1 / 4 threads):

| phase | tiny | small |
| --- | --- | --- |
| mel spectrogram | 0.028 / 0.027 | 0.027 / 0.028 |
| conv1 and conv2 | 0.51 / 0.14 | 1.51 / 0.39 |
| encoder attention and projection | 0.61 / 0.16 | 4.30 / 1.13 |
| encoder Q K V and FFN matmuls | 0.17 / 0.05 | 1.60 / 0.50 |
| encoder GELU | 0.20 / 0.05 | 1.14 / 0.33 |
| cross keys and values | 0.04 / 0.02 | 0.38 / 0.14 |
| decoder (24 to 26 tokens) | 0.08 / 0.03 | 0.51 / 0.15 |

Before the encoder keys went through the tiled attention kernel the
attention took 15.0 s of the 20.7 s of small on one thread. The next
candidates are the convolutions (a scalar loop, 1.5 s) and the GELU
(tanhf per value, 1.1 s); the matmuls are already the VNNI GEMM.

## Pocket TTS against pockettts.cpp

`faisca speak` against `pocket-tts` (pockettts.cpp, stock build, CPU, which
picks half the logical cores: 2 threads here) on the English pack, voice
alba, seed 1, the matrices on Q8_0 on both sides (the convolutions stay
float32 on both). Time of the synthesis alone, as both tools report it;
the load is 0.2 s for faisca and 0.15 s for pocket-tts.

| text | audio | faisca 1 thread | 2 threads | 4 threads | pocket-tts (2 threads) |
| --- | --- | --- | --- | --- | --- |
| one sentence (14 tokens, 37 frames) | 2.9 s | 1.3 s | 0.76 s | 0.51 s | 1.45 s |
| three sentences (87 tokens, 177 frames) | 13.9 s | 6.8 s | 3.7 s | 2.2 s | 6.9 s |

Per frame on the one sentence text (trace timestamps, 1 / 2 / 4 threads):
flow LM step 13.2 / 8.8 / 6.2 ms, Mimi decode 22 / 11.5 / 7.4 ms;
pocket-tts reports 13 ms per step and 25 ms per frame of decode on its 2
threads. The step is 6 layers of 1024 with 72 MiB of Q8_0 weights read
per token plus the flow head, so it is bound by memory and by the
fork/join of the thread pool, like the LLM generation. The decode was 108
ms per frame on one thread with a plain loop over the SEANet taps; the
convolutions are a GEMM of 7.5 GMAC for 3 s of audio and now run through
`conv1d_rows`, which keeps 4 output channels x 64 positions in AVX-512
registers (4 x 32 in AVX2) and loads every input vector once per tap:
22 ms per frame. The transposed convolutions of the upsampling stages are
run as stride separate 2 tap convolutions, one per output phase, with the
kernels rearranged at load.

## What the numbers say

- **Parsing a model does not depend on the containers.** All three land within
  noise of each other. The time goes into walking the 100,257 length-prefixed
  strings of the vocabulary and merges: GGUF has no offset table, so the only
  way to find where such an array ends is to step through it.
- The faisca containers are faster in every container-level test here, by
  1.1x to 2.6x against ct and more against the standard library.
- In absolute terms the gain is small: tens of nanoseconds per lookup. It shows
  up in tokenizing long prompts, not in loading a model.
- The tokenizer's lead over llama.cpp is algorithmic, not container
  level: llama.cpp rebuilds a `std::forward_list` of fragments, a
  `std::priority_queue` and several `std::string`s per word, faisca reuses its
  buffers and remembers words. Most text repeats its words.
- One measurement was unstable and is left out of the table: `push_back` of
  100,000 records without `reserve` on `std::vector` ranged from 15 to 71 ns
  between runs, depending on how the allocator handled the large blocks.

These are numbers from one cloud VM. Run the benchmarks on your own hardware
before relying on them.
