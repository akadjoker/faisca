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

How the generation speed got there, one thread, Q8_0: float32 weights 4.7
tok/s; Q8_0 rows with an AVX2 kernel 7.8; block scales converted with F16C
13.4; packed groups of 16 rows with VNNI 24.6. The kernels are checked
against llama.cpp's own outputs on its own inputs (`matmul_check`, zero
error) and the thread splits never change a sum, so these numbers come with
bit identical results across thread counts.

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
