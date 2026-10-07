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

## What the numbers say

- **Parsing a model does not depend on the containers.** All three land within
  noise of each other. The time goes into walking the 100,257 length-prefixed
  strings of the vocabulary and merges: GGUF has no offset table, so the only
  way to find where such an array ends is to step through it.
- The faisca containers are faster in every container-level test here, by
  1.1x to 2.6x against ct and more against the standard library.
- In absolute terms the gain is small: tens of nanoseconds per lookup. It shows
  up in tokenizing long prompts, not in loading a model.
- One measurement was unstable and is left out of the table: `push_back` of
  100,000 records without `reserve` on `std::vector` ranged from 15 to 71 ns
  between runs, depending on how the allocator handled the large blocks.

These are numbers from one cloud VM. Run the benchmarks on your own hardware
before relying on them.
