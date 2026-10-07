# faisca

A small LLM runner written from scratch in C++17. A library first, with a
command line tool on top. No third party dependencies, plus the GPU driver once
the Vulkan backend lands.

The library does not use standard library containers, strings, smart pointers
or exceptions. It has its own two containers (`faisca::Array` and
`faisca::StringMap`), chosen by measurement: see [bench/README.md](bench/README.md).

`faisca` is a working name.

## Status

| Step | State |
| --- | --- |
| GGUF loader (memory mapped, zero-copy) | done |
| Tokenizer | next |
| CPU runner, float32, validated against a reference | planned |
| CPU optimization (threads, SIMD, quantized weights) | planned |
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
```

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
- `compare_with_gguf_py.py`: reads the same file with faisca and with gguf-py,
  the reference implementation, and compares every key, value, tensor shape,
  offset and tensor byte checksum. Works on real model files too.
- `gguf_malformed`: truncated and randomly damaged files under
  AddressSanitizer and UndefinedBehaviorSanitizer. A damaged file must be
  rejected with an error or stay fully inside its bounds.

## Layout

```
include/faisca/   public headers
src/              library
cli/              command line tool
tests/            tests and test tools
bench/            benchmarks of the container choices
```
