// Lists a safetensors model (one file or an index with shards): name, dtype, shape and a 64-bit
// FNV-1a of the bytes of each tensor, sorted by name, for comparison with the reference reader.
// Usage: safetensors_dump <model.safetensors | model.safetensors.index.json>
#include "faisca/safetensors.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: safetensors_dump <model.safetensors | model.safetensors.index.json>\n");
        return 2;
    }
    faisca::Error error;
    faisca::SafetensorsModel model;
    if (!model.open(argv[1], error))
    {
        std::printf("safetensors_dump: %s\n", error.message);
        return 1;
    }
    std::vector<size_t> order(model.tensor_count());
    for (size_t i = 0; i < order.size(); ++i)
    {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return model.tensor(a).name < model.tensor(b).name; });
    for (size_t i : order)
    {
        const faisca::SafetensorsTensor& t = model.tensor(i);
        uint64_t hash = 0xcbf29ce484222325ull;
        for (uint64_t b = 0; b < t.n_bytes; ++b)
        {
            hash = (hash ^ t.data[b]) * 0x100000001b3ull;
        }
        std::string shape;
        for (uint32_t d = 0; d < t.n_dims; ++d)
        {
            shape += (d ? "," : "") + std::to_string(t.shape[d]);
        }
        std::printf("%.*s %.*s [%s] %016llx\n", static_cast<int>(t.name.size()), t.name.data(),
                    static_cast<int>(t.dtype_name.size()), t.dtype_name.data(), shape.c_str(),
                    static_cast<unsigned long long>(hash));
    }
    return 0;
}
