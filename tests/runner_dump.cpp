#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/runner.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// usage: runner_dump <model.gguf> <out.bin> <generate N> <token ids...>
//
// Same output as tests/reference/llama_layer_dump.cpp: every traced vector of
// every prompt token, assembled per name into [values, tokens], then N greedy
// tokens on stdout.

namespace
{

struct Traced
{
    size_t count = 0;
    std::vector<float> values;
};

struct Capture
{
    std::vector<std::string> order;
    std::map<std::string, Traced> tensors;
};

void trace(void* user, const char* name, int layer, const float* data, size_t count)
{
    Capture* capture = static_cast<Capture*>(user);
    std::string key(name);
    if (layer >= 0)
    {
        key += "-" + std::to_string(layer);
    }
    auto it = capture->tensors.find(key);
    if (it == capture->tensors.end())
    {
        capture->order.push_back(key);
        it = capture->tensors.emplace(key, Traced()).first;
        it->second.count = count;
    }
    it->second.values.insert(it->second.values.end(), data, data + count);
}

}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::fprintf(stderr, "usage: runner_dump <model.gguf> <out.bin> <generate N> <token ids...>\n");
        return 2;
    }

    const int generate = std::atoi(argv[3]);
    std::vector<int32_t> prompt;
    for (int i = 4; i < argc; ++i)
    {
        prompt.push_back(std::atoi(argv[i]));
    }

    faisca::Error error;
    faisca::GgufFile file;
    faisca::LlamaModel model;
    faisca::Runner runner;
    const uint32_t n_ctx = static_cast<uint32_t>(prompt.size() + static_cast<size_t>(generate) + 1);
    if (!file.open(argv[1], error) || !model.load(file, error) || !runner.init(model, n_ctx, error))
    {
        std::fprintf(stderr, "runner_dump: %s\n", error.c_str());
        return 1;
    }

    Capture capture;
    runner.set_trace(trace, &capture);
    faisca::Array<float> logits;
    if (!runner.forward(prompt.data(), prompt.size(), logits, error))
    {
        std::fprintf(stderr, "runner_dump: %s\n", error.c_str());
        return 1;
    }
    runner.set_trace(nullptr, nullptr);

    FILE* out = std::fopen(argv[2], "wb");
    if (out == nullptr)
    {
        std::fprintf(stderr, "runner_dump: cannot write '%s'\n", argv[2]);
        return 1;
    }
    for (const std::string& key : capture.order)
    {
        const Traced& traced = capture.tensors[key];
        const uint32_t name_length = static_cast<uint32_t>(key.size());
        const uint32_t dims = 2;
        const uint64_t ne[2] = {traced.count, traced.values.size() / traced.count};
        std::fwrite(&name_length, 4, 1, out);
        std::fwrite(key.data(), 1, key.size(), out);
        std::fwrite(&dims, 4, 1, out);
        std::fwrite(ne, 8, 2, out);
        std::fwrite(traced.values.data(), 4, traced.values.size(), out);
    }
    std::fclose(out);
    std::fprintf(stderr, "runner_dump: %zu tensors written\n", capture.order.size());

    std::printf("generated:");
    for (int g = 0; g < generate; ++g)
    {
        int32_t best = 0;
        for (size_t t = 1; t < logits.size(); ++t)
        {
            if (logits[t] > logits[static_cast<size_t>(best)])
            {
                best = static_cast<int32_t>(t);
            }
        }
        std::printf(" %" PRId32, best);
        if (!runner.forward(&best, 1, logits, error))
        {
            std::fprintf(stderr, "runner_dump: %s\n", error.c_str());
            return 1;
        }
    }
    std::printf("\n");
    return 0;
}
