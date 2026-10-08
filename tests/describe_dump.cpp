// Runs the language model on a prompt with one image, as faisca describe does, with a
// trace of the chosen tensors, and prints the greedy tokens. The prompt is given as token
// ids with the word "image" where the image embeddings go:
//
//   describe_dump [--dequantize] [--kv-f32] [--threads N] [--keep name,name] <model.gguf>
//                 <mmproj.gguf> <image> <out.bin> <generate N> <ids...> image <ids...>
//
// out.bin has the format of runner_dump; --keep lists the traced names (default inp_embd,
// l_out and result_norm, which keep the file small on a 4B model with 256 image tokens).

#include "faisca/gguf.h"
#include "faisca/image.h"
#include "faisca/model.h"
#include "faisca/runner.h"
#include "faisca/thread.h"
#include "faisca/vision_model.h"
#include "faisca/vision_runner.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace
{

struct Traced
{
    size_t count = 0;
    std::vector<float> values;
};

struct Capture
{
    std::set<std::string> keep;
    std::vector<std::string> order;
    std::map<std::string, Traced> tensors;
};

void trace(void* user, const char* name, int layer, const float* data, size_t count)
{
    Capture* capture = static_cast<Capture*>(user);
    if (capture->keep.count(name) == 0)
    {
        return;
    }
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
    bool dequantize = false;
    bool kv_f16 = true;
    int threads = 1;
    Capture capture;
    capture.keep = {"inp_embd", "l_out", "result_norm"};
    while (argc > 1 && argv[1][0] == '-')
    {
        if (std::strcmp(argv[1], "--dequantize") == 0)
        {
            dequantize = true;
        }
        else if (std::strcmp(argv[1], "--kv-f32") == 0)
        {
            kv_f16 = false;
        }
        else if (std::strcmp(argv[1], "--threads") == 0 && argc > 2)
        {
            threads = std::atoi(argv[2]);
            ++argv;
            --argc;
        }
        else if (std::strcmp(argv[1], "--keep") == 0 && argc > 2)
        {
            capture.keep.clear();
            std::string list(argv[2]);
            size_t start = 0;
            while (start <= list.size())
            {
                const size_t comma = list.find(',', start);
                capture.keep.insert(list.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
                start = comma == std::string::npos ? list.size() + 1 : comma + 1;
            }
            ++argv;
            --argc;
        }
        else
        {
            std::fprintf(stderr, "describe_dump: unknown option '%s'\n", argv[1]);
            return 2;
        }
        ++argv;
        --argc;
    }
    if (argc < 7)
    {
        std::fprintf(stderr, "usage: describe_dump [--dequantize] [--kv-f32] [--threads N] [--keep names] <model.gguf> "
                             "<mmproj.gguf> <image> <out.bin> <generate N> <ids...> image <ids...>\n");
        return 2;
    }
    const int generate = std::atoi(argv[5]);
    std::vector<int32_t> head;
    std::vector<int32_t> tail;
    bool after_image = false;
    for (int i = 6; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "image") == 0)
        {
            after_image = true;
            continue;
        }
        (after_image ? tail : head).push_back(std::atoi(argv[i]));
    }
    if (!after_image || head.empty() || tail.empty())
    {
        std::fprintf(stderr, "describe_dump: the prompt needs ids, the word image, then ids\n");
        return 2;
    }

    faisca::Error error;
    faisca::GgufFile file;
    faisca::LlamaModel model;
    faisca::ThreadPool pool;
    faisca::Runner runner;
    faisca::VisionModel vision;
    faisca::VisionRunner encoder;
    faisca::Image image;
    const uint32_t n_ctx = static_cast<uint32_t>(head.size() + tail.size() + 4096 + static_cast<size_t>(generate) + 1);
    if (!file.open(argv[1], error) || !pool.start(threads, error) || !model.load(file, error, dequantize, &pool) ||
        !runner.init(model, n_ctx, pool, error, kv_f16) || !vision.load(argv[2], error, !dequantize) ||
        !encoder.init(vision, pool, error) || !faisca::read_image(argv[3], image, error))
    {
        std::fprintf(stderr, "describe_dump: %s\n", error.c_str());
        return 1;
    }
    faisca::Array<float> embeddings;
    if (!encoder.encode(image, embeddings, error))
    {
        std::fprintf(stderr, "describe_dump: %s\n", error.c_str());
        return 1;
    }
    const size_t n_image = vision.hparams().n_tokens();

    runner.set_trace(trace, &capture);
    faisca::Array<float> logits;
    if (!runner.forward(head.data(), head.size(), logits, error) ||
        !runner.forward_embeddings(embeddings.data(), n_image, error) ||
        !runner.forward(tail.data(), tail.size(), logits, error))
    {
        std::fprintf(stderr, "describe_dump: %s\n", error.c_str());
        return 1;
    }
    runner.set_trace(nullptr, nullptr);

    FILE* out = std::fopen(argv[4], "wb");
    if (out == nullptr)
    {
        std::fprintf(stderr, "describe_dump: cannot write '%s'\n", argv[4]);
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
    std::fprintf(stderr, "describe_dump: %zu tensors written, %zu prompt tokens\n", capture.order.size(),
                 head.size() + n_image + tail.size());

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
        if (g + 1 < generate && !runner.forward(&best, 1, logits, error))
        {
            std::fprintf(stderr, "describe_dump: %s\n", error.c_str());
            return 1;
        }
    }
    std::printf("\n");
    return 0;
}
