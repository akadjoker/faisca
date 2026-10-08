// Encodes an image with the faisca vision encoder and writes the image embeddings as raw
// float32 (n_tokens x n_proj), to compare with the reference's mtmd_get_output_embd.
//
// usage: vision_dump [--threads N] [--f32] [--trace trace.bin] [--layers N] <mmproj.gguf> <image> <embd.bin>
//
// --trace writes every traced tensor in the format of tests/reference/llama_layer_dump.cpp
// (and of llama_vision_dump --trace), for tests/compare_vision.py; --layers N keeps the
// per layer tensors of the first N layers only (a whole trace is about 8 GB).

#include "faisca/image.h"
#include "faisca/thread.h"
#include "faisca/vision_model.h"
#include "faisca/vision_runner.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
    int layers = 1 << 30;
    std::vector<std::string> order;
    std::map<std::string, Traced> tensors;
};

void trace(void* user, const char* name, int layer, const float* data, size_t count)
{
    Capture* capture = static_cast<Capture*>(user);
    if (layer >= capture->layers)
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

bool write_capture(const Capture& capture, const char* path)
{
    FILE* out = std::fopen(path, "wb");
    if (out == nullptr)
    {
        return false;
    }
    for (const std::string& key : capture.order)
    {
        const Traced& traced = capture.tensors.at(key);
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
    return true;
}

}

int main(int argc, char** argv)
{
    int threads = 1;
    bool f32 = false;
    const char* trace_path = nullptr;
    int layers = 1 << 30;
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
        {
            threads = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--f32") == 0)
        {
            f32 = true;
        }
        else if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc)
        {
            trace_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--layers") == 0 && i + 1 < argc)
        {
            layers = std::atoi(argv[++i]);
        }
        else
        {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() != 3)
    {
        std::fprintf(stderr, "usage: vision_dump [--threads N] [--f32] [--trace trace.bin] [--layers N] <mmproj.gguf> <image> <embd.bin>\n");
        return 2;
    }
    faisca::Error error;
    faisca::VisionModel model;
    faisca::Image image;
    faisca::ThreadPool pool;
    faisca::VisionRunner runner;
    const auto t0 = std::chrono::steady_clock::now();
    if (!model.load(positional[0], error, !f32) || !faisca::read_image(positional[1], image, error) || !pool.start(threads, error) ||
        !runner.init(model, pool, error))
    {
        std::fprintf(stderr, "vision_dump: %s\n", error.c_str());
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    Capture capture;
    capture.layers = layers;
    if (trace_path != nullptr)
    {
        runner.set_trace(trace, &capture);
    }
    faisca::Array<float> embeddings;
    if (!runner.encode(image, embeddings, error))
    {
        std::fprintf(stderr, "vision_dump: %s\n", error.c_str());
        return 1;
    }
    const auto t2 = std::chrono::steady_clock::now();
    if (trace_path != nullptr && !write_capture(capture, trace_path))
    {
        std::fprintf(stderr, "vision_dump: cannot write %s\n", trace_path);
        return 1;
    }
    FILE* f = std::fopen(positional[2], "wb");
    if (f == nullptr || std::fwrite(embeddings.data(), sizeof(float), embeddings.size(), f) != embeddings.size())
    {
        std::fprintf(stderr, "vision_dump: cannot write %s\n", positional[2]);
        return 1;
    }
    std::fclose(f);
    std::printf("%s: %u x %u, %u tokens of %u, load %.2f s, encode %.2f s, %d threads, %s\n", positional[1], image.width, image.height,
                model.hparams().n_tokens(), model.hparams().n_proj, std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count(), threads, model.quantized() ? "Q8_0" : "float32");
    return 0;
}
