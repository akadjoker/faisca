#include "faisca/triposr.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct Dump
{
    std::string dir;
};

void write_trace(void* user, const char* name, int layer, const float* data, size_t count)
{
    const Dump& dump = *static_cast<const Dump*>(user);
    std::string path = dump.dir + "/" + name + (layer >= 0 ? std::to_string(layer) : std::string()) + ".f32";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr || std::fwrite(data, sizeof(float), count, f) != count)
    {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
    }
    if (f != nullptr)
    {
        std::fclose(f);
    }
}

double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: triposr_dump <model.ckpt> <reference dir with rgb_cond.f32> <out dir> [threads]\n");
        return 2;
    }
    const int threads = argc > 4 ? std::atoi(argv[4]) : static_cast<int>(std::thread::hardware_concurrency());

    const size_t size = faisca::TripoModel::IMAGE_SIZE;
    std::vector<float> rgb(size * size * 3);
    const std::string rgb_path = std::string(argv[2]) + "/rgb_cond.f32";
    FILE* f = std::fopen(rgb_path.c_str(), "rb");
    if (f == nullptr || std::fread(rgb.data(), sizeof(float), rgb.size(), f) != rgb.size())
    {
        std::fprintf(stderr, "cannot read %s\n", rgb_path.c_str());
        return 1;
    }
    std::fclose(f);

    faisca::Error error;
    auto start = std::chrono::steady_clock::now();
    faisca::TripoModel model;
    if (!model.open(argv[1], error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "open %.3f s\n", seconds_since(start));

    faisca::ThreadPool pool;
    faisca::TripoRunner runner;
    if (!pool.start(threads, error) || !runner.init(model, pool, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    Dump dump = {argv[3]};
    runner.set_trace(write_trace, &dump);

    start = std::chrono::steady_clock::now();
    if (!runner.encode_image(rgb.data(), error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "image encoder %.3f s with %d threads (traces included)\n", seconds_since(start), threads);
    start = std::chrono::steady_clock::now();
    if (!runner.build_triplane(error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "triplane %.3f s (traces included)\n", seconds_since(start));
    return 0;
}
