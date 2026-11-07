// DitCudaRunner over tests/dit_network_reference.py's passes (several chunks, 3 timesteps each,
// the self attention window rolling), against its float64 run, next to PyTorch's bf16 run.
// Usage: cuda_dit_network <reference dir>
#include "faisca/omnidreams_cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

template <typename T>
bool read_file(const std::string& path, std::vector<T>& out, size_t count)
{
    out.resize(count);
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr)
    {
        std::printf("cannot open %s\n", path.c_str());
        return false;
    }
    const bool ok = std::fread(out.data(), sizeof(T), count, file) == count;
    std::fclose(file);
    if (!ok)
    {
        std::printf("%s is too short\n", path.c_str());
    }
    return ok;
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: cuda_dit_network <reference dir>\n");
        return 2;
    }
    const std::string dir = std::string(argv[1]) + "/";
    unsigned t = 0, h = 0, w = 0, blocks = 0, window = 0, passes = 0;
    FILE* list = std::fopen((dir + "passes.txt").c_str(), "r");
    if (list == nullptr || std::fscanf(list, "%u %u %u %u %u %u", &t, &h, &w, &blocks, &window, &passes) != 6)
    {
        std::printf("cannot read %spasses.txt\n", dir.c_str());
        return 1;
    }
    const size_t L = size_t(t) * h * w;
    const size_t D = faisca::DIT_WIDTH;

    std::vector<std::vector<uint16_t>> storage;
    storage.reserve(64 + 24 * blocks);
    const auto load = [&](const std::string& name, size_t count) -> const uint16_t* {
        storage.emplace_back();
        return read_file(dir + name + ".bf16", storage.back(), count) ? storage.back().data() : nullptr;
    };
    faisca::DitHostWeights weights = {};
    weights.x_embedder = load("x_embedder", D * 68);
    weights.hdmap_embedder = load("hdmap_embedder", D * 64);
    weights.t_linear1 = load("t_linear1", D * D);
    weights.t_linear2 = load("t_linear2", 3 * D * D);
    weights.t_norm = load("t_norm", D);
    weights.final_mod_down = load("final_mod_down", 256 * D);
    weights.final_mod_up = load("final_mod_up", 2 * D * 256);
    weights.final_linear = load("final_linear", 64 * D);
    weights.text_proj = load("text_proj", size_t(1024) * 100352);
    weights.text_proj_bias = load("text_proj_bias", 1024);
    weights.block_count = blocks;
    bool ok = true;
    for (unsigned b = 0; b < blocks; ++b)
    {
        const std::string p = "b" + std::to_string(b) + "_";
        faisca::DitHostBlock& block = weights.blocks[b];
        block.self_q = load(p + "self_q", D * D);
        block.self_k = load(p + "self_k", D * D);
        block.self_v = load(p + "self_v", D * D);
        block.self_out = load(p + "self_out", D * D);
        block.self_q_norm = load(p + "self_q_norm", 128);
        block.self_k_norm = load(p + "self_k_norm", 128);
        block.cross_q = load(p + "cross_q", D * D);
        block.cross_k = load(p + "cross_k", D * 1024);
        block.cross_v = load(p + "cross_v", D * 1024);
        block.cross_out = load(p + "cross_out", D * D);
        block.cross_q_norm = load(p + "cross_q_norm", 128);
        block.cross_k_norm = load(p + "cross_k_norm", 128);
        block.mlp_up = load(p + "mlp_up", 8192 * D);
        block.mlp_down = load(p + "mlp_down", 8192 * D);
        for (int i = 0; i < 3; ++i)
        {
            block.mod_down[i] = load(p + "mod_down" + std::to_string(i), 256 * D);
            block.mod_up[i] = load(p + "mod_up" + std::to_string(i), 3 * D * 256);
        }
    }
    for (const auto& s : storage)
    {
        ok = ok && !s.empty();
    }
    std::vector<uint16_t> text;
    ok = ok && read_file(dir + "text.bf16", text, size_t(512) * 100352);
    if (!ok)
    {
        return 1;
    }

    faisca::Error error;
    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::printf("cuda_dit_network: skipped, %s\n", error.message);
        return 77;
    }
    faisca::DitConfig config;
    config.frames = t;
    config.height = h;
    config.width = w;
    config.window_chunks = window;
    faisca::DitCudaRunner runner;
    if (!runner.init(device, weights, config, error) || !runner.set_text(text.data(), error))
    {
        std::printf("cuda_dit_network: %s\n", error.message);
        return 1;
    }
    bool better = true;
    std::vector<float> out(L * 64), torch;
    std::vector<double> reference;
    std::vector<uint16_t> tokens, hdmap;
    for (unsigned i = 0; i < passes; ++i)
    {
        unsigned chunk = 0;
        float timestep = 0;
        if (std::fscanf(list, "%u %f", &chunk, &timestep) != 2 ||
            !read_file(dir + "tokens_" + std::to_string(i) + ".bf16", tokens, L * 68) ||
            !read_file(dir + "hdmap_" + std::to_string(i) + ".bf16", hdmap, L * 64) ||
            !read_file(dir + "reference_" + std::to_string(i) + ".f64", reference, L * 64) ||
            !read_file(dir + "torch_bf16_" + std::to_string(i) + ".f32", torch, L * 64))
        {
            return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!runner.forward(tokens.data(), hdmap.data(), timestep, chunk, out.data(), error))
        {
            std::printf("cuda_dit_network: %s\n", error.message);
            return 1;
        }
        const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
        double e = 0, et = 0, r = 0;
        for (size_t j = 0; j < reference.size(); ++j)
        {
            e += (out[j] - reference[j]) * (out[j] - reference[j]);
            et += (torch[j] - reference[j]) * (torch[j] - reference[j]);
            r += reference[j] * reference[j];
        }
        const double ours = std::sqrt(e / r);
        const double theirs = std::sqrt(et / r);
        better = better && ours <= theirs;
        std::printf("pass %u chunk %u t %g: rel L2 to float64 ours %.3e, torch bf16 %.3e (%.1f ms)\n", i, chunk,
                    timestep, ours, theirs, ms);
    }
    std::fclose(list);
    return better ? 0 : 1;
}
