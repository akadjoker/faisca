// The real OmniDreams DiT at its real size (2 x 44 x 80 tokens a chunk, window 3 chunks, 28
// blocks) on random inputs: time per pass and device memory, block weights streamed or not.
// Usage: cuda_dit_bench <checkpoint .pt> [chunks] [--resident]
#include "faisca/omnidreams.h"
#include "faisca/omnidreams_cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace
{

uint16_t to_bf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: cuda_dit_bench <checkpoint .pt> [chunks] [--resident]\n");
        return 2;
    }
    const int chunks = argc >= 3 && argv[2][0] != '-' ? std::atoi(argv[2]) : 4;
    bool resident = false;
    for (int i = 2; i < argc; ++i)
    {
        resident = resident || std::string(argv[i]) == "--resident";
    }
    faisca::Error error;
    faisca::DitCheckpoint checkpoint;
    const auto open_start = std::chrono::steady_clock::now();
    if (!checkpoint.open(argv[1], error))
    {
        std::printf("cuda_dit_bench: %s\n", error.message);
        return 1;
    }
    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::printf("cuda_dit_bench: skipped, %s\n", error.message);
        return 77;
    }
    size_t free_before = 0, total = 0;
    device.memory_info(free_before, total, error);
    faisca::DitConfig config;
    config.stream_weights = !resident;
    faisca::DitCudaRunner runner;
    if (!runner.init(device, checkpoint.weights(), config, error))
    {
        std::printf("cuda_dit_bench: %s\n", error.message);
        return 1;
    }
    const double init_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - open_start).count();
    std::mt19937 random(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> text(size_t(faisca::DIT_TEXT_TOKENS) * faisca::DIT_TEXT_IN);
    for (uint16_t& v : text)
    {
        v = to_bf16(normal(random) * 0.1f);
    }
    if (!runner.set_text(text.data(), error))
    {
        std::printf("cuda_dit_bench: %s\n", error.message);
        return 1;
    }
    size_t free_after = 0;
    device.memory_info(free_after, total, error);
    std::printf("%s weights, init %.1f s, device memory used %.2f GB of %.2f\n", resident ? "resident" : "streamed",
                init_seconds, static_cast<double>(free_before - free_after) / 1e9, static_cast<double>(total) / 1e9);
    const size_t L = runner.chunk_tokens();
    std::vector<uint16_t> tokens(L * faisca::DIT_PATCH_IN), hdmap(L * faisca::DIT_HDMAP_IN);
    std::vector<float> out(L * faisca::DIT_PATCH_OUT);
    const float timesteps[] = {1000.0f, 500.0f, 128.0f};
    for (int chunk = 0; chunk < chunks; ++chunk)
    {
        double chunk_ms = 0;
        for (float timestep : timesteps)
        {
            for (uint16_t& v : tokens)
            {
                v = to_bf16(normal(random));
            }
            for (uint16_t& v : hdmap)
            {
                v = to_bf16(normal(random));
            }
            const auto start = std::chrono::steady_clock::now();
            if (!runner.forward(tokens.data(), hdmap.data(), timestep, chunk, out.data(), error))
            {
                std::printf("cuda_dit_bench: %s\n", error.message);
                return 1;
            }
            const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
            chunk_ms += ms;
            double sum = 0;
            for (float v : out)
            {
                sum += static_cast<double>(v) * v;
            }
            std::printf("chunk %d t %g: %.0f ms, output rms %.4f\n", chunk, timestep, ms, std::sqrt(sum / static_cast<double>(out.size())));
        }
        std::printf("chunk %d: %.2f s for 3 passes\n", chunk, chunk_ms / 1e3);
    }
    return 0;
}
