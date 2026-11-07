// WanVaeCudaEncoder against tests/wanvae_reference.py (float64, seed + 4-frame chunks, two calls);
// with --time also 1 + 4 + 4 frames at 704 x 1280 timed, with the device memory it takes.
// Usage: cuda_wanvae <lightvaew2_1.pth> <reference dir> [--time]
#include "faisca/wanvae_cuda.h"

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
    return ok;
}

}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: cuda_wanvae <lightvaew2_1.pth> <reference dir> [--time]\n");
        return 2;
    }
    const std::string dir = std::string(argv[2]) + "/";
    unsigned h = 0, w = 0;
    FILE* shape = std::fopen((dir + "shape.txt").c_str(), "r");
    if (shape == nullptr || std::fscanf(shape, "%u %u", &h, &w) != 2)
    {
        std::printf("cannot read %sshape.txt\n", dir.c_str());
        return 1;
    }
    std::fclose(shape);
    faisca::Error error;
    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::printf("cuda_wanvae: skipped, %s\n", error.message);
        return 77;
    }
    bool ok = true;
    {
        faisca::WanVaeCudaEncoder encoder;
        if (!encoder.init(device, argv[1], h, w, error))
        {
            std::printf("cuda_wanvae: %s\n", error.message);
            return 1;
        }
        const size_t latent = size_t(16) * (h / 8) * (w / 8);
        const unsigned counts[2] = {5, 4};
        const unsigned expected[2] = {2, 1};
        for (unsigned call = 0; call < 2; ++call)
        {
            std::vector<float> frames, out(4 * latent);
            std::vector<double> reference;
            if (!read_file(dir + "frames_" + std::to_string(call) + ".f32", frames, size_t(counts[call]) * 3 * h * w) ||
                !read_file(dir + "latents_" + std::to_string(call) + ".f64", reference, expected[call] * latent))
            {
                return 1;
            }
            uint32_t produced = 0;
            if (!encoder.encode(frames.data(), counts[call], out.data(), produced, error))
            {
                std::printf("cuda_wanvae: %s\n", error.message);
                return 1;
            }
            double e = 0, r = 0, worst = 0;
            for (size_t i = 0; i < expected[call] * latent; ++i)
            {
                const double d = out[i] - reference[i];
                e += d * d;
                r += reference[i] * reference[i];
                worst = std::fabs(d) > worst ? std::fabs(d) : worst;
            }
            const double l2 = std::sqrt(e / r);
            ok = ok && produced == expected[call] && l2 < 1e-3;
            std::printf("call %u: %u frames -> %u latent frames (expected %u), rel L2 to float64 %.2e, worst %.2e\n",
                        call, counts[call], produced, expected[call], l2, worst);
        }
    }
    if (argc >= 4 && std::string(argv[3]) == "--time")
    {
        size_t free_before = 0, free_after = 0, total = 0;
        device.memory_info(free_before, total, error);
        faisca::WanVaeCudaEncoder full;
        if (!full.init(device, argv[1], 704, 1280, error))
        {
            std::printf("cuda_wanvae: %s\n", error.message);
            return 1;
        }
        device.memory_info(free_after, total, error);
        std::vector<float> frames(size_t(4) * 3 * 704 * 1280, 0.1f), out(size_t(2) * 16 * 88 * 160);
        const unsigned counts[3] = {1, 4, 4};
        for (unsigned c : counts)
        {
            uint32_t produced = 0;
            const auto start = std::chrono::steady_clock::now();
            if (!full.encode(frames.data(), c, out.data(), produced, error))
            {
                std::printf("cuda_wanvae: %s\n", error.message);
                return 1;
            }
            const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
            std::printf("704 x 1280: %u frames -> %u latent frames, %.0f ms\n", c, produced, ms);
        }
        std::printf("encoder device memory at 704 x 1280: %.2f GB\n",
                    static_cast<double>(free_before - free_after) / 1e9);
    }
    return ok ? 0 : 1;
}
