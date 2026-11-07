// TaehvCudaDecoder against tests/taehv_reference.py (float64, whole chunks), chunk by chunk;
// with --time, also one chunk of 2 latent frames at the real size (88 x 160 -> 704 x 1280) timed.
// Usage: cuda_taehv <lighttaew2_1.pth> <reference dir> [--time]
#include "faisca/taehv_cuda.h"

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
        std::printf("usage: cuda_taehv <lighttaew2_1.pth> <reference dir> [--time]\n");
        return 2;
    }
    const std::string dir = std::string(argv[2]) + "/";
    unsigned h = 0, w = 0, chunks = 0;
    FILE* shape = std::fopen((dir + "shape.txt").c_str(), "r");
    if (shape == nullptr || std::fscanf(shape, "%u %u %u", &h, &w, &chunks) != 3)
    {
        std::printf("cannot read %sshape.txt\n", dir.c_str());
        return 1;
    }
    std::fclose(shape);
    faisca::Error error;
    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::printf("cuda_taehv: skipped, %s\n", error.message);
        return 77;
    }
    faisca::TaehvCudaDecoder decoder;
    if (!decoder.init(device, argv[1], h, w, error))
    {
        std::printf("cuda_taehv: %s\n", error.message);
        return 1;
    }
    const size_t frame = size_t(3) * h * 8 * w * 8;
    bool ok = true;
    for (unsigned c = 0; c < chunks; ++c)
    {
        std::vector<float> latents;
        std::vector<double> reference;
        if (!read_file(dir + "latents_" + std::to_string(c) + ".f32", latents, size_t(2) * 16 * h * w))
        {
            return 1;
        }
        const size_t expected = c == 0 ? 5 : 8;
        if (!read_file(dir + "frames_" + std::to_string(c) + ".f64", reference, expected * frame))
        {
            return 1;
        }
        std::vector<float> out(8 * frame);
        uint32_t produced = 0;
        if (!decoder.decode(latents.data(), 2, out.data(), produced, error))
        {
            std::printf("cuda_taehv: %s\n", error.message);
            return 1;
        }
        double e = 0, r = 0, worst = 0;
        for (size_t i = 0; i < expected * frame; ++i)
        {
            const double d = out[i] - reference[i];
            e += d * d;
            r += reference[i] * reference[i];
            worst = std::fabs(d) > worst ? std::fabs(d) : worst;
        }
        const double l2 = std::sqrt(e / r);
        ok = ok && produced == expected && l2 < 1e-5;
        std::printf("chunk %u: %u frames (expected %zu), rel L2 to float64 %.2e, worst %.2e (of 255: %.4f)\n", c,
                    produced, expected, l2, worst, worst * 255);
    }
    if (argc >= 4 && std::string(argv[3]) == "--time")
    {
        faisca::TaehvCudaDecoder full;
        if (!full.init(device, argv[1], 88, 160, error))
        {
            std::printf("cuda_taehv: %s\n", error.message);
            return 1;
        }
        size_t free_bytes = 0, total = 0;
        device.memory_info(free_bytes, total, error);
        std::vector<float> latents(size_t(2) * 16 * 88 * 160, 0.1f), out(size_t(8) * 3 * 704 * 1280);
        for (int i = 0; i < 3; ++i)
        {
            uint32_t produced = 0;
            const auto start = std::chrono::steady_clock::now();
            if (!full.decode(latents.data(), 2, out.data(), produced, error))
            {
                std::printf("cuda_taehv: %s\n", error.message);
                return 1;
            }
            const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
            std::printf("704 x 1280, 2 latent frames -> %u frames: %.0f ms\n", produced, ms);
        }
        std::printf("device memory free with both decoders: %.2f GB\n", static_cast<double>(free_bytes) / 1e9);
    }
    return ok ? 0 : 1;
}
