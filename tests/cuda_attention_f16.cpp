// attention_f16 / attention_bf16 against a double softmax over the same rounded inputs; with
// sizes, timed, and with an output directory q, k, v and out written as raw halves
// (rows of heads * 128) for tests/compare_attention_f16.py.
// Usage: cuda_attention_f16 [n_q n_kv heads [reps [out dir]]]
#include "faisca/cuda.h"
#include "faisca/cuda_ops.h"
#include "faisca/ops.h"

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

float from_bf16(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

bool write_file(const std::string& path, const std::vector<uint16_t>& data)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
    {
        return false;
    }
    const bool ok = std::fwrite(data.data(), 2, data.size(), file) == data.size();
    return std::fclose(file) == 0 && ok;
}

bool run(faisca::CudaDevice& device, faisca::CudaKernels& kernels, const faisca::CudaStream& stream, uint32_t n_q,
         uint32_t n_kv, uint32_t heads, bool bf16, int reps, bool check, const char* dump, faisca::Error& error)
{
    const uint32_t width = heads * 128;
    std::mt19937 random(n_q * 131 + n_kv * 7 + heads + (bf16 ? 1 : 0));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(n_q) * width), k(static_cast<size_t>(n_kv) * width),
        v(static_cast<size_t>(n_kv) * width), out(static_cast<size_t>(n_q) * width);
    std::vector<float> qf(q.size()), kf(k.size()), vf(v.size());
    const auto fill = [&](std::vector<uint16_t>& h, std::vector<float>& f) {
        for (size_t i = 0; i < h.size(); ++i)
        {
            h[i] = bf16 ? to_bf16(normal(random)) : faisca::f32_to_f16(normal(random));
            f[i] = bf16 ? from_bf16(h[i]) : faisca::f16_to_f32(h[i]);
        }
    };
    fill(q, qf);
    fill(k, kf);
    fill(v, vf);
    const float scale = 1.0f / std::sqrt(128.0f);
    faisca::CudaBuffer dq, dk, dv, dout;
    if (!device.create_buffer(q.size() * 2, dq, error) || !device.create_buffer(k.size() * 2, dk, error) ||
        !device.create_buffer(v.size() * 2, dv, error) || !device.create_buffer(out.size() * 2, dout, error) ||
        !device.upload(dq, q.data(), q.size() * 2, error) || !device.upload(dk, k.data(), k.size() * 2, error) ||
        !device.upload(dv, v.data(), v.size() * 2, error) ||
        !kernels.attention_f16(dq, width, dk, dv, width, dout, width, n_q, n_kv, heads, scale, bf16, stream, error) ||
        !device.synchronize(error) || !device.download(dout, out.data(), out.size() * 2, error))
    {
        return false;
    }
    const char* name = bf16 ? "bf16" : "f16";
    bool ok = true;
    if (check)
    {
        double squared_error = 0;
        double squared_reference = 0;
        std::vector<double> p(n_kv);
        for (uint32_t h = 0; h < heads; ++h)
        {
            for (uint32_t r = 0; r < n_q; ++r)
            {
                double max = -INFINITY;
                for (uint32_t j = 0; j < n_kv; ++j)
                {
                    double s = 0;
                    for (uint32_t d = 0; d < 128; ++d)
                    {
                        s += static_cast<double>(qf[static_cast<size_t>(r) * width + h * 128 + d]) *
                             kf[static_cast<size_t>(j) * width + h * 128 + d];
                    }
                    p[j] = s * scale;
                    max = p[j] > max ? p[j] : max;
                }
                double sum = 0;
                for (uint32_t j = 0; j < n_kv; ++j)
                {
                    p[j] = std::exp(p[j] - max);
                    sum += p[j];
                }
                for (uint32_t d = 0; d < 128; ++d)
                {
                    double o = 0;
                    for (uint32_t j = 0; j < n_kv; ++j)
                    {
                        o += p[j] * vf[static_cast<size_t>(j) * width + h * 128 + d];
                    }
                    o /= sum;
                    const size_t index = static_cast<size_t>(r) * width + h * 128 + d;
                    const double actual = bf16 ? from_bf16(out[index]) : faisca::f16_to_f32(out[index]);
                    squared_error += (actual - o) * (actual - o);
                    squared_reference += o * o;
                }
            }
        }
        const double l2 = std::sqrt(squared_error / squared_reference);
        // P and O rounded to 11 (f16) or 8 (bf16) significant bits
        const double limit = bf16 ? 1.5e-2 : 2e-3;
        ok = std::isfinite(l2) && l2 <= limit;
        std::printf("%s q %u kv %u heads %u: rel L2 %.2e (limit %.0e)%s\n", name, n_q, n_kv, heads, l2, limit,
                    ok ? "" : "  FAIL");
    }
    if (dump != nullptr)
    {
        const std::string base = std::string(dump) + "/" + name + "_";
        if (!write_file(base + "q.bin", q) || !write_file(base + "k.bin", k) || !write_file(base + "v.bin", v) ||
            !write_file(base + "out.bin", out))
        {
            error.set("cannot write %s", base.c_str());
            return false;
        }
    }
    if (reps > 0)
    {
        // warm up first: the clock of an idle laptop GPU takes a few launches to rise
        for (int i = 0; i < reps; ++i)
        {
            if (!kernels.attention_f16(dq, width, dk, dv, width, dout, width, n_q, n_kv, heads, scale, bf16, stream,
                                       error))
            {
                return false;
            }
        }
        if (!device.synchronize(error))
        {
            return false;
        }
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i)
        {
            if (!kernels.attention_f16(dq, width, dk, dv, width, dout, width, n_q, n_kv, heads, scale, bf16, stream,
                                       error))
            {
                return false;
            }
        }
        if (!device.synchronize(error))
        {
            return false;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / reps;
        std::printf("%s q %u kv %u heads %u: %.3f ms, %.1f TFLOPS\n", name, n_q, n_kv, heads, seconds * 1e3,
                    4.0 * n_q * n_kv * 128 * heads / seconds * 1e-12);
    }
    device.destroy_buffer(dq, error);
    device.destroy_buffer(dk, error);
    device.destroy_buffer(dv, error);
    device.destroy_buffer(dout, error);
    return ok;
}

}

int main(int argc, char** argv)
{
    faisca::Error error;
    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::printf("cuda_attention_f16: skipped, %s\n", error.message);
        return 77;
    }
    faisca::CudaKernels kernels;
    faisca::CudaStream stream;
    if (!kernels.init(device, error) || !device.create_stream(stream, error))
    {
        std::printf("cuda_attention_f16: %s\n", error.message);
        return 1;
    }
    bool ok = true;
    if (argc >= 4)
    {
        const uint32_t n_q = static_cast<uint32_t>(std::atoi(argv[1]));
        const uint32_t n_kv = static_cast<uint32_t>(std::atoi(argv[2]));
        const uint32_t heads = static_cast<uint32_t>(std::atoi(argv[3]));
        const int reps = argc >= 5 ? std::atoi(argv[4]) : 10;
        const char* dump = argc >= 6 ? argv[5] : nullptr;
        for (int bf16 = 0; bf16 < 2 && ok; ++bf16)
        {
            ok = run(device, kernels, stream, n_q, n_kv, heads, bf16 != 0, reps, false, dump, error);
        }
    }
    else
    {
        const uint32_t shapes[][3] = {{128, 32, 1}, {1, 1, 1}, {77, 45, 2}, {200, 300, 3}, {257, 1000, 2}};
        for (const auto& shape : shapes)
        {
            for (int bf16 = 0; bf16 < 2; ++bf16)
            {
                ok = run(device, kernels, stream, shape[0], shape[1], shape[2], bf16 != 0, 0, true, nullptr, error) &&
                     ok;
            }
        }
    }
    if (!ok)
    {
        std::printf("cuda_attention_f16: %s\n", error.message[0] != 0 ? error.message : "reference mismatch");
        return 1;
    }
    return 0;
}
