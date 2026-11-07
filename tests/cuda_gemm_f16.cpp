// gemm_f16 / gemm_bf16 against a double reference over the same rounded inputs, then timed.
// Usage: cuda_gemm_f16 [m n k [reps]]; with no sizes it runs the check shapes only.
#include "faisca/cuda.h"
#include "faisca/cuda_ops.h"
#include "faisca/ops.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

bool run(faisca::CudaDevice& device, faisca::CudaKernels& kernels, const faisca::CudaStream& stream, uint32_t m,
         uint32_t n, uint32_t k, bool bf16, int reps, bool check, faisca::Error& error)
{
    std::mt19937 random(m * 131 + n * 7 + k + (bf16 ? 1 : 0));
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<uint16_t> x(static_cast<size_t>(m) * k), w(static_cast<size_t>(n) * k);
    std::vector<float> xf(x.size()), wf(w.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = bf16 ? to_bf16(uniform(random)) : faisca::f32_to_f16(uniform(random));
        xf[i] = bf16 ? from_bf16(x[i]) : faisca::f16_to_f32(x[i]);
    }
    for (size_t i = 0; i < w.size(); ++i)
    {
        w[i] = bf16 ? to_bf16(uniform(random)) : faisca::f32_to_f16(uniform(random));
        wf[i] = bf16 ? from_bf16(w[i]) : faisca::f16_to_f32(w[i]);
    }
    faisca::CudaBuffer dx, dw, dout;
    if (!device.create_buffer(x.size() * 2, dx, error) || !device.create_buffer(w.size() * 2, dw, error) ||
        !device.create_buffer(static_cast<size_t>(m) * n * 4, dout, error) ||
        !device.upload(dx, x.data(), x.size() * 2, error) || !device.upload(dw, w.data(), w.size() * 2, error) ||
        !kernels.gemm_f16(dx, dw, dout, m, n, k, bf16, stream, error) || !device.synchronize(error))
    {
        return false;
    }
    const char* name = bf16 ? "bf16" : "f16";
    bool ok = true;
    if (check)
    {
        std::vector<float> out(static_cast<size_t>(m) * n);
        if (!device.download(dout, out.data(), out.size() * 4, error))
        {
            return false;
        }
        double squared_error = 0;
        double squared_reference = 0;
        double worst = 0;
        for (uint32_t r = 0; r < m; ++r)
        {
            for (uint32_t c = 0; c < n; ++c)
            {
                double sum = 0;
                double magnitude = 0;
                for (uint32_t i = 0; i < k; ++i)
                {
                    const double p = static_cast<double>(xf[static_cast<size_t>(r) * k + i]) * wf[static_cast<size_t>(c) * k + i];
                    sum += p;
                    magnitude += std::fabs(p);
                }
                const double delta = out[static_cast<size_t>(r) * n + c] - sum;
                squared_error += delta * delta;
                squared_reference += sum * sum;
                const double relative = std::fabs(delta) / (magnitude > 0 ? magnitude : 1);
                worst = relative > worst ? relative : worst;
            }
        }
        const double l2 = std::sqrt(squared_error / squared_reference);
        // f32 accumulation over k terms: |error| <= k * 2^-24 * sum |products|
        ok = std::isfinite(l2) && worst <= k * 0x1p-24;
        std::printf("%s %u x %u x %u: rel L2 %.2e, worst |error| / sum |products| %.2e%s\n", name, m, n, k, l2, worst,
                    ok ? "" : "  FAIL");
    }
    if (reps > 0)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i)
        {
            if (!kernels.gemm_f16(dx, dw, dout, m, n, k, bf16, stream, error))
            {
                return false;
            }
        }
        if (!device.synchronize(error))
        {
            return false;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / reps;
        std::printf("%s %u x %u x %u: %.3f ms, %.1f TFLOPS\n", name, m, n, k, seconds * 1e3,
                    2.0 * m * n * k / seconds * 1e-12);
    }
    device.destroy_buffer(dx, error);
    device.destroy_buffer(dw, error);
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
        std::printf("cuda_gemm_f16: skipped, %s\n", error.message);
        return 77;
    }
    faisca::CudaKernels kernels;
    faisca::CudaStream stream;
    if (!kernels.init(device, error) || !device.create_stream(stream, error))
    {
        std::printf("cuda_gemm_f16: %s\n", error.message);
        return 1;
    }
    bool ok = true;
    if (argc >= 4)
    {
        const uint32_t m = static_cast<uint32_t>(std::atoi(argv[1]));
        const uint32_t n = static_cast<uint32_t>(std::atoi(argv[2]));
        const uint32_t k = static_cast<uint32_t>(std::atoi(argv[3]));
        const int reps = argc >= 5 ? std::atoi(argv[4]) : 20;
        for (int bf16 = 0; bf16 < 2 && ok; ++bf16)
        {
            ok = run(device, kernels, stream, m, n, k, bf16 != 0, 2, false, error) &&
                 run(device, kernels, stream, m, n, k, bf16 != 0, reps, false, error);
        }
    }
    else
    {
        const uint32_t shapes[][3] = {{128, 128, 32}, {1, 8, 32}, {77, 200, 96}, {300, 136, 1024}, {257, 520, 2048}};
        for (const auto& shape : shapes)
        {
            for (int bf16 = 0; bf16 < 2; ++bf16)
            {
                ok = run(device, kernels, stream, shape[0], shape[1], shape[2], bf16 != 0, 0, true, error) && ok;
            }
        }
    }
    if (!ok)
    {
        std::printf("cuda_gemm_f16: %s\n", error.message[0] != 0 ? error.message : "reference mismatch");
        return 1;
    }
    return 0;
}
