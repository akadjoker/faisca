#include "faisca/gguf.h"
#include "faisca/ops.h"
#include "faisca/vulkan.h"
#include "faisca/vulkan_ops.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// usage: vulkan_check <model.gguf> [--tokens N] [--rel 1e-5] [--device N]
//
// Runs the Vulkan kernels over the matrices of block 0 and the output of a model
// and compares them with a double precision product of the dequantized rows.

namespace
{

double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: vulkan_check <model.gguf> [--tokens N] [--rel 1e-5] [--device N]\n");
        return 2;
    }
    size_t n_tokens = 1;
    double limit = 1e-5;
    int device_index = -1;
    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
        {
            n_tokens = static_cast<size_t>(std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--rel") == 0 && i + 1 < argc)
        {
            limit = std::atof(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
        {
            device_index = std::atoi(argv[++i]);
        }
    }

    faisca::Error error;
    faisca::GgufFile file;
    if (!file.open(argv[1], error))
    {
        std::fprintf(stderr, "vulkan_check: %s\n", error.c_str());
        return 1;
    }
    faisca::VulkanDevice device;
    faisca::VulkanKernels kernels;
    if (!device.init(error, device_index) || !kernels.init(device, error))
    {
        std::fprintf(stderr, "vulkan_check: %s\n", error.c_str());
        return 1;
    }
    std::printf("device: %s (%s memory)\n", device.device_name(), device.unified_memory() ? "unified" : "discrete");

    const char* names[] = {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
                           "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "token_embd.weight"};
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    int failures = 0;
    int checked = 0;
    double worst = 0.0;
    for (const char* name : names)
    {
        const faisca::GgufTensor* tensor = file.find_tensor(name);
        if (tensor == nullptr || tensor->n_dims != 2)
        {
            std::printf("  %-28s missing\n", name);
            continue;
        }
        const size_t n_in = static_cast<size_t>(tensor->dims[0]);
        const size_t n_out = static_cast<size_t>(tensor->dims[1]);
        const bool q8 = tensor->type == faisca::GgmlType::Q8_0;
        if (!q8 && tensor->type != faisca::GgmlType::F32)
        {
            std::printf("  %-28s %s, not checked\n", name, faisca::ggml_type_info(tensor->type)->name);
            continue;
        }

        std::vector<float> x(n_tokens * n_in);
        for (float& v : x)
        {
            v = normal(rng);
        }
        std::vector<float> dense(n_out * n_in);
        if (q8)
        {
            for (size_t o = 0; o < n_out; ++o)
            {
                faisca::dequantize_q8_0(tensor->data + o * (n_in / 32) * 34, n_in, dense.data() + o * n_in);
            }
        }
        else
        {
            std::memcpy(dense.data(), tensor->data, dense.size() * sizeof(float));
        }

        faisca::VulkanBuffer w_scales, w_quants, w_f32, x_buffer, y_buffer;
        std::vector<float> y(n_tokens * n_out);
        bool ok = device.create_host_buffer(x.size() * sizeof(float), x_buffer, error) &&
                  device.create_host_buffer(y.size() * sizeof(float), y_buffer, error) &&
                  device.upload(x_buffer, x.data(), x.size() * sizeof(float), error);
        if (ok && q8)
        {
            std::vector<uint32_t> scales(faisca::q8_0_split_scale_words(n_in, n_out));
            std::vector<uint32_t> quants(faisca::q8_0_split_quant_words(n_in, n_out));
            faisca::split_q8_0(tensor->data, n_in, n_out, scales.data(), quants.data());
            ok = device.create_buffer(scales.size() * 4, w_scales, error) && device.create_buffer(quants.size() * 4, w_quants, error) &&
                 device.upload(w_scales, scales.data(), scales.size() * 4, error) &&
                 device.upload(w_quants, quants.data(), quants.size() * 4, error);
        }
        else if (ok)
        {
            ok = device.create_buffer(dense.size() * 4, w_f32, error) && device.upload(w_f32, dense.data(), dense.size() * 4, error);
        }
        double best = 1e30;
        for (int rep = 0; ok && rep < 3; ++rep)
        {
            const double t0 = now();
            ok = device.begin(error) &&
                 (q8 ? kernels.matvec_q8_0(w_scales, w_quants, x_buffer, y_buffer, static_cast<uint32_t>(n_in),
                                           static_cast<uint32_t>(n_out), static_cast<uint32_t>(n_tokens),
                                           static_cast<uint32_t>(n_in), static_cast<uint32_t>(n_out), error)
                     : kernels.matvec_f32(w_f32, x_buffer, y_buffer, static_cast<uint32_t>(n_in), static_cast<uint32_t>(n_out),
                                          static_cast<uint32_t>(n_tokens), static_cast<uint32_t>(n_in),
                                          static_cast<uint32_t>(n_out), error)) &&
                 device.submit(error);
            const double dt = now() - t0;
            best = dt < best ? dt : best;
        }
        ok = ok && device.download(y_buffer, y.data(), y.size() * sizeof(float), error);
        if (!ok)
        {
            std::printf("  %-28s %s\n", name, error.c_str());
            ++failures;
            continue;
        }
        double err = 0.0;
        double norm = 0.0;
        for (size_t t = 0; t < n_tokens; ++t)
        {
            for (size_t o = 0; o < n_out; ++o)
            {
                double expected = 0.0;
                for (size_t i = 0; i < n_in; ++i)
                {
                    expected += static_cast<double>(dense[o * n_in + i]) * x[t * n_in + i];
                }
                const double diff = y[t * n_out + o] - expected;
                err += diff * diff;
                norm += expected * expected;
            }
        }
        const double rel = std::sqrt(err / norm);
        ++checked;
        worst = rel > worst ? rel : worst;
        const bool bad = rel > limit;
        failures += bad ? 1 : 0;
        const double bytes = q8 ? static_cast<double>(n_out * (n_in / 32) * 34) : static_cast<double>(dense.size() * 4);
        std::printf("  %-28s %-5s [%zu x %zu] rel L2 %.3e  %.3f ms, %.1f GB/s%s\n", name, q8 ? "Q8_0" : "F32", n_in, n_out, rel,
                    best * 1e3, bytes / best / 1e9, bad ? "  FAIL" : "");
        device.destroy_buffer(w_scales);
        device.destroy_buffer(w_quants);
        device.destroy_buffer(w_f32);
        device.destroy_buffer(x_buffer);
        device.destroy_buffer(y_buffer);
    }
    std::printf("%s %s: %d matrices checked with %zu inputs, worst rel L2 %.3e, %d failures\n", failures == 0 ? "ok  " : "FAIL",
                argv[1], checked, n_tokens, worst, failures);
    return failures == 0 ? 0 : 1;
}
