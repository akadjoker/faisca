#include "faisca/gguf.h"
#include "faisca/model.h"
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

// usage: vulkan_check <model.gguf> [--tokens N] [--rel 1e-5] [--device N] [--all-down]
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
        std::fprintf(stderr, "usage: vulkan_check <model.gguf> [--tokens N] [--rel 1e-5] [--device N] [--all-down]\n");
        return 2;
    }
    size_t n_tokens = 1;
    double limit = 1e-5;
    int device_index = -1;
    bool all_down = false;
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
        else if (std::strcmp(argv[i], "--all-down") == 0)
        {
            all_down = true;
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

    std::vector<std::string> names = {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                                      "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
                                      "blk.0.ffn_down.weight", "token_embd.weight"};
    if (all_down)
    {
        for (uint64_t l = 1; file.find_tensor(("blk." + std::to_string(l) + ".ffn_down.weight").c_str()) != nullptr; ++l)
        {
            names.push_back("blk." + std::to_string(l) + ".ffn_down.weight");
        }
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    int failures = 0;
    int checked = 0;
    double worst = 0.0;
    for (const std::string& name_string : names)
    {
        const char* name = name_string.c_str();
        const faisca::GgufTensor* tensor = file.find_tensor(name);
        if (tensor == nullptr || tensor->n_dims != 2)
        {
            std::printf("  %-28s missing\n", name);
            continue;
        }
        const size_t n_in = static_cast<size_t>(tensor->dims[0]);
        const size_t n_out = static_cast<size_t>(tensor->dims[1]);
        const faisca::GgmlType type = tensor->type;
        const bool q8 = type == faisca::GgmlType::Q8_0;
        const bool split = q8 || type == faisca::GgmlType::Q4_0 || type == faisca::GgmlType::Q5_0;
        const bool k_quant = type == faisca::GgmlType::Q4_K || type == faisca::GgmlType::Q6_K;
        if (!split && !k_quant && type != faisca::GgmlType::F32)
        {
            std::printf("  %-28s %s, not checked\n", name, faisca::ggml_type_info(type)->name);
            continue;
        }

        std::vector<float> x(n_tokens * n_in);
        for (float& v : x)
        {
            v = normal(rng);
        }
        std::vector<float> dense(n_out * n_in);
        faisca::Weight weight;
        weight.type = type;
        weight.data = tensor->data;
        weight.n_in = n_in;
        weight.n_out = n_out;
        for (size_t o = 0; o < n_out; ++o)
        {
            weight.row(o, dense.data() + o * n_in);
        }

        faisca::VulkanBuffer w_scales, w_quants, w_f32, x_buffer, y_buffer;
        std::vector<float> y(n_tokens * n_out);
        bool ok = device.create_host_buffer(x.size() * sizeof(float), x_buffer, error) &&
                  device.create_host_buffer(y.size() * sizeof(float), y_buffer, error) &&
                  device.upload(x_buffer, x.data(), x.size() * sizeof(float), error);
        if (ok && split)
        {
            std::vector<uint32_t> scales(faisca::q8_0_split_scale_words(n_in, n_out));
            std::vector<uint32_t> quants(q8 ? faisca::q8_0_split_quant_words(n_in, n_out)
                                        : type == faisca::GgmlType::Q4_0 ? faisca::q4_0_split_quant_words(n_in, n_out)
                                                                         : faisca::q5_0_split_quant_words(n_in, n_out));
            if (q8)
            {
                faisca::split_q8_0(tensor->data, n_in, n_out, scales.data(), quants.data());
            }
            else if (type == faisca::GgmlType::Q4_0)
            {
                faisca::split_q4_0(tensor->data, n_in, n_out, scales.data(), quants.data());
            }
            else
            {
                faisca::split_q5_0(tensor->data, n_in, n_out, scales.data(), quants.data());
            }
            ok = device.create_buffer(scales.size() * 4, w_scales, error) && device.create_buffer(quants.size() * 4, w_quants, error) &&
                 device.upload(w_scales, scales.data(), scales.size() * 4, error) &&
                 device.upload(w_quants, quants.data(), quants.size() * 4, error);
        }
        else if (ok && k_quant)
        {
            std::vector<uint32_t> blocks(type == faisca::GgmlType::Q4_K ? faisca::q4_k_device_words(n_in, n_out)
                                                                        : faisca::q6_k_device_words(n_in, n_out));
            if (type == faisca::GgmlType::Q4_K)
            {
                faisca::pack_q4_k_device(tensor->data, n_in, n_out, blocks.data());
            }
            else
            {
                faisca::pack_q6_k_device(tensor->data, n_in, n_out, blocks.data());
            }
            ok = device.create_buffer(blocks.size() * 4, w_quants, error) && device.upload(w_quants, blocks.data(), blocks.size() * 4, error);
        }
        else if (ok)
        {
            ok = device.create_buffer(dense.size() * 4, w_f32, error) && device.upload(w_f32, dense.data(), dense.size() * 4, error);
        }
        double best = 1e30;
        for (int rep = 0; ok && rep < 3; ++rep)
        {
            const double t0 = now();
            const uint32_t ni = static_cast<uint32_t>(n_in);
            const uint32_t no = static_cast<uint32_t>(n_out);
            const uint32_t nt = static_cast<uint32_t>(n_tokens);
            ok = device.begin(error);
            switch (type)
            {
                case faisca::GgmlType::Q8_0:
                    ok = ok && kernels.matvec_q8_0(w_scales, w_quants, x_buffer, y_buffer, ni, no, nt, ni, no, error);
                    break;
                case faisca::GgmlType::Q4_0:
                    ok = ok && kernels.matvec_q4_0(w_scales, w_quants, x_buffer, y_buffer, ni, no, nt, ni, no, error);
                    break;
                case faisca::GgmlType::Q5_0:
                    ok = ok && kernels.matvec_q5_0(w_scales, w_quants, x_buffer, y_buffer, ni, no, nt, ni, no, error);
                    break;
                case faisca::GgmlType::Q4_K:
                    ok = ok && kernels.matvec_q4_k(w_quants, x_buffer, y_buffer, ni, no, nt, ni, no, error);
                    break;
                case faisca::GgmlType::Q6_K:
                    ok = ok && kernels.matvec_q6_k(w_quants, x_buffer, y_buffer, ni, no, nt, ni, no, error);
                    break;
                default:
                    ok = ok && kernels.matvec_f32(w_f32, x_buffer, y_buffer, ni, no, nt, ni, no, error);
                    break;
            }
            ok = ok && device.submit(error);
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
        const double bytes = static_cast<double>(weight.row_bytes() * n_out);
        std::printf("  %-28s %-5s [%zu x %zu] rel L2 %.3e  %.3f ms, %.1f GB/s%s\n", name, faisca::ggml_type_info(type)->name,
                    n_in, n_out, rel, best * 1e3, bytes / best / 1e9, bad ? "  FAIL" : "");
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
