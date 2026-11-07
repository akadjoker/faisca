#include "faisca/cuda.h"
#include "faisca/cuda_ops.h"
#include "faisca/model.h"
#include "faisca/ops.h"
#if defined(FAISCA_CUDA_CHECK_GENERATED_PTX)
#include "kernels_ptx.h"
#endif

#include <cstdio>
#include <cstring>
#include <limits>
#include <cmath>
#include <random>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define FAISCA_TEST_CUDA_CALL __stdcall
#else
#include <dlfcn.h>
#define FAISCA_TEST_CUDA_CALL
#endif

namespace
{

bool check_values(faisca::CudaDevice& device, const faisca::CudaBuffer& buffer, const std::vector<float>& expected,
                  double limit, const char* name, faisca::Error& error)
{
    std::vector<float> actual(expected.size());
    if (!device.synchronize(error) || !device.download(buffer, actual.data(), actual.size() * sizeof(float), error))
    {
        return false;
    }
    double squared_error = 0;
    double squared_reference = 0;
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const double delta = static_cast<double>(actual[index]) - expected[index];
        squared_error += delta * delta;
        squared_reference += static_cast<double>(expected[index]) * expected[index];
    }
    const double relative = std::sqrt(squared_error / (squared_reference > 0 ? squared_reference : 1));
    std::printf("%s: rel L2 %.3e\n", name, relative);
    if (!std::isfinite(relative) || relative > limit)
    {
        error.set("CUDA %s failed its reference", name);
        return false;
    }
    return true;
}

bool check_ops(faisca::CudaDevice& device, faisca::Error& error)
{
    faisca::CudaKernels kernels;
    faisca::CudaStream stream;
    if (!kernels.init(device, error) || !device.create_stream(stream, error))
    {
        return false;
    }
    constexpr uint32_t width = 64;
    constexpr uint32_t count = 3;
    std::vector<float> weights(width * 7);
    std::vector<unsigned char> quants(weights.size() / 32 * 34);
    for (size_t index = 0; index < weights.size(); ++index)
    {
        const int value = static_cast<int>(index % 13) - 6;
        weights[index] = static_cast<float>(value) * 0.03125f;
        quants[(index / 32) * 34 + 1] = 0x28;
        quants[(index / 32) * 34 + 2 + index % 32] = static_cast<unsigned char>(value);
    }
    std::vector<unsigned char> packed_quants(faisca::cuda_q8_0_bytes(width, 7));
    faisca::cuda_pack_q8_0(quants.data(), width, 7, packed_quants.data());
    quants = packed_quants;
    uint32_t ids[count] = {6, 0, 3};
    std::vector<float> expected(width * count);
    for (uint32_t token = 0; token < count; ++token)
    {
        std::memcpy(expected.data() + token * width, weights.data() + ids[token] * width, width * sizeof(float));
    }
    faisca::CudaBuffer matrix;
    faisca::CudaBuffer quantized;
    faisca::CudaBuffer tokens;
    faisca::CudaBuffer output;
    if (!device.create_buffer(weights.size() * 4, matrix, error) ||
        !device.create_buffer(quants.size(), quantized, error) ||
        !device.create_buffer(sizeof(ids), tokens, error) || !device.create_buffer(expected.size() * 4, output, error) ||
        !device.upload(matrix, weights.data(), weights.size() * 4, error) ||
        !device.upload(quantized, quants.data(), quants.size(), error) || !device.upload(tokens, ids, sizeof(ids), error) ||
        !kernels.embed(matrix, tokens, output, width, 7, count, false, stream, error) ||
        !check_values(device, output, expected, 0, "embed_f32", error) ||
        !kernels.embed(quantized, tokens, output, width, 7, count, true, stream, error) ||
        !check_values(device, output, expected, 0, "embed_q8_0", error))
    {
        return false;
    }
    faisca::CudaBuffer normalized;
    std::vector<float> norm_reference(expected.size());
    for (uint32_t token = 0; token < count; ++token)
    {
        double sum = 0;
        for (uint32_t column = 0; column < width; ++column)
        {
            const double value = expected[token * width + column];
            sum += value * value;
        }
        const double scale = 1.0 / std::sqrt(sum / width + 1e-5f);
        for (uint32_t column = 0; column < width; ++column)
        {
            norm_reference[token * width + column] = static_cast<float>(expected[token * width + column] * scale * weights[column]);
        }
    }
    if (!device.create_buffer(expected.size() * 4, normalized, error) ||
        !kernels.rms_norm(output, matrix, normalized, width, count, 1e-5f, stream, error) ||
        !check_values(device, normalized, norm_reference, 1e-6, "rms_norm", error))
    {
        return false;
    }
    faisca::CudaBuffer factors;
    std::vector<float> factor_values(width / 2);
    for (size_t index = 0; index < factor_values.size(); ++index)
    {
        factor_values[index] = 1.0f + static_cast<float>(index) * 0.03125f;
    }
    if (!device.create_buffer(factor_values.size() * 4, factors, error) ||
        !device.upload(factors, factor_values.data(), factor_values.size() * 4, error))
    {
        return false;
    }
    std::vector<float> rotated;
    for (const bool has_factors : {false, true})
    {
        rotated = norm_reference;
        const float theta_scale = std::pow(10000.0f, -2.0f / width);
        for (uint32_t token = 0; token < count; ++token)
        {
            float theta = static_cast<float>(5 + token);
            for (uint32_t pair = 0; pair < width / 2; ++pair)
            {
                const float angle = has_factors ? theta / factor_values[pair] : theta;
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const size_t index = token * width + pair * 2;
                const float first = rotated[index];
                const float second = rotated[index + 1];
                rotated[index] = first * cosine - second * sine;
                rotated[index + 1] = first * sine + second * cosine;
                theta *= theta_scale;
            }
        }
        if (!device.upload(normalized, norm_reference.data(), norm_reference.size() * 4, error) ||
            !kernels.rope(normalized, factors, has_factors, 1, width, width, 5, 10000, count, stream, error) ||
            !check_values(device, normalized, rotated, 1e-6, has_factors ? "rope_factors" : "rope", error))
        {
            return false;
        }
    }
    constexpr uint32_t context = 11;
    std::vector<float> cached_keys(width * context * 2);
    std::vector<float> cached_values(cached_keys.size());
    for (size_t index = 0; index < cached_keys.size(); ++index)
    {
        cached_keys[index] = std::sin(static_cast<float>(index) * 0.01f) * 0.2f;
        cached_values[index] = std::cos(static_cast<float>(index) * 0.017f) * 0.3f;
    }
    faisca::CudaBuffer key_cache;
    faisca::CudaBuffer value_cache;
    if (!device.create_buffer(cached_keys.size() * 4, key_cache, error) ||
        !device.create_buffer(cached_values.size() * 4, value_cache, error) ||
        !device.upload(key_cache, cached_keys.data(), cached_keys.size() * 4, error) ||
        !device.upload(value_cache, cached_values.data(), cached_values.size() * 4, error) ||
        !device.download(normalized, rotated.data(), rotated.size() * 4, error))
    {
        return false;
    }
    for (uint32_t token = 0; token < count; ++token)
    {
        std::memcpy(cached_keys.data() + (context + 5 + token) * width, rotated.data() + token * width, width * 4);
        std::memcpy(cached_values.data() + (context + 5 + token) * width, expected.data() + token * width, width * 4);
    }
    if (!kernels.kv_store(normalized, output, key_cache, value_cache, 1, width, context, 5, 1, count, stream, error) ||
        !check_values(device, key_cache, cached_keys, 0, "kv_store_keys", error) ||
        !check_values(device, value_cache, cached_values, 0, "kv_store_values", error))
    {
        return false;
    }
    faisca::CudaBuffer query;
    faisca::CudaBuffer attended;
    faisca::CudaBuffer scores;
    std::vector<float> query_values(count * 2 * width);
    std::vector<float> attention_reference(query_values.size());
    for (uint32_t token = 0; token < count; ++token)
    {
        for (uint32_t head = 0; head < 2; ++head)
        {
            for (uint32_t column = 0; column < width; ++column)
            {
                query_values[(token * 2 + head) * width + column] =
                    rotated[token * width + column] * (head == 0 ? 1.0f : -0.75f);
            }
            std::vector<double> probabilities(6 + token);
            double maximum = -std::numeric_limits<double>::infinity();
            for (size_t past = 0; past < probabilities.size(); ++past)
            {
                double sum = 0;
                for (uint32_t column = 0; column < width; ++column)
                {
                    sum += static_cast<double>(query_values[(token * 2 + head) * width + column]) *
                           cached_keys[(context + past) * width + column];
                }
                probabilities[past] = sum / std::sqrt(static_cast<double>(width));
                maximum = std::fmax(maximum, probabilities[past]);
            }
            double denominator = 0;
            for (double& value : probabilities)
            {
                value = std::exp(value - maximum);
                denominator += value;
            }
            for (uint32_t column = 0; column < width; ++column)
            {
                double sum = 0;
                for (size_t past = 0; past < probabilities.size(); ++past)
                {
                    sum += probabilities[past] / denominator * cached_values[(context + past) * width + column];
                }
                attention_reference[(token * 2 + head) * width + column] = static_cast<float>(sum);
            }
        }
    }
    if (!device.create_buffer(query_values.size() * 4, query, error) ||
        !device.create_buffer(query_values.size() * 4, attended, error) ||
        !device.create_buffer(count * 2 * context * 4, scores, error) ||
        !device.upload(query, query_values.data(), query_values.size() * 4, error) ||
        !kernels.attention(query, key_cache, value_cache, scores, attended, 2, 1, width, context, 5, 1, count, stream, error) ||
        !check_values(device, attended, attention_reference, 1e-6, "attention_causal_gqa", error))
    {
        return false;
    }
    std::vector<float> ffn_reference(expected.size());
    for (size_t index = 0; index < expected.size(); ++index)
    {
        const double gate = expected[index];
        ffn_reference[index] = static_cast<float>((gate / (1.0 + std::exp(-gate))) * rotated[index]);
    }
    if (!kernels.swiglu(output, normalized, attended, static_cast<uint32_t>(expected.size()), stream, error) ||
        !check_values(device, attended, ffn_reference, 1e-6, "swiglu", error))
    {
        return false;
    }
    if (!device.download(attended, ffn_reference.data(), ffn_reference.size() * 4, error))
    {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index)
    {
        ffn_reference[index] += expected[index];
    }
    if (!kernels.add(attended, output, static_cast<uint32_t>(expected.size()), stream, error) ||
        !check_values(device, attended, ffn_reference, 0, "add", error))
    {
        return false;
    }
    return device.destroy_buffer(query, error) && device.destroy_buffer(attended, error) && device.destroy_buffer(scores, error) &&
           device.destroy_buffer(key_cache, error) && device.destroy_buffer(value_cache, error) &&
           device.destroy_buffer(factors, error) && device.destroy_buffer(normalized, error) && device.destroy_buffer(matrix, error) && device.destroy_buffer(quantized, error) &&
           device.destroy_buffer(tokens, error) && device.destroy_buffer(output, error) && device.destroy_stream(stream, error);
}

// quantize_q8_0 and gemm_q8_0 against the CPU's quantize_q8_0 and matmul_q8_0_packed on
// random weights and inputs: the quantized bytes and every product must be bit identical.
bool check_gemm(faisca::CudaDevice& device, faisca::Error& error)
{
    if (!faisca::q8_0_packed_supported())
    {
        std::printf("gemm_q8_0: skipped, this CPU build has no VNNI kernel to compare with\n");
        return true;
    }
    faisca::CudaKernels kernels;
    faisca::CudaStream stream;
    if (!kernels.init(device, error) || !device.create_stream(stream, error))
    {
        return false;
    }
    std::mt19937 random(11);
    std::normal_distribution<float> normal;
    std::uniform_int_distribution<int> byte(-127, 127);
    const uint32_t shapes[][2] = {{960, 320}, {2560, 96}, {64, 16}};
    for (const auto& shape : shapes)
    {
        const uint32_t n_in = shape[0];
        const uint32_t n_out = shape[1];
        const uint32_t blocks = n_in / 32;
        std::vector<unsigned char> rows(static_cast<size_t>(blocks) * n_out * faisca::Q8_0_BLOCK_BYTES);
        for (size_t block = 0; block < static_cast<size_t>(blocks) * n_out; ++block)
        {
            unsigned char* b = rows.data() + block * faisca::Q8_0_BLOCK_BYTES;
            const uint16_t scale = faisca::f32_to_f16(std::fabs(normal(random)) * 0.01f);
            std::memcpy(b, &scale, 2);
            for (size_t j = 0; j < 32; ++j)
            {
                b[2 + j] = static_cast<unsigned char>(static_cast<signed char>(byte(random)));
            }
        }
        std::vector<unsigned char> device_rows(faisca::cuda_q8_0_bytes(n_in, n_out));
        faisca::cuda_pack_q8_0(rows.data(), n_in, n_out, device_rows.data());
        std::vector<unsigned char> packed(faisca::q8_0_packed_bytes(n_in, n_out));
        faisca::pack_q8_0(rows.data(), n_in, n_out, packed.data());
        for (uint32_t tokens : {1u, 7u, 13u, 32u, 77u})
        {
            std::vector<float> input(static_cast<size_t>(n_in) * 80);
            for (float& value : input)
            {
                value = normal(random);
            }
            input[5] = 0; // a block with a zero next to large values
            std::vector<unsigned char> xq(static_cast<size_t>(blocks) * faisca::Q8_0_BLOCK_BYTES * tokens);
            std::vector<int32_t> sums(static_cast<size_t>(blocks) * tokens);
            std::vector<float> expected(static_cast<size_t>(n_out) * tokens);
            for (uint32_t token = 0; token < tokens; ++token)
            {
                unsigned char* x = xq.data() + static_cast<size_t>(token) * blocks * faisca::Q8_0_BLOCK_BYTES;
                faisca::quantize_q8_0(input.data() + static_cast<size_t>(token) * n_in, n_in, x);
                faisca::q8_0_block_sums(x, n_in, sums.data() + static_cast<size_t>(token) * blocks);
            }
            // the VNNI kernel the CPU runner uses (the scalar fallback rounds the product and the
            // sum apart, without a fused multiply add)
            faisca::matmul_q8_0_packed(packed.data(), xq.data(), blocks * faisca::Q8_0_BLOCK_BYTES, sums.data(), blocks,
                                       tokens, n_in, n_out, 0, n_out, expected.data(), n_out);
            faisca::CudaBuffer weight, device_input, quants, scales, output;
            std::vector<signed char> got_quants(static_cast<size_t>(n_in) * 80);
            std::vector<uint16_t> got_scales(static_cast<size_t>(blocks) * 80); // block major, 80 tokens apart
            std::vector<float> got(static_cast<size_t>(n_out) * tokens);
            if (!device.create_buffer(device_rows.size(), weight, error) ||
                !device.create_buffer(input.size() * 4, device_input, error) ||
                !device.create_buffer(got_quants.size(), quants, error) ||
                !device.create_buffer(got_scales.size() * 2, scales, error) ||
                !device.create_buffer(got.size() * 4, output, error) ||
                !device.upload(weight, device_rows.data(), device_rows.size(), error) ||
                !device.upload(device_input, input.data(), input.size() * 4, error) ||
                !kernels.quantize_q8_0(device_input, quants, scales, n_in, 80, 80, stream, error) ||
                !kernels.gemm_q8_0(weight, quants, scales, 80, output, n_in, n_out, tokens, stream, error) ||
                !device.synchronize(stream, error) ||
                !device.download(quants, got_quants.data(), got_quants.size(), error) ||
                !device.download(scales, got_scales.data(), got_scales.size() * 2, error) ||
                !device.download(output, got.data(), got.size() * 4, error))
            {
                return false;
            }
            size_t byte_mismatches = 0;
            for (uint32_t token = 0; token < tokens; ++token)
            {
                for (uint32_t block = 0; block < blocks; ++block)
                {
                    const unsigned char* x = xq.data() + (static_cast<size_t>(token) * blocks + block) * faisca::Q8_0_BLOCK_BYTES;
                    uint16_t scale = 0;
                    std::memcpy(&scale, x, 2);
                    byte_mismatches += scale != got_scales[static_cast<size_t>(block) * 80 + token];
                    byte_mismatches += std::memcmp(x + 2, got_quants.data() + static_cast<size_t>(token) * n_in + block * 32, 32) != 0;
                }
            }
            size_t mismatches = 0;
            double worst = 0;
            for (size_t index = 0; index < got.size(); ++index)
            {
                mismatches += std::memcmp(&got[index], &expected[index], 4) != 0;
                const double delta = std::fabs(static_cast<double>(got[index]) - expected[index]);
                const double scale = std::fabs(static_cast<double>(expected[index])) + 1e-30;
                worst = delta / scale > worst ? delta / scale : worst;
            }
            if (mismatches != 0)
            {
                std::printf("  worst relative difference %.3e, first: %.9g against %.9g\n", worst, got[0], expected[0]);
            }
            std::printf("gemm_q8_0 %u x %u, %u tokens: %zu quantized blocks and %zu of %zu products differ from the CPU\n",
                        n_in, n_out, tokens, byte_mismatches, mismatches, got.size());
            if (byte_mismatches != 0 || mismatches != 0 ||
                !device.destroy_buffer(weight, error) || !device.destroy_buffer(device_input, error) ||
                !device.destroy_buffer(quants, error) || !device.destroy_buffer(scales, error) ||
                !device.destroy_buffer(output, error))
            {
                if (error.empty())
                {
                    error.set("CUDA Q8_0 GEMM differs from the CPU");
                }
                return false;
            }
        }
    }
    return device.destroy_stream(stream, error);
}

bool check_matrices(faisca::CudaDevice& device, const char* path, faisca::Error& error)
{
    faisca::GgufFile file;
    faisca::CudaKernels kernels;
    faisca::CudaStream stream;
    if (!file.open(path, error) || !kernels.init(device, error) || !device.create_stream(stream, error))
    {
        return false;
    }
    const char* names[] = {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                          "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
                          "blk.0.ffn_down.weight", "token_embd.weight"};
    std::mt19937 random(7);
    std::normal_distribution<float> normal;
    for (const char* name : names)
    {
        const faisca::GgufTensor* tensor = file.find_tensor(name);
        if (tensor == nullptr || tensor->n_dims != 2)
        {
            error.set("missing test matrix %s", name);
            return false;
        }
        const uint32_t n_in = static_cast<uint32_t>(tensor->dims[0]);
        const uint32_t n_out = static_cast<uint32_t>(tensor->dims[1]);
        std::vector<float> weights(static_cast<size_t>(n_in) * n_out);
        faisca::Weight weight;
        weight.type = tensor->type;
        weight.data = tensor->data;
        weight.n_in = n_in;
        weight.n_out = n_out;
        for (uint32_t row = 0; row < n_out; ++row)
        {
            weight.row(row, weights.data() + static_cast<size_t>(row) * n_in);
        }
        faisca::CudaBuffer matrix;
        faisca::CudaBuffer quantized;
        if (!device.create_buffer(weights.size() * sizeof(float), matrix, error) ||
            !device.upload(matrix, weights.data(), weights.size() * sizeof(float), error))
        {
            return false;
        }
        if (tensor->type == faisca::GgmlType::Q8_0)
        {
            std::vector<unsigned char> packed(faisca::cuda_q8_0_bytes(n_in, n_out));
            faisca::cuda_pack_q8_0(tensor->data, n_in, n_out, packed.data());
            if (!device.create_buffer(packed.size(), quantized, error) ||
                !device.upload(quantized, packed.data(), packed.size(), error))
            {
                return false;
            }
        }
        for (const uint32_t tokens : {1u, 3u, 16u})
        {
            std::vector<float> input(static_cast<size_t>(tokens) * n_in);
            std::vector<float> output(static_cast<size_t>(tokens) * n_out);
            for (float& value : input)
            {
                value = normal(random);
            }
            faisca::CudaBuffer device_input;
            faisca::CudaBuffer device_output;
            if (!device.create_buffer(input.size() * sizeof(float), device_input, error) ||
                !device.create_buffer(output.size() * sizeof(float), device_output, error) ||
                !device.upload(device_input, input.data(), input.size() * sizeof(float), error) ||
                !kernels.matvec_f32(matrix, device_input, device_output, n_in, n_out, tokens, stream, error) ||
                !device.synchronize(stream, error) ||
                !device.download(device_output, output.data(), output.size() * sizeof(float), error))
            {
                return false;
            }
            double squared_error = 0;
            double squared_reference = 0;
            for (uint32_t token = 0; token < tokens; ++token)
            {
                for (uint32_t row = 0; row < n_out; ++row)
                {
                    double expected = 0;
                    for (uint32_t column = 0; column < n_in; ++column)
                    {
                        expected += static_cast<double>(weights[static_cast<size_t>(row) * n_in + column]) *
                                    input[static_cast<size_t>(token) * n_in + column];
                    }
                    const double delta = output[static_cast<size_t>(token) * n_out + row] - expected;
                    squared_error += delta * delta;
                    squared_reference += expected * expected;
                }
            }
            const double relative = std::sqrt(squared_error / (squared_reference > 0 ? squared_reference : 1));
            std::printf("matvec_f32 %s, %u inputs: rel L2 %.3e\n", name, tokens, relative);
            if (!std::isfinite(relative) || relative > 5e-7)
            {
                error.set("CUDA matrix %s failed double precision reference", name);
                return false;
            }
            if (quantized.address != 0)
            {
                if (!kernels.matvec_q8_0(quantized, device_input, device_output, n_in, n_out, tokens, stream, error) ||
                    !device.synchronize(stream, error) ||
                    !device.download(device_output, output.data(), output.size() * sizeof(float), error))
                {
                    return false;
                }
                double q8_error = 0;
                for (uint32_t token = 0; token < tokens; ++token)
                {
                    for (uint32_t row = 0; row < n_out; ++row)
                    {
                        double expected = 0;
                        for (uint32_t column = 0; column < n_in; ++column)
                        {
                            expected += static_cast<double>(weights[static_cast<size_t>(row) * n_in + column]) *
                                        input[static_cast<size_t>(token) * n_in + column];
                        }
                        const double delta = output[static_cast<size_t>(token) * n_out + row] - expected;
                        q8_error += delta * delta;
                    }
                }
                const double q8_relative = std::sqrt(q8_error / (squared_reference > 0 ? squared_reference : 1));
                std::printf("matvec_q8_0 %s, %u inputs: rel L2 %.3e\n", name, tokens, q8_relative);
                if (!std::isfinite(q8_relative) || q8_relative > 5e-7)
                {
                    error.set("CUDA Q8_0 matrix %s failed double precision reference", name);
                    return false;
                }
            }
            if (!device.destroy_buffer(device_input, error) || !device.destroy_buffer(device_output, error))
            {
                return false;
            }
        }
        if (!device.destroy_buffer(matrix, error) || !device.destroy_buffer(quantized, error))
        {
            return false;
        }
    }
    return device.destroy_stream(stream, error);
}

class CallerContext
{
public:
    ~CallerContext()
    {
        if (m_context != nullptr)
        {
            m_destroy(m_context);
        }
        if (m_library != nullptr)
        {
#if defined(_WIN32)
            FreeLibrary(reinterpret_cast<HMODULE>(m_library));
#else
            dlclose(m_library);
#endif
        }
    }

    bool init(faisca::Error& error)
    {
#if defined(_WIN32)
        m_library = LoadLibraryA("nvcuda.dll");
#else
        m_library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (m_library == nullptr)
        {
            error.set("test driver could not be loaded");
            return false;
        }
        const auto symbol = [&](const char* name) -> void*
        {
#if defined(_WIN32)
            return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(m_library), name));
#else
            return dlsym(m_library, name);
#endif
        };
        m_destroy = reinterpret_cast<Destroy>(symbol("cuCtxDestroy_v2"));
        m_current = reinterpret_cast<Current>(symbol("cuCtxGetCurrent"));
        const auto create = reinterpret_cast<Create>(symbol("cuCtxCreate_v2"));
        const auto get_device = reinterpret_cast<GetDevice>(symbol("cuDeviceGet"));
        int device = 0;
        if (m_destroy == nullptr || m_current == nullptr || create == nullptr || get_device == nullptr ||
            get_device(&device, 0) != 0 || create(&m_context, 0, device) != 0)
        {
            error.set("test caller context could not be created");
            return false;
        }
        return preserved(error);
    }

    bool preserved(faisca::Error& error) const
    {
        void* current = nullptr;
        if (m_current(&current) != 0 || current != m_context)
        {
            error.set("CudaDevice changed the caller's current CUDA context");
            return false;
        }
        return true;
    }

private:
    using Create = int (FAISCA_TEST_CUDA_CALL*)(void**, unsigned int, int);
    using Destroy = int (FAISCA_TEST_CUDA_CALL*)(void*);
    using Current = int (FAISCA_TEST_CUDA_CALL*)(void**);
    using GetDevice = int (FAISCA_TEST_CUDA_CALL*)(int*, int);
    void* m_library = nullptr;
    void* m_context = nullptr;
    Destroy m_destroy = nullptr;
    Current m_current = nullptr;
};

#if defined(FAISCA_CUDA_CHECK_GENERATED_PTX)
const char* const smoke_ptx = faisca::ptx_fill_words;
#else
const char smoke_ptx[] = R"ptx(
.version 7.8
.target sm_89
.address_size 64
.visible .entry fill_words(.param .u64 output, .param .u32 base)
{
    .reg .b64 address, offset;
    .reg .b32 index, value;
    ld.param.u64 address, [output];
    ld.param.u32 value, [base];
    mov.u32 index, %tid.x;
    add.u32 value, value, index;
    mul.wide.u32 offset, index, 4;
    add.u64 address, address, offset;
    st.global.u32 [address], value;
    ret;
}
)ptx";
#endif

}

int main(int argc, char** argv)
{
    bool require_device = false;
    const char* model_path = nullptr;
    for (int index = 1; index < argc; ++index)
    {
        if (std::strcmp(argv[index], "--require-device") == 0)
        {
            require_device = true;
        }
        else if (std::strcmp(argv[index], "--model") == 0 && index + 1 < argc)
        {
            model_path = argv[++index];
        }
        else
        {
            std::fprintf(stderr, "usage: cuda_check [--require-device] [--model model.gguf]\n");
            return 2;
        }
    }
    faisca::Error error;
    faisca::CudaDevice missing;
    if (missing.init(error, 0, "faisca-missing-cuda-driver") || missing.ready() || error.empty())
    {
        std::fprintf(stderr, "FAIL: missing driver was not handled\n");
        return 1;
    }
    missing.release();
    missing.release();

    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::fprintf(stderr, "%s: %s\n", require_device ? "FAIL" : "SKIP", error.c_str());
        return require_device ? 1 : 77;
    }
    std::printf("CUDA device: %s, compute %d.%d, driver API %d\n", device.device_name(),
                device.compute_major(), device.compute_minor(), device.driver_version());
    CallerContext caller;
    if (!caller.init(error) || !device.init(error) || !caller.preserved(error))
    {
        std::fprintf(stderr, "FAIL: caller context during init: %s\n", error.c_str());
        return 1;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!device.memory_info(free_bytes, total_bytes, error) || total_bytes == 0 || free_bytes > total_bytes)
    {
        std::fprintf(stderr, "FAIL: memory info: %s\n", error.c_str());
        return 1;
    }
    std::printf("CUDA memory: %zu free / %zu total bytes\n", free_bytes, total_bytes);

    constexpr size_t bytes = 4096;
    unsigned char input[bytes];
    unsigned char output[bytes] = {};
    for (size_t index = 0; index < bytes; ++index)
    {
        input[index] = static_cast<unsigned char>((index * 37 + 11) & 255);
    }
    faisca::CudaBuffer buffer;
    faisca::CudaHostBuffer pinned;
    faisca::CudaStream stream;
    faisca::CudaBuffer failed;
    if (device.create_buffer(0, failed, error) ||
        device.create_buffer(std::numeric_limits<size_t>::max(), failed, error) || failed.address != 0 ||
        !device.create_buffer(bytes, buffer, error) ||
        !device.create_host_buffer(bytes * 2, pinned, error) || !device.create_stream(stream, error) ||
        !device.upload(buffer, input, bytes, error) || !device.download(buffer, output, bytes, error) ||
        std::memcmp(input, output, bytes) != 0)
    {
        std::fprintf(stderr, "FAIL: synchronous roundtrip: %s\n", error.c_str());
        return 1;
    }
    std::memcpy(pinned.data, input, bytes);
    if (!device.upload_async(buffer, pinned, bytes, stream, error) ||
        !device.download_async(buffer, pinned, bytes, stream, error, 0, bytes) ||
        !device.synchronize(stream, error) ||
        std::memcmp(input, static_cast<const char*>(pinned.data) + bytes, bytes) != 0)
    {
        std::fprintf(stderr, "FAIL: asynchronous roundtrip: %s\n", error.c_str());
        return 1;
    }
    faisca::CudaStream consumer;
    faisca::CudaEvent uploaded;
    faisca::CudaEvent downloaded;
    bool complete = false;
    if (!device.create_stream(consumer, error) || !device.create_event(uploaded, error) ||
        !device.create_event(downloaded, error) || !device.upload_async(buffer, pinned, bytes, stream, error) ||
        !device.record_event(uploaded, stream, error) || !device.wait_event(consumer, uploaded, error) ||
        !device.download_async(buffer, pinned, bytes, consumer, error, 0, bytes) ||
        !device.record_event(downloaded, consumer, error) || !device.event_ready(downloaded, complete, error) ||
        !device.synchronize(downloaded, error) || !device.event_ready(downloaded, complete, error) || !complete ||
        std::memcmp(input, static_cast<const char*>(pinned.data) + bytes, bytes) != 0)
    {
        std::fprintf(stderr, "FAIL: cross-stream event ordering: %s\n", error.c_str());
        return 1;
    }
    if (device.compute_major() == 8 && device.compute_minor() == 9)
    {
        faisca::CudaModule module;
        faisca::CudaModule invalid;
        faisca::CudaKernel kernel;
        if (device.load_module("invalid PTX", invalid, error) ||
            !device.load_module(smoke_ptx, module, error) ||
            device.find_kernel(module, "missing_kernel", kernel, error) ||
            !device.find_kernel(module, "fill_words", kernel, error))
        {
            std::fprintf(stderr, "FAIL: PTX module loading: %s\n", error.c_str());
            return 1;
        }
        uint64_t address = buffer.address;
        uint32_t base = 1234;
        void* arguments[] = {&address, &base};
        faisca::CudaLaunch launch;
        launch.block_x = 32;
        uint32_t words[32] = {};
        if (!device.launch(kernel, stream, launch, arguments, error) ||
            !device.synchronize(stream, error) || !device.download(buffer, words, sizeof(words), error))
        {
            std::fprintf(stderr, "FAIL: PTX launch: %s\n", error.c_str());
            return 1;
        }
        for (uint32_t index = 0; index < 32; ++index)
        {
            if (words[index] != base + index)
            {
                std::fprintf(stderr, "FAIL: PTX output at %u\n", index);
                return 1;
            }
        }
        launch.block_x = 0;
        if (device.launch(kernel, stream, launch, arguments, error) ||
            !device.destroy_module(module, error))
        {
            std::fprintf(stderr, "FAIL: launch bounds or module destruction\n");
            return 1;
        }
        launch.block_x = 32;
        if (device.launch(kernel, stream, launch, arguments, error))
        {
            std::fprintf(stderr, "FAIL: kernel from destroyed module accepted\n");
            return 1;
        }
    }
    else
    {
        std::printf("PTX smoke test requires sm_89; transfer tests still ran\n");
    }
    if (!device.upload(buffer, input + 17, 113, error, 29) ||
        !device.download(buffer, output, 113, error, 29) || std::memcmp(input + 17, output, 113) != 0 ||
        !device.upload(buffer, nullptr, 0, error, bytes) ||
        device.upload(buffer, input, 1, error, bytes) ||
        device.download(buffer, output, 1, error, std::numeric_limits<size_t>::max()) ||
        device.upload(buffer, nullptr, 1, error) ||
        device.create_buffer(bytes, buffer, error))
    {
        std::fprintf(stderr, "FAIL: bounds or allocation reuse checks\n");
        return 1;
    }
    faisca::CudaDevice other;
    if (!other.init(error) || other.upload(buffer, input, 1, error) || !caller.preserved(error))
    {
        std::fprintf(stderr, "FAIL: foreign resource accepted\n");
        return 1;
    }
    const faisca::CudaBuffer stale = buffer;
    if (!device.destroy_buffer(buffer, error) || !device.destroy_buffer(buffer, error) ||
        device.upload(stale, input, 1, error) || !device.destroy_host_buffer(pinned, error) ||
        !device.destroy_stream(stream, error) || !device.destroy_stream(consumer, error) ||
        !device.destroy_event(uploaded, error) || !device.destroy_event(downloaded, error))
    {
        std::fprintf(stderr, "FAIL: explicit cleanup: %s\n", error.c_str());
        return 1;
    }
    if (!device.create_buffer(bytes, buffer, error) || !device.create_host_buffer(bytes, pinned, error) ||
        !device.create_stream(stream, error) || !device.create_event(uploaded, error))
    {
        std::fprintf(stderr, "FAIL: resources for release: %s\n", error.c_str());
        return 1;
    }
    std::memcpy(pinned.data, input, bytes);
    if (!device.upload_async(buffer, pinned, bytes, stream, error) ||
        !device.record_event(uploaded, stream, error))
    {
        std::fprintf(stderr, "FAIL: pending work for release: %s\n", error.c_str());
        return 1;
    }
    faisca::CudaModule retained;
    if (device.compute_major() == 8 && device.compute_minor() == 9 &&
        !device.load_module(smoke_ptx, retained, error))
    {
        std::fprintf(stderr, "FAIL: module for automatic cleanup: %s\n", error.c_str());
        return 1;
    }
    device.release();
    device.release();
    if (device.ready() || device.device_name()[0] != '\0' || device.upload(buffer, input, 1, error) ||
        !caller.preserved(error) || !device.init(error) || device.upload(buffer, input, 1, error) ||
        device.record_event(uploaded, stream, error) || !caller.preserved(error))
    {
        std::fprintf(stderr, "FAIL: release/reinit or stale resource checks: %s\n", error.c_str());
        return 1;
    }
    if (model_path != nullptr && !check_matrices(device, model_path, error))
    {
        std::fprintf(stderr, "FAIL: CUDA matrices: %s\n", error.c_str());
        return 1;
    }
    if (!check_gemm(device, error))
    {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    if (!check_ops(device, error))
    {
        std::fprintf(stderr, "FAIL: CUDA operations: %s\n", error.c_str());
        return 1;
    }
    device.release();
    other.release();
    if (!caller.preserved(error))
    {
        std::fprintf(stderr, "FAIL: caller context after release: %s\n", error.c_str());
        return 1;
    }
    std::printf("ok: driver loading, buffers, pinned transfers, bounds, streams, events, PTX and lifecycle\n");
    return 0;
}