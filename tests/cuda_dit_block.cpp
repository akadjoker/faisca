// One OmniDreams DiT block on the CUDA kernels (bf16 matrices, f32 residual) against
// tests/dit_block_reference.py's float64 run, next to PyTorch's bf16 run of the same block.
// Usage: cuda_dit_block <reference dir> [reps]
#include "faisca/cuda.h"
#include "faisca/cuda_ops.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{

constexpr uint32_t D = 2048;
constexpr uint32_t HEADS = 16;
constexpr uint32_t CONTEXT = 512;
constexpr uint32_t CONTEXT_DIM = 1024;
constexpr uint32_t MLP = 8192;

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

faisca::CudaBuffer view(const faisca::CudaBuffer& buffer, size_t offset)
{
    faisca::CudaBuffer out = buffer;
    out.address += offset;
    out.bytes -= offset;
    return out;
}

struct Errors
{
    double total;
    double update;
};

template <typename T>
Errors compare(const std::vector<T>& actual, const std::vector<double>& reference, const std::vector<float>& x)
{
    double e = 0, r = 0, eu = 0, ru = 0;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        const double d = static_cast<double>(actual[i]) - reference[i];
        const double u = reference[i] - x[i];
        e += d * d;
        r += reference[i] * reference[i];
        eu += d * d;
        ru += u * u;
    }
    return {std::sqrt(e / r), std::sqrt(eu / ru)};
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: cuda_dit_block <reference dir> [reps]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const int reps = argc >= 3 ? std::atoi(argv[2]) : 5;
    uint32_t t = 0, h = 0, w = 0;
    {
        FILE* file = std::fopen((dir + "/shape.txt").c_str(), "r");
        if (file == nullptr || std::fscanf(file, "%u %u %u", &t, &h, &w) != 3)
        {
            std::printf("cannot read %s/shape.txt\n", dir.c_str());
            return 1;
        }
        std::fclose(file);
    }
    const uint32_t L = t * h * w;

    std::vector<float> x, modulation, angles, torch_bf16;
    std::vector<double> reference;
    std::vector<uint16_t> context, qkv_w, q_norm, k_norm, o_w, cq_w, ckv_w, cq_norm, ck_norm, co_w, mlp1, mlp2, part;
    bool ok = read_file(dir + "/x.f32", x, size_t(L) * D) &&
              read_file(dir + "/modulation.f32", modulation, size_t(9) * D) &&
              read_file(dir + "/angles.f32", angles, size_t(L) * 64) &&
              read_file(dir + "/torch_bf16.f32", torch_bf16, size_t(L) * D) &&
              read_file(dir + "/reference.f64", reference, size_t(L) * D) &&
              read_file(dir + "/context.bf16", context, size_t(CONTEXT) * CONTEXT_DIM) &&
              read_file(dir + "/self_q_norm.bf16", q_norm, 128) && read_file(dir + "/self_k_norm.bf16", k_norm, 128) &&
              read_file(dir + "/cross_q_norm.bf16", cq_norm, 128) &&
              read_file(dir + "/cross_k_norm.bf16", ck_norm, 128) && read_file(dir + "/self_o.bf16", o_w, size_t(D) * D) &&
              read_file(dir + "/cross_q.bf16", cq_w, size_t(D) * D) && read_file(dir + "/cross_o.bf16", co_w, size_t(D) * D) &&
              read_file(dir + "/mlp1.bf16", mlp1, size_t(MLP) * D) && read_file(dir + "/mlp2.bf16", mlp2, size_t(D) * MLP);
    for (const char* name : {"self_q", "self_k", "self_v"})
    {
        ok = ok && read_file(dir + "/" + name + ".bf16", part, size_t(D) * D);
        qkv_w.insert(qkv_w.end(), part.begin(), part.end());
    }
    for (const char* name : {"cross_k", "cross_v"})
    {
        ok = ok && read_file(dir + "/" + name + ".bf16", part, size_t(D) * CONTEXT_DIM);
        ckv_w.insert(ckv_w.end(), part.begin(), part.end());
    }
    if (!ok)
    {
        return 1;
    }

    faisca::Error error;
    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::printf("cuda_dit_block: skipped, %s\n", error.message);
        return 77;
    }
    faisca::CudaKernels kernels;
    faisca::CudaStream stream;
    faisca::CudaBuffer d_x, d_x0, d_mod, d_angles, d_context, d_qkv_w, d_qn, d_kn, d_o_w, d_cq_w, d_ckv_w, d_cqn, d_ckn,
        d_co_w, d_mlp1, d_mlp2, d_h, d_qkv, d_attn, d_ckv, d_hidden;
    const auto upload = [&](faisca::CudaBuffer& buffer, const void* data, size_t bytes) {
        return device.create_buffer(bytes, buffer, error) && device.upload(buffer, data, bytes, error);
    };
    if (!kernels.init(device, error) || !device.create_stream(stream, error) ||
        !upload(d_x0, x.data(), x.size() * 4) || !device.create_buffer(x.size() * 4, d_x, error) ||
        !upload(d_mod, modulation.data(), modulation.size() * 4) ||
        !upload(d_angles, angles.data(), angles.size() * 4) || !upload(d_context, context.data(), context.size() * 2) ||
        !upload(d_qkv_w, qkv_w.data(), qkv_w.size() * 2) || !upload(d_qn, q_norm.data(), 256) ||
        !upload(d_kn, k_norm.data(), 256) || !upload(d_o_w, o_w.data(), o_w.size() * 2) ||
        !upload(d_cq_w, cq_w.data(), cq_w.size() * 2) || !upload(d_ckv_w, ckv_w.data(), ckv_w.size() * 2) ||
        !upload(d_cqn, cq_norm.data(), 256) || !upload(d_ckn, ck_norm.data(), 256) ||
        !upload(d_co_w, co_w.data(), co_w.size() * 2) || !upload(d_mlp1, mlp1.data(), mlp1.size() * 2) ||
        !upload(d_mlp2, mlp2.data(), mlp2.size() * 2) || !device.create_buffer(size_t(L) * D * 2, d_h, error) ||
        !device.create_buffer(size_t(L) * 3 * D * 2, d_qkv, error) ||
        !device.create_buffer(size_t(L) * D * 2, d_attn, error) ||
        !device.create_buffer(size_t(CONTEXT) * 2 * D * 2, d_ckv, error) ||
        !device.create_buffer(size_t(L) * MLP * 2, d_hidden, error))
    {
        std::printf("cuda_dit_block: %s\n", error.message);
        return 1;
    }
    using Out = faisca::CudaKernels::GemmOutput;
    const float scale = 1.0f / std::sqrt(128.0f);
    const auto mod = [](int i) { return static_cast<uint64_t>(i) * D * 4; };
    // the text keys and values: computed once per scene (Block.initialize_cache)
    const auto context_kv = [&]() {
        return kernels.gemm_f16(d_context, d_ckv_w, d_ckv, CONTEXT, 2 * D, CONTEXT_DIM, true, stream, error, Out::Half) &&
               kernels.dit_qk_norm_rope(d_ckv, 2 * D, 0, d_ckn, nullptr, CONTEXT, HEADS, true, stream, error);
    };
    const auto block = [&]() {
        return kernels.dit_modulate_norm(d_x, d_mod, mod(0), mod(1), d_h, L, D, true, stream, error) &&
               kernels.gemm_f16(d_h, d_qkv_w, d_qkv, L, 3 * D, D, true, stream, error, Out::Half) &&
               kernels.dit_qk_norm_rope(d_qkv, 3 * D, 0, d_qn, &d_angles, L, HEADS, true, stream, error) &&
               kernels.dit_qk_norm_rope(d_qkv, 3 * D, D, d_kn, &d_angles, L, HEADS, true, stream, error) &&
               kernels.attention_f16(d_qkv, 3 * D, view(d_qkv, D * 2), view(d_qkv, 2 * D * 2), 3 * D, d_attn, D, L, L,
                                     HEADS, scale, true, stream, error) &&
               kernels.gemm_f16(d_attn, d_o_w, d_x, L, D, D, true, stream, error, Out::Residual, &d_mod, mod(2)) &&
               kernels.dit_modulate_norm(d_x, d_mod, mod(3), mod(4), d_h, L, D, true, stream, error) &&
               kernels.gemm_f16(d_h, d_cq_w, d_qkv, L, D, D, true, stream, error, Out::Half) &&
               kernels.dit_qk_norm_rope(d_qkv, D, 0, d_cqn, nullptr, L, HEADS, true, stream, error) &&
               kernels.attention_f16(d_qkv, D, d_ckv, view(d_ckv, D * 2), 2 * D, d_attn, D, L, CONTEXT, HEADS, scale,
                                     true, stream, error) &&
               kernels.gemm_f16(d_attn, d_co_w, d_x, L, D, D, true, stream, error, Out::Residual, &d_mod, mod(5)) &&
               kernels.dit_modulate_norm(d_x, d_mod, mod(6), mod(7), d_h, L, D, true, stream, error) &&
               kernels.gemm_f16(d_h, d_mlp1, d_hidden, L, MLP, D, true, stream, error, Out::HalfGelu) &&
               kernels.gemm_f16(d_hidden, d_mlp2, d_x, L, D, MLP, true, stream, error, Out::Residual, &d_mod, mod(8));
    };
    std::vector<float> out(size_t(L) * D);
    if (!device.upload(d_x, x.data(), x.size() * 4, error) || !context_kv() || !block() ||
        !device.synchronize(error) || !device.download(d_x, out.data(), out.size() * 4, error))
    {
        std::printf("cuda_dit_block: %s\n", error.message);
        return 1;
    }
    const Errors ours = compare(out, reference, x);
    const Errors torch = compare(torch_bf16, reference, x);
    std::printf("block %u tokens (%u x %u x %u), rel L2 to float64: ours %.3e (update %.3e), torch bf16 %.3e "
                "(update %.3e)\n",
                L, t, h, w, ours.total, ours.update, torch.total, torch.update);
    if (reps > 0)
    {
        for (int i = 0; i < reps; ++i)
        {
            if (!block())
            {
                std::printf("cuda_dit_block: %s\n", error.message);
                return 1;
            }
        }
        if (!device.synchronize(error))
        {
            return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i)
        {
            if (!block())
            {
                return 1;
            }
        }
        if (!device.synchronize(error))
        {
            return 1;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / reps;
        std::printf("block: %.2f ms\n", seconds * 1e3);
    }
    return ours.update <= torch.update ? 0 : 1;
}
