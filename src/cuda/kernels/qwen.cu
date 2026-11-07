#include <cuda_bf16.h>

// Pieces of the Qwen2.5 text model for the Cosmos-Reason1 text encoder (transformers
// modeling_qwen2_5_vl.py Qwen2_5_VLTextModel; flashdreams infra/encoder/text/cosmos_reason1.py):
// f32 residual rows, bf16 matrix inputs, one block of 256 threads per row of width a multiple of
// 256 up to 4096.

namespace
{

__device__ __forceinline__ float load_bf16(const unsigned short* p)
{
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));
}

__device__ __forceinline__ unsigned short store_bf16(float v)
{
    const __nv_bfloat16 h = __float2bfloat16_rn(v);
    return *reinterpret_cast<const unsigned short*>(&h);
}

__device__ __forceinline__ float block_sum(float v, float* shared)
{
    for (unsigned int offset = 16; offset > 0; offset /= 2)
    {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0)
    {
        shared[threadIdx.x / 32] = v;
    }
    __syncthreads();
    float total = 0.0f;
    for (unsigned int w = 0; w < blockDim.x / 32; ++w)
    {
        total += shared[w];
    }
    return total;
}

// Qwen2RMSNorm: x * rsqrt(mean(x^2) + eps) * weight, into v (count values a thread)
__device__ __forceinline__ void rms_norm_row(float* v, unsigned int count, const unsigned short* gamma,
                                             unsigned int width, float eps, float* shared)
{
    float squares = 0.0f;
    for (unsigned int i = 0; i < count; ++i)
    {
        squares = fmaf(v[i], v[i], squares);
    }
    const float inverse = rsqrtf(block_sum(squares, shared) / static_cast<float>(width) + eps);
    for (unsigned int i = 0; i < count; ++i)
    {
        v[i] = v[i] * inverse * load_bf16(gamma + threadIdx.x + i * 256);
    }
}

}

// out (bf16 rows of width) = Qwen2RMSNorm(x) with gamma
extern "C" __global__ void __launch_bounds__(256)
    qwen_rms_norm(const float* x, const unsigned short* gamma, unsigned short* out, unsigned int width, float eps)
{
    __shared__ float partial[8];
    const float* row = x + static_cast<unsigned long long>(blockIdx.x) * width;
    float v[16];
    const unsigned int count = width / 256;
    for (unsigned int i = 0; i < count; ++i)
    {
        v[i] = row[threadIdx.x + i * 256];
    }
    rms_norm_row(v, count, gamma, width, eps, partial);
    unsigned short* destination = out + static_cast<unsigned long long>(blockIdx.x) * width;
    for (unsigned int i = 0; i < count; ++i)
    {
        destination[threadIdx.x + i * 256] = store_bf16(v[i]);
    }
}

// Qwen2MLP's act_fn(gate_proj(x)) * up_proj(x): rows of [gate (inner) | up (inner)] bf16 to rows of
// inner bf16
extern "C" __global__ void __launch_bounds__(256)
    qwen_swiglu(const unsigned short* gate_up, unsigned short* out, unsigned int inner, unsigned long long count)
{
    for (unsigned long long i = blockIdx.x * 256ull + threadIdx.x; i < count; i += gridDim.x * 256ull)
    {
        const unsigned long long row = i / inner;
        const unsigned int c = static_cast<unsigned int>(i % inner);
        const float g = load_bf16(gate_up + row * 2 * inner + c);
        const float u = load_bf16(gate_up + row * 2 * inner + inner + c);
        out[i] = store_bf16(g / (1.0f + expf(-g)) * u);
    }
}

// one hidden state of the embedding: the row (through the final Qwen2RMSNorm first when gamma is
// not null, as transformers swaps the last hidden state for the normed last_hidden_state), then
// CosmosReason1TextEncoder._mean_normalize: (h - mean) / (std + 1e-8), std with N - 1, as bf16 at
// out + row * out_stride
extern "C" __global__ void __launch_bounds__(256)
    qwen_hidden_out(const float* x, const unsigned short* gamma, unsigned short* out, unsigned int width,
                    unsigned long long out_stride, float eps)
{
    __shared__ float partial[8];
    const float* row = x + static_cast<unsigned long long>(blockIdx.x) * width;
    float v[16];
    const unsigned int count = width / 256;
    for (unsigned int i = 0; i < count; ++i)
    {
        v[i] = row[threadIdx.x + i * 256];
    }
    if (gamma != nullptr)
    {
        rms_norm_row(v, count, gamma, width, eps, partial);
    }
    float sum = 0.0f;
    for (unsigned int i = 0; i < count; ++i)
    {
        sum += v[i];
    }
    const float mean = block_sum(sum, partial) / static_cast<float>(width);
    float squares = 0.0f;
    for (unsigned int i = 0; i < count; ++i)
    {
        const float c = v[i] - mean;
        squares = fmaf(c, c, squares);
    }
    const float deviation = sqrtf(block_sum(squares, partial) / static_cast<float>(width - 1));
    const float inverse = 1.0f / (deviation + 1e-8f);
    unsigned short* destination = out + static_cast<unsigned long long>(blockIdx.x) * out_stride;
    for (unsigned int i = 0; i < count; ++i)
    {
        destination[threadIdx.x + i * 256] = store_bf16((v[i] - mean) * inverse);
    }
}
