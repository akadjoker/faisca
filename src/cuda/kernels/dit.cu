#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Elementwise pieces of the OmniDreams (Cosmos-Predict2.5) DiT block, FlashDreams
// integrations_v2/omnidreams/impl/transformer/modules.py Block.forward: the residual stream
// stays in f32, the matrix inputs are rounded once to f16 / bf16.

namespace
{

template <bool bf16>
__device__ __forceinline__ float load_half(const unsigned short* p)
{
    if (bf16)
    {
        return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));
    }
    return __half2float(*reinterpret_cast<const __half*>(p));
}

template <bool bf16>
__device__ __forceinline__ unsigned short store_half(float v)
{
    if (bf16)
    {
        const __nv_bfloat16 h = __float2bfloat16_rn(v);
        return *reinterpret_cast<const unsigned short*>(&h);
    }
    const __half h = __float2half_rn(v);
    return *reinterpret_cast<const unsigned short*>(&h);
}

__device__ __forceinline__ float block_sum(float v, float* shared)
{
    for (unsigned int offset = 16; offset > 0; offset /= 2)
    {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    const unsigned int warp = threadIdx.x / 32;
    __syncthreads();
    if ((threadIdx.x & 31) == 0)
    {
        shared[warp] = v;
    }
    __syncthreads();
    float total = 0.0f;
    for (unsigned int w = 0; w < blockDim.x / 32; ++w)
    {
        total += shared[w];
    }
    return total;
}

// out = layer_norm(x) * (1 + scale) + shift, nn.LayerNorm(elementwise_affine=False, eps=1e-6)
// (Block.forward lines 1031, 1056, 1066): 256 threads per row, width a multiple of 256 up to
// 4096, mean then the variance around it in f32
template <bool bf16>
__device__ __forceinline__ void modulate_norm(const float* x, const float* shift, const float* scale,
                                              unsigned short* out, unsigned int width)
{
    __shared__ float partial[8];
    const float* row = x + static_cast<unsigned long long>(blockIdx.x) * width;
    float v[16];
    const unsigned int count = width / 256;
    float sum = 0.0f;
    for (unsigned int i = 0; i < count; ++i)
    {
        v[i] = row[threadIdx.x + i * 256];
        sum += v[i];
    }
    const float mean = block_sum(sum, partial) / static_cast<float>(width);
    float squares = 0.0f;
    for (unsigned int i = 0; i < count; ++i)
    {
        const float c = v[i] - mean;
        squares = fmaf(c, c, squares);
    }
    const float variance = block_sum(squares, partial) / static_cast<float>(width);
    const float inverse = rsqrtf(variance + 1e-6f);
    unsigned short* destination = out + static_cast<unsigned long long>(blockIdx.x) * width;
    for (unsigned int i = 0; i < count; ++i)
    {
        const unsigned int c = threadIdx.x + i * 256;
        destination[c] = store_half<bf16>(fmaf((v[i] - mean) * inverse, 1.0f + scale[c], shift[c]));
    }
}

// In place on heads of 128 at column first + head * 128 of rows stride halves apart: RMSNorm
// over the head with gamma (skipped when gamma is null) (nn.RMSNorm(128, eps=1e-6), MultiHeadAttention q_norm / k_norm),
// then, with angles, the rotation of pairs (d, d + 64) by angles[row * 64 + d]
// (rope_kernel.py, non-interleaved). One warp per (row, head); lane l holds d = l, l + 32,
// l + 64, l + 96, so both pairs of a lane stay in the lane.
template <bool bf16>
__device__ __forceinline__ void qk_norm_rope(unsigned short* data, unsigned int stride, unsigned int first,
                                             const unsigned short* gamma, const float* angles, unsigned int rows,
                                             unsigned int heads)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (item >= rows * heads)
    {
        return;
    }
    const unsigned int row = item / heads;
    const unsigned int head = item % heads;
    unsigned short* p = data + static_cast<unsigned long long>(row) * stride + first + head * 128;
    float v[4];
    float squares = 0.0f;
    for (unsigned int i = 0; i < 4; ++i)
    {
        v[i] = load_half<bf16>(p + lane + 32 * i);
        squares = fmaf(v[i], v[i], squares);
    }
    for (unsigned int offset = 16; offset > 0; offset /= 2)
    {
        squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    }
    const float inverse = rsqrtf(squares / 128.0f + 1e-6f);
    if (gamma != nullptr)
    {
        for (unsigned int i = 0; i < 4; ++i)
        {
            v[i] = v[i] * inverse * load_half<bf16>(gamma + lane + 32 * i);
        }
    }
    if (angles != nullptr)
    {
        for (unsigned int i = 0; i < 2; ++i)
        {
            float s;
            float c;
            sincosf(angles[static_cast<unsigned long long>(row) * 64 + lane + 32 * i], &s, &c);
            const float a = v[i];
            const float b = v[i + 2];
            v[i] = a * c - b * s;
            v[i + 2] = b * c + a * s;
        }
    }
    for (unsigned int i = 0; i < 4; ++i)
    {
        p[lane + 32 * i] = store_half<bf16>(v[i]);
    }
}

}

// out[n] = add[n] + sum_k w[n][k] * f(x[k]), f = SiLU when silu (nn.SiLU before the linear of
// the AdaLN and time embedding MLPs), bf16 weights, f32 vectors; one warp per row, k a multiple
// of 256
extern "C" __global__ void __launch_bounds__(256)
    dit_matvec_bf16(const unsigned short* w, const float* x, const float* add, float* out, unsigned int n,
                    unsigned int k, unsigned int silu)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= n)
    {
        return;
    }
    const unsigned short* wr = w + static_cast<unsigned long long>(row) * k;
    float sum = 0.0f;
    for (unsigned int c = lane * 8; c < k; c += 256)
    {
        const uint4 packed = *reinterpret_cast<const uint4*>(wr + c);
        const unsigned short* h = reinterpret_cast<const unsigned short*>(&packed);
        for (unsigned int i = 0; i < 8; ++i)
        {
            float v = x[c + i];
            if (silu != 0)
            {
                v = v / (1.0f + expf(-v));
            }
            sum = fmaf(load_half<true>(h + i), v, sum);
        }
    }
    for (unsigned int offset = 16; offset > 0; offset /= 2)
    {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0)
    {
        out[row] = add != nullptr ? sum + add[row] : sum;
    }
}

// count 16 byte words from src to dst, the ranges not overlapping
extern "C" __global__ void __launch_bounds__(256)
    dit_copy(const uint4* src, uint4* dst, unsigned long long count)
{
    for (unsigned long long i = blockIdx.x * 256ull + threadIdx.x; i < count; i += gridDim.x * 256ull)
    {
        dst[i] = src[i];
    }
}

extern "C" __global__ void __launch_bounds__(256)
    dit_modulate_norm_f16(const float* x, const float* shift, const float* scale, unsigned short* out,
                          unsigned int width)
{
    modulate_norm<false>(x, shift, scale, out, width);
}

extern "C" __global__ void __launch_bounds__(256)
    dit_modulate_norm_bf16(const float* x, const float* shift, const float* scale, unsigned short* out,
                           unsigned int width)
{
    modulate_norm<true>(x, shift, scale, out, width);
}

extern "C" __global__ void __launch_bounds__(256)
    dit_qk_norm_rope_f16(unsigned short* data, unsigned int stride, unsigned int first, const unsigned short* gamma,
                         const float* angles, unsigned int rows, unsigned int heads)
{
    qk_norm_rope<false>(data, stride, first, gamma, angles, rows, heads);
}

extern "C" __global__ void __launch_bounds__(256)
    dit_qk_norm_rope_bf16(unsigned short* data, unsigned int stride, unsigned int first, const unsigned short* gamma,
                          const float* angles, unsigned int rows, unsigned int heads)
{
    qk_norm_rope<true>(data, stride, first, gamma, angles, rows, heads);
}
