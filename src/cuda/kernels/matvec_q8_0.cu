#include <cuda_fp16.h>

__device__ __forceinline__ float dot16(uint4 packed, const float values[16])
{
    const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
    float sum = 0;
#pragma unroll
    for (unsigned int i = 0; i < 16; ++i)
    {
        const signed char q = static_cast<signed char>(words[i / 4] >> (8 * (i % 4)));
        sum = __fmaf_rn(static_cast<float>(q), values[i], sum);
    }
    return sum;
}

// One warp, R packed rows (row_quants, row_scales), up to 4 tokens from token. Each lane loads
// its 16 input values once for all R rows; every row is summed in the same order whatever R
// is. The totals end in lane 0.
template <unsigned int R>
__device__ __forceinline__ void row_sums(const uint4* const row_quants[R], const __half* const row_scales[R],
                                         const float* input, unsigned int n_in, unsigned int token,
                                         unsigned int n_tokens, unsigned int lane, float sums[R][4])
{
    const unsigned int chunks = n_in / 16;
#pragma unroll
    for (unsigned int r = 0; r < R; ++r)
    {
#pragma unroll
        for (unsigned int slot = 0; slot < 4; ++slot)
        {
            sums[r][slot] = 0;
        }
    }
    for (unsigned int chunk = lane; chunk < chunks; chunk += 32)
    {
        uint4 packed[R];
        float scale[R];
#pragma unroll
        for (unsigned int r = 0; r < R; ++r)
        {
            packed[r] = __ldg(row_quants[r] + chunk);
            scale[r] = __half2float(row_scales[r][chunk / 2]);
        }
        const unsigned int column = chunk * 16;
#pragma unroll
        for (unsigned int slot = 0; slot < 4; ++slot)
        {
            if (token + slot < n_tokens)
            {
                const float* x = input + static_cast<unsigned long long>(token + slot) * n_in + column;
                const float4 a = *reinterpret_cast<const float4*>(x);
                const float4 b = *reinterpret_cast<const float4*>(x + 4);
                const float4 c = *reinterpret_cast<const float4*>(x + 8);
                const float4 d = *reinterpret_cast<const float4*>(x + 12);
                const float values[16] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w,
                                          c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
#pragma unroll
                for (unsigned int r = 0; r < R; ++r)
                {
                    sums[r][slot] = __fmaf_rn(scale[r], dot16(packed[r], values), sums[r][slot]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned int r = 0; r < R; ++r)
    {
#pragma unroll
        for (unsigned int slot = 0; slot < 4; ++slot)
        {
            for (unsigned int offset = 16; offset > 0; offset /= 2)
            {
                sums[r][slot] += __shfl_down_sync(0xffffffffu, sums[r][slot], offset);
            }
        }
    }
}

// Up to three matrices over the same input, stacked by rows (q, k and v), each written to its
// own output; accumulate adds the products to the outputs (the residual connection).
struct MatvecSegments
{
    const unsigned char* quants[3];
    const __half* scales[3];
    float* output[3];
    unsigned int rows[3];
};

// Warp w computes rows R*w .. R*w + R - 1; every segment has a multiple of R rows.
template <unsigned int R>
__device__ __forceinline__ void matvec_q8_0_body(const MatvecSegments& segments, const float* input,
                                                 unsigned int n_in, unsigned int n_tokens, unsigned int accumulate)
{
    const unsigned int lane = threadIdx.x & 31;
    unsigned int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R;
    const unsigned int token = blockIdx.y * 4;
    unsigned int segment = 0;
    while (segment < 3 && row >= segments.rows[segment])
    {
        row -= segments.rows[segment];
        ++segment;
    }
    if (segment == 3)
    {
        return;
    }
    const uint4* row_quants[R];
    const __half* row_scales[R];
#pragma unroll
    for (unsigned int r = 0; r < R; ++r)
    {
        row_quants[r] = reinterpret_cast<const uint4*>(segments.quants[segment] + static_cast<unsigned long long>(row + r) * n_in);
        row_scales[r] = segments.scales[segment] + static_cast<unsigned long long>(row + r) * (n_in / 32);
    }
    float sums[R][4];
    row_sums<R>(row_quants, row_scales, input, n_in, token, n_tokens, lane, sums);
    if (lane != 0)
    {
        return;
    }
    const unsigned int width = segments.rows[segment];
    float* output = segments.output[segment];
#pragma unroll
    for (unsigned int r = 0; r < R; ++r)
    {
#pragma unroll
        for (unsigned int slot = 0; slot < 4; ++slot)
        {
            if (token + slot < n_tokens)
            {
                float& out = output[static_cast<unsigned long long>(token + slot) * width + row + r];
                out = accumulate != 0 ? out + sums[r][slot] : sums[r][slot];
            }
        }
    }
}

extern "C" __global__ void matvec_q8_0(MatvecSegments segments, const float* input, unsigned int n_in,
                                      unsigned int n_tokens, unsigned int accumulate)
{
    matvec_q8_0_body<1>(segments, input, n_in, n_tokens, accumulate);
}

extern "C" __global__ void matvec_q8_0_rows4(MatvecSegments segments, const float* input, unsigned int n_in,
                                            unsigned int n_tokens, unsigned int accumulate)
{
    matvec_q8_0_body<4>(segments, input, n_in, n_tokens, accumulate);
}

// The gate and up rows of R outputs together, then swiglu as kernels/swiglu.cu computes it.
template <unsigned int R>
__device__ __forceinline__ void matvec_q8_0_swiglu_body(const unsigned char* gate_quants, const __half* gate_scales,
                                                        const unsigned char* up_quants, const __half* up_scales,
                                                        const float* input, float* output, unsigned int n_in,
                                                        unsigned int n_out, unsigned int n_tokens)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R;
    const unsigned int token = blockIdx.y * 4;
    if (row >= n_out)
    {
        return;
    }
    const uint4* row_quants[2 * R];
    const __half* row_scales[2 * R];
#pragma unroll
    for (unsigned int r = 0; r < R; ++r)
    {
        row_quants[r] = reinterpret_cast<const uint4*>(gate_quants + static_cast<unsigned long long>(row + r) * n_in);
        row_scales[r] = gate_scales + static_cast<unsigned long long>(row + r) * (n_in / 32);
        row_quants[R + r] = reinterpret_cast<const uint4*>(up_quants + static_cast<unsigned long long>(row + r) * n_in);
        row_scales[R + r] = up_scales + static_cast<unsigned long long>(row + r) * (n_in / 32);
    }
    float sums[2 * R][4];
    row_sums<2 * R>(row_quants, row_scales, input, n_in, token, n_tokens, lane, sums);
    if (lane != 0)
    {
        return;
    }
#pragma unroll
    for (unsigned int r = 0; r < R; ++r)
    {
#pragma unroll
        for (unsigned int slot = 0; slot < 4; ++slot)
        {
            if (token + slot < n_tokens)
            {
                const float value = sums[r][slot];
                output[static_cast<unsigned long long>(token + slot) * n_out + row + r] =
                    (value / (1.0f + expf(-value))) * sums[R + r][slot];
            }
        }
    }
}

extern "C" __global__ void matvec_q8_0_swiglu(const unsigned char* gate_quants, const __half* gate_scales,
                                             const unsigned char* up_quants, const __half* up_scales,
                                             const float* input, float* output, unsigned int n_in,
                                             unsigned int n_out, unsigned int n_tokens)
{
    matvec_q8_0_swiglu_body<1>(gate_quants, gate_scales, up_quants, up_scales, input, output, n_in, n_out, n_tokens);
}

extern "C" __global__ void matvec_q8_0_swiglu_rows2(const unsigned char* gate_quants, const __half* gate_scales,
                                                   const unsigned char* up_quants, const __half* up_scales,
                                                   const float* input, float* output, unsigned int n_in,
                                                   unsigned int n_out, unsigned int n_tokens)
{
    matvec_q8_0_swiglu_body<2>(gate_quants, gate_scales, up_quants, up_scales, input, output, n_in, n_out, n_tokens);
}
