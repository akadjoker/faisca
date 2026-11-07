#include <cuda_fp16.h>

// Activations quantized to Q8_0 as the CPU runner does (src/ops.cpp quantize_q8_0, after
// llama.cpp): per block of 32, d = amax / 127 stored as f16, values round(x * 127 / amax) to
// nearest even. One warp per block; quants as rows of n_in bytes, scales block major:
// scales[block * token_stride + token].
extern "C" __global__ void quantize_q8_0(const float* input, signed char* quants, __half* scales, unsigned int n_in,
                                        unsigned int token_stride)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int block = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const unsigned int token = blockIdx.y;
    if (block >= n_in / 32)
    {
        return;
    }
    const unsigned long long index = static_cast<unsigned long long>(token) * n_in + block * 32 + lane;
    const float value = input[index];
    float amax = fabsf(value);
    for (unsigned int offset = 16; offset > 0; offset /= 2)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float d = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    quants[index] = static_cast<signed char>(__float2int_rn(value * id));
    if (lane == 0)
    {
        scales[static_cast<unsigned long long>(block) * token_stride + token] = __float2half_rn(d);
    }
}

__device__ __forceinline__ void mma_s8(const int a[4], const int b[2], int c[4])
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11, %12, %13};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0), "r"(0), "r"(0), "r"(0));
}

// 16 bytes from global to shared memory without the registers; outside the source the
// destination is filled with zeros
__device__ __forceinline__ void copy_async(void* shared, const void* global, bool inside)
{
    const unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    const unsigned int size = inside ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(global), "r"(size));
}

// Q8_0 weights times Q8_0 activations on the int8 tensor cores. Weights: quants rows of n_in
// bytes, f16 scales row major, then the same scales block major (scales_by_block[block * n_out +
// row]); activations as quantize_q8_0 writes them. Block (x, y): 64 rows by 64 tokens; warp w:
// 32 rows (w % 2) by 32 tokens (w / 2) as 2 x 4 tiles of mma.sync m16n8k32, one per Q8_0 block,
// so each block's integer sum is exact; then, block by block in order,
// acc = fma(float(sum), d_w * d_x, acc), as matmul_q8_0_packed on the CPU. Per stage the quants
// of 4 blocks (rows padded to 144 bytes against bank conflicts) and their scales go to shared
// memory with cp.async, two stages in flight.
// Fragments (PTX ISA, m16n8k32 .s8): g = lane / 4, t = lane % 4; A rows g and g + 8, bytes
// 4t..4t+3 and 16+4t..; B column g, the same bytes; C rows g and g + 8, columns 2t, 2t + 1.
extern "C" __global__ void gemm_q8_0(const signed char* weight_quants, const __half* weight_scales_by_block,
                                    const signed char* quants, const __half* scales, float* output, unsigned int n_in,
                                    unsigned int n_out, unsigned int n_tokens, unsigned int padded_tokens,
                                    unsigned int token_stride, unsigned int accumulate)
{
    constexpr unsigned int M = 2;
    constexpr unsigned int N = 4;
    constexpr unsigned int KB = 4;    // Q8_0 blocks per stage
    constexpr unsigned int ROW = 144; // bytes per staged row
    __shared__ __align__(16) signed char tile_a[2][64 * ROW];
    __shared__ __align__(16) signed char tile_b[2][64 * ROW];
    __shared__ __align__(16) __half tile_w[2][KB][64];
    __shared__ __align__(16) __half tile_x[2][KB][64];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int g = lane / 4;
    const unsigned int t = lane % 4;
    const unsigned int block_row = blockIdx.x * 64;
    const unsigned int block_token = blockIdx.y * 64;
    const unsigned int blocks = n_in / 32;
    const unsigned int stages = (blocks + KB - 1) / KB;

    const auto load_stage = [&](unsigned int stage, unsigned int buffer) {
        const unsigned int first = stage * KB;
        // quants: 64 rows x 8 chunks of 16 bytes for each operand, 4 + 4 per thread
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i)
        {
            const unsigned int index = threadIdx.x + i * 128;
            const unsigned int r = index / 8;
            const unsigned int chunk = index % 8;
            const bool in_k = first + chunk / 2 < blocks;
            const bool row_inside = block_row + r < n_out && in_k;
            copy_async(tile_a[buffer] + r * ROW + chunk * 16,
                       row_inside ? weight_quants + static_cast<unsigned long long>(block_row + r) * n_in + first * 32 + chunk * 16
                                  : weight_quants,
                       row_inside);
            const bool token_inside = block_token + r < padded_tokens && in_k;
            copy_async(tile_b[buffer] + r * ROW + chunk * 16,
                       token_inside ? quants + static_cast<unsigned long long>(block_token + r) * n_in + first * 32 + chunk * 16
                                    : quants,
                       token_inside);
        }
        // scales: per block 64 rows (or tokens) of f16, 8 chunks of 16 bytes; threads 0..31 the
        // weights', 32..63 the activations'
        if (threadIdx.x < 64)
        {
            const unsigned int kb = (threadIdx.x % 32) / 8;
            const unsigned int chunk = threadIdx.x % 8;
            const unsigned int block = first + kb;
            if (threadIdx.x < 32)
            {
                const bool inside = block < blocks && block_row + chunk * 8 < n_out;
                copy_async(&tile_w[buffer][kb][chunk * 8],
                           inside ? weight_scales_by_block + static_cast<unsigned long long>(block) * n_out + block_row + chunk * 8
                                  : weight_scales_by_block,
                           inside);
            }
            else
            {
                const bool inside = block < blocks && block_token + chunk * 8 < token_stride;
                copy_async(&tile_x[buffer][kb][chunk * 8],
                           inside ? scales + static_cast<unsigned long long>(block) * token_stride + block_token + chunk * 8
                                  : scales,
                           inside);
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    const unsigned int row0 = (warp % 2) * 32;
    const unsigned int token0 = (warp / 2) * 32;
    float acc[M][N][4] = {};
    load_stage(0, 0);
    for (unsigned int stage = 0; stage < stages; ++stage)
    {
        const unsigned int buffer = stage & 1;
        if (stage + 1 < stages)
        {
            load_stage(stage + 1, buffer ^ 1);
            asm volatile("cp.async.wait_group 1;");
        }
        else
        {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
#pragma unroll
        for (unsigned int kb = 0; kb < KB; ++kb)
        {
            if (stage * KB + kb >= blocks)
            {
                break;
            }
            int a[M][4];
            float w[M][2];
#pragma unroll
            for (unsigned int m = 0; m < M; ++m)
            {
                const unsigned int r = row0 + m * 16 + g;
                const signed char* low = tile_a[buffer] + r * ROW + kb * 32 + 4 * t;
                const signed char* high = low + 8 * ROW;
                a[m][0] = *reinterpret_cast<const int*>(low);
                a[m][1] = *reinterpret_cast<const int*>(high);
                a[m][2] = *reinterpret_cast<const int*>(low + 16);
                a[m][3] = *reinterpret_cast<const int*>(high + 16);
                w[m][0] = __half2float(tile_w[buffer][kb][r]);
                w[m][1] = __half2float(tile_w[buffer][kb][r + 8]);
            }
            int b[N][2];
            float x[N][2];
#pragma unroll
            for (unsigned int n = 0; n < N; ++n)
            {
                const signed char* column = tile_b[buffer] + (token0 + n * 8 + g) * ROW + kb * 32 + 4 * t;
                b[n][0] = *reinterpret_cast<const int*>(column);
                b[n][1] = *reinterpret_cast<const int*>(column + 16);
                const unsigned int token = token0 + n * 8 + 2 * t;
                x[n][0] = __half2float(tile_x[buffer][kb][token]);
                x[n][1] = __half2float(tile_x[buffer][kb][token + 1]);
            }
#pragma unroll
            for (unsigned int m = 0; m < M; ++m)
            {
#pragma unroll
                for (unsigned int n = 0; n < N; ++n)
                {
                    int c[4];
                    mma_s8(a[m], b[n], c);
                    acc[m][n][0] = __fmaf_rn(static_cast<float>(c[0]), w[m][0] * x[n][0], acc[m][n][0]);
                    acc[m][n][1] = __fmaf_rn(static_cast<float>(c[1]), w[m][0] * x[n][1], acc[m][n][1]);
                    acc[m][n][2] = __fmaf_rn(static_cast<float>(c[2]), w[m][1] * x[n][0], acc[m][n][2]);
                    acc[m][n][3] = __fmaf_rn(static_cast<float>(c[3]), w[m][1] * x[n][1], acc[m][n][3]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned int m = 0; m < M; ++m)
    {
#pragma unroll
        for (unsigned int n = 0; n < N; ++n)
        {
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
            {
                const unsigned int token = block_token + token0 + n * 8 + 2 * t + (i & 1);
                const unsigned int row = block_row + row0 + m * 16 + g + (i >= 2 ? 8 : 0);
                if (row < n_out && token < n_tokens)
                {
                    float& out = output[static_cast<unsigned long long>(token) * n_out + row];
                    out = accumulate != 0 ? out + acc[m][n][i] : acc[m][n][i];
                }
            }
        }
    }
}
