#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Attention over f16 / bf16 heads of 128 on the tensor cores, no mask, after FlashAttention-2's
// forward kernel (flash-attention csrc/flash_attn/src, commit 7c53542):
// - configuration for sm86 / sm89 without a causal mask: Flash_fwd_kernel_traits<128, 128, 32, 4>
//   (flash_fwd_launch_template.h:255-259), 128 queries x 32 keys per step, 4 warps, Q, K and V
//   in 48 KB of shared memory; TiledMma of 4 warps along M with a 64 x 16 x 16 tile, so warp w
//   owns query rows 16w and 64 + 16w (kernel_traits.h:66-69);
// - rows of 128 halves swizzled as Swizzle<3, 3, 3> over 8 x 64 atoms (kernel_traits.h:71-76):
//   16 byte chunk c of row r at c ^ (r & 7);
// - main loop of compute_attn_1rowblock (flash_fwd_kernel.h): key blocks from the last to the
//   first; wait for K, issue V, S = Q K^T, mask the keys past n_kv on the first step only, wait
//   for V, issue the next K, online softmax, P to f16 in registers, O += P V with V^T through
//   ldmatrix.trans;
// - with causal, query i sees keys 0..i (n_q == n_kv, no cache offset): only the key blocks up to
//   the diagonal, the keys past each row masked to -inf, and softmax_rescale_o's Check_inf (a row
//   max of -inf counts as 0 in the rescale, so a fully masked first block gives zeros, not NaN);
// - GQA: query head h reads key / value head h / kv_group;
// - softmax.h: exp2f(s * scale_log2 - max * scale_log2) with the fused multiply add, per thread
//   sums reduced over the quad (shuffle 2 then 1) only at the end, 1 / sum, then O to f16.

namespace
{

constexpr unsigned int BLOCK_M = 128;
constexpr unsigned int BLOCK_N = 32;
constexpr unsigned int HEAD = 128;

__device__ __forceinline__ unsigned int swizzled(unsigned int row, unsigned int chunk)
{
    return row * HEAD + ((chunk ^ (row & 7)) * 8);
}

__device__ __forceinline__ void copy_async(void* shared, const void* global, bool inside)
{
    const unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    const unsigned int size = inside ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(global), "r"(size));
}

__device__ __forceinline__ void load_matrix_x4(const void* shared, unsigned int r[4])
{
    const unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

__device__ __forceinline__ void load_matrix_x4_transposed(const void* shared, unsigned int r[4])
{
    const unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

template <bool bf16>
__device__ __forceinline__ void mma(const unsigned int a[4], unsigned int b0, unsigned int b1, float c[4])
{
    if (bf16)
    {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    else
    {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
}

template <bool bf16>
__device__ __forceinline__ unsigned int pack(float low, float high)
{
    if (bf16)
    {
        const __nv_bfloat162 v = __floats2bfloat162_rn(low, high);
        return *reinterpret_cast<const unsigned int*>(&v);
    }
    const __half2 v = __floats2half2_rn(low, high);
    return *reinterpret_cast<const unsigned int*>(&v);
}

// one head per blockIdx.y, at column head * 128 of each row; strides in elements
template <bool bf16>
__device__ __forceinline__ void attention_tiles(const unsigned short* q, unsigned int q_stride, const unsigned short* k,
                                                const unsigned short* v, unsigned int kv_stride, unsigned short* out,
                                                unsigned int out_stride, unsigned int n_q, unsigned int n_kv,
                                                float scale_log2, unsigned int kv_group, unsigned int causal)
{
    __shared__ __align__(128) unsigned short tile_q[BLOCK_M * HEAD];
    __shared__ __align__(128) unsigned short tile_k[BLOCK_N * HEAD];
    __shared__ __align__(128) unsigned short tile_v[BLOCK_N * HEAD];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int g = lane / 4;
    const unsigned int t = lane % 4;
    const unsigned int m0 = blockIdx.x * BLOCK_M;
    const unsigned int column = blockIdx.y * HEAD;
    const unsigned int kv_column = (blockIdx.y / kv_group) * HEAD;

    const auto load_kv = [&](const unsigned short* source, unsigned short* tile, unsigned int block) {
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i)
        {
            const unsigned int index = threadIdx.x + i * 128;
            const unsigned int row = index / 16;
            const unsigned int chunk = index % 16;
            const unsigned int key = block * BLOCK_N + row;
            const bool inside = key < n_kv;
            copy_async(tile + swizzled(row, chunk),
                       inside ? source + static_cast<unsigned long long>(key) * kv_stride + kv_column + chunk * 8 : source,
                       inside);
        }
        asm volatile("cp.async.commit_group;");
    };

#pragma unroll
    for (unsigned int i = 0; i < 16; ++i)
    {
        const unsigned int index = threadIdx.x + i * 128;
        const unsigned int row = index / 16;
        const unsigned int chunk = index % 16;
        const bool inside = m0 + row < n_q;
        copy_async(tile_q + swizzled(row, chunk),
                   inside ? q + static_cast<unsigned long long>(m0 + row) * q_stride + column + chunk * 8 : q, inside);
    }
    const unsigned int last_key = causal != 0 ? (m0 + BLOCK_M < n_kv ? m0 + BLOCK_M : n_kv) : n_kv;
    const unsigned int blocks = (last_key + BLOCK_N - 1) / BLOCK_N;
    load_kv(k, tile_k, blocks - 1);

    float acc_o[2][16][4] = {};
    float row_max[2][2];
    float row_sum[2][2];
    for (int block = static_cast<int>(blocks) - 1; block >= 0; --block)
    {
        const bool first = block == static_cast<int>(blocks) - 1;
        float acc_s[2][4][4] = {};
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        load_kv(v, tile_v, static_cast<unsigned int>(block));
#pragma unroll
        for (unsigned int kk = 0; kk < HEAD / 16; ++kk)
        {
            unsigned int a[2][4];
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i)
            {
                const unsigned int row = i * 64 + warp * 16 + lane % 16;
                load_matrix_x4(tile_q + swizzled(row, kk * 2 + lane / 16), a[i]);
            }
            unsigned int b[2][4];
#pragma unroll
            for (unsigned int np = 0; np < 2; ++np)
            {
                const unsigned int row = np * 16 + lane % 8 + (lane / 16) * 8;
                load_matrix_x4(tile_k + swizzled(row, kk * 2 + (lane / 8) % 2), b[np]);
            }
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i)
            {
#pragma unroll
                for (unsigned int np = 0; np < 2; ++np)
                {
                    mma<bf16>(a[i], b[np][0], b[np][1], acc_s[i][2 * np]);
                    mma<bf16>(a[i], b[np][2], b[np][3], acc_s[i][2 * np + 1]);
                }
            }
        }
        if (causal != 0 && static_cast<unsigned int>(block) * BLOCK_N + BLOCK_N - 1 > m0)
        {
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i)
            {
#pragma unroll
                for (unsigned int nt = 0; nt < 4; ++nt)
                {
#pragma unroll
                    for (unsigned int e = 0; e < 4; ++e)
                    {
                        const unsigned int key = static_cast<unsigned int>(block) * BLOCK_N + nt * 8 + 2 * t + e % 2;
                        const unsigned int query = m0 + i * 64 + warp * 16 + g + (e / 2) * 8;
                        if (key > query)
                        {
                            acc_s[i][nt][e] = -INFINITY;
                        }
                    }
                }
            }
        }
        if (first && n_kv % BLOCK_N != 0)
        {
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i)
            {
#pragma unroll
                for (unsigned int nt = 0; nt < 4; ++nt)
                {
#pragma unroll
                    for (unsigned int e = 0; e < 4; ++e)
                    {
                        const unsigned int key = static_cast<unsigned int>(block) * BLOCK_N + nt * 8 + 2 * t + e % 2;
                        if (key >= n_kv)
                        {
                            acc_s[i][nt][e] = -INFINITY;
                        }
                    }
                }
            }
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (block > 0)
        {
            load_kv(k, tile_k, static_cast<unsigned int>(block - 1));
        }

#pragma unroll
        for (unsigned int i = 0; i < 2; ++i)
        {
#pragma unroll
            for (unsigned int h = 0; h < 2; ++h)
            {
                const float previous = row_max[i][h];
                float m = first ? acc_s[i][0][2 * h] : fmaxf(previous, acc_s[i][0][2 * h]);
                m = fmaxf(m, acc_s[i][0][2 * h + 1]);
#pragma unroll
                for (unsigned int nt = 1; nt < 4; ++nt)
                {
                    m = fmaxf(m, acc_s[i][nt][2 * h]);
                    m = fmaxf(m, acc_s[i][nt][2 * h + 1]);
                }
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                row_max[i][h] = m;
                if (!first)
                {
                    const float current = causal != 0 && m == -INFINITY ? 0.0f : m;
                    const float rescale = exp2f((previous - current) * scale_log2);
                    row_sum[i][h] *= rescale;
#pragma unroll
                    for (unsigned int dt = 0; dt < 16; ++dt)
                    {
                        acc_o[i][dt][2 * h] *= rescale;
                        acc_o[i][dt][2 * h + 1] *= rescale;
                    }
                }
                const float max_scaled = m == -INFINITY ? 0.0f : m * scale_log2;
                float sum = first ? 0.0f : row_sum[i][h];
#pragma unroll
                for (unsigned int nt = 0; nt < 4; ++nt)
                {
#pragma unroll
                    for (unsigned int e = 0; e < 2; ++e)
                    {
                        const float p = exp2f(fmaf(acc_s[i][nt][2 * h + e], scale_log2, -max_scaled));
                        acc_s[i][nt][2 * h + e] = p;
                        sum = first && nt == 0 && e == 0 ? p : sum + p;
                    }
                }
                row_sum[i][h] = sum;
            }
        }

#pragma unroll
        for (unsigned int j = 0; j < BLOCK_N / 16; ++j)
        {
            unsigned int p[2][4];
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i)
            {
                p[i][0] = pack<bf16>(acc_s[i][2 * j][0], acc_s[i][2 * j][1]);
                p[i][1] = pack<bf16>(acc_s[i][2 * j][2], acc_s[i][2 * j][3]);
                p[i][2] = pack<bf16>(acc_s[i][2 * j + 1][0], acc_s[i][2 * j + 1][1]);
                p[i][3] = pack<bf16>(acc_s[i][2 * j + 1][2], acc_s[i][2 * j + 1][3]);
            }
#pragma unroll
            for (unsigned int dp = 0; dp < 8; ++dp)
            {
                unsigned int b[4];
                const unsigned int key = j * 16 + lane % 8 + ((lane / 8) % 2) * 8;
                load_matrix_x4_transposed(tile_v + swizzled(key, dp * 2 + lane / 16), b);
#pragma unroll
                for (unsigned int i = 0; i < 2; ++i)
                {
                    mma<bf16>(p[i], b[0], b[1], acc_o[i][2 * dp]);
                    mma<bf16>(p[i], b[2], b[3], acc_o[i][2 * dp + 1]);
                }
            }
        }
    }

#pragma unroll
    for (unsigned int i = 0; i < 2; ++i)
    {
#pragma unroll
        for (unsigned int h = 0; h < 2; ++h)
        {
            float sum = row_sum[i][h];
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            const float inverse = sum == 0.0f || sum != sum ? 1.0f : 1.0f / sum;
            const unsigned int row = m0 + i * 64 + warp * 16 + g + h * 8;
            if (row >= n_q)
            {
                continue;
            }
            unsigned short* destination = out + static_cast<unsigned long long>(row) * out_stride + column + 2 * t;
#pragma unroll
            for (unsigned int dt = 0; dt < 16; ++dt)
            {
                *reinterpret_cast<unsigned int*>(destination + dt * 8) =
                    pack<bf16>(acc_o[i][dt][2 * h] * inverse, acc_o[i][dt][2 * h + 1] * inverse);
            }
        }
    }
}

}

extern "C" __global__ void __launch_bounds__(128)
    attention_f16(const unsigned short* q, unsigned int q_stride, const unsigned short* k, const unsigned short* v,
                  unsigned int kv_stride, unsigned short* out, unsigned int out_stride, unsigned int n_q,
                  unsigned int n_kv, float scale_log2, unsigned int kv_group, unsigned int causal)
{
    attention_tiles<false>(q, q_stride, k, v, kv_stride, out, out_stride, n_q, n_kv, scale_log2, kv_group, causal);
}

extern "C" __global__ void __launch_bounds__(128)
    attention_bf16(const unsigned short* q, unsigned int q_stride, const unsigned short* k, const unsigned short* v,
                   unsigned int kv_stride, unsigned short* out, unsigned int out_stride, unsigned int n_q,
                   unsigned int n_kv, float scale_log2, unsigned int kv_group, unsigned int causal)
{
    attention_tiles<true>(q, q_stride, k, v, kv_stride, out, out_stride, n_q, n_kv, scale_log2, kv_group, causal);
}
