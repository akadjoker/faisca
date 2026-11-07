#include <cuda_bf16.h>
#include <cuda_fp16.h>

// out[m][n] = sum_k x[m][k] * w[n][k] on the f16 / bf16 tensor cores with f32 accumulation.
// Tiling after CUTLASS's Sm80 tensor op GEMM (gemm/device/default_gemm_configuration.h:461,
// 128x256x64 with 64x64 warps and 3 stages), cut to 128x128x32 so the 3 stages fit the 48 KB of
// static shared memory: 4 warps of 64x64, cp.async multistage (CUTLASS mma_multistage.h order:
// wait for the oldest stage, sync, issue the next, compute). Shared rows of 32 halves (64 bytes)
// swizzled as flash-attention's Swizzle<2, 3, 3> for kBlockKSmem == 32 (kernel_traits.h): the 16
// byte chunk c of row r lives at chunk c ^ ((r >> 1) & 3), so the 8 rows of an ldmatrix phase hit
// 8 different bank groups. Fragments (PTX ISA, mma.m16n8k16 and ldmatrix.m8n8.x4): g = lane / 4,
// t = lane % 4; C rows g and g + 8, columns 2t and 2t + 1.

namespace
{

constexpr unsigned int TILE_M = 128;
constexpr unsigned int TILE_N = 128;
constexpr unsigned int TILE_K = 32;
constexpr unsigned int STAGES = 3;

__device__ __forceinline__ unsigned int swizzled(unsigned int row, unsigned int chunk)
{
    return row * TILE_K + ((chunk ^ ((row >> 1) & 3)) * 8);
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

// x: m rows of k halves, w: n rows of k halves, k a multiple of 32, n a multiple of 8, both
// row strides k; out: m rows of n floats
template <bool bf16>
__device__ __forceinline__ void gemm_tiles(const unsigned short* x, const unsigned short* w, float* out,
                                           unsigned int m, unsigned int n, unsigned int k)
{
    __shared__ __align__(128) unsigned short tile_a[STAGES][TILE_M * TILE_K];
    __shared__ __align__(128) unsigned short tile_b[STAGES][TILE_N * TILE_K];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int warp_m = (warp / 2) * 64;
    const unsigned int warp_n = (warp % 2) * 64;
    const unsigned int block_m = blockIdx.y * TILE_M;
    const unsigned int block_n = blockIdx.x * TILE_N;
    const unsigned int steps = k / TILE_K;

    const auto load_stage = [&](unsigned int step, unsigned int buffer) {
        if (step < steps)
        {
            const unsigned int k0 = step * TILE_K;
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
            {
                const unsigned int index = threadIdx.x + i * 128;
                const unsigned int row = index / 4;
                const unsigned int chunk = index % 4;
                const bool a_inside = block_m + row < m;
                copy_async(tile_a[buffer] + swizzled(row, chunk),
                           a_inside ? x + static_cast<unsigned long long>(block_m + row) * k + k0 + chunk * 8 : x,
                           a_inside);
                const bool b_inside = block_n + row < n;
                copy_async(tile_b[buffer] + swizzled(row, chunk),
                           b_inside ? w + static_cast<unsigned long long>(block_n + row) * k + k0 + chunk * 8 : w,
                           b_inside);
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc[4][8][4] = {};
#pragma unroll
    for (unsigned int s = 0; s + 1 < STAGES; ++s)
    {
        load_stage(s, s);
    }
    for (unsigned int step = 0; step < steps; ++step)
    {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
        __syncthreads();
        load_stage(step + STAGES - 1, (step + STAGES - 1) % STAGES);
        const unsigned short* a_tile = tile_a[step % STAGES];
        const unsigned short* b_tile = tile_b[step % STAGES];
#pragma unroll
        for (unsigned int kk = 0; kk < TILE_K; kk += 16)
        {
            unsigned int a[4][4];
#pragma unroll
            for (unsigned int mt = 0; mt < 4; ++mt)
            {
                const unsigned int row = warp_m + mt * 16 + lane % 16;
                load_matrix_x4(a_tile + swizzled(row, (kk + (lane / 16) * 8) / 8), a[mt]);
            }
            unsigned int b[4][4];
#pragma unroll
            for (unsigned int np = 0; np < 4; ++np)
            {
                const unsigned int row = warp_n + np * 16 + lane % 8 + (lane / 16) * 8;
                load_matrix_x4(b_tile + swizzled(row, (kk + ((lane / 8) % 2) * 8) / 8), b[np]);
            }
#pragma unroll
            for (unsigned int mt = 0; mt < 4; ++mt)
            {
#pragma unroll
                for (unsigned int np = 0; np < 4; ++np)
                {
                    mma<bf16>(a[mt], b[np][0], b[np][1], acc[mt][2 * np]);
                    mma<bf16>(a[mt], b[np][2], b[np][3], acc[mt][2 * np + 1]);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");

    const unsigned int g = lane / 4;
    const unsigned int t = lane % 4;
#pragma unroll
    for (unsigned int mt = 0; mt < 4; ++mt)
    {
#pragma unroll
        for (unsigned int nt = 0; nt < 8; ++nt)
        {
            const unsigned int col = block_n + warp_n + nt * 8 + 2 * t;
            if (col >= n)
            {
                continue;
            }
#pragma unroll
            for (unsigned int half = 0; half < 2; ++half)
            {
                const unsigned int row = block_m + warp_m + mt * 16 + g + half * 8;
                if (row < m)
                {
                    *reinterpret_cast<float2*>(out + static_cast<unsigned long long>(row) * n + col) =
                        make_float2(acc[mt][nt][2 * half], acc[mt][nt][2 * half + 1]);
                }
            }
        }
    }
}

}

extern "C" __global__ void __launch_bounds__(128) gemm_f16(const unsigned short* x, const unsigned short* w, float* out,
                                                            unsigned int m, unsigned int n, unsigned int k)
{
    gemm_tiles<false>(x, w, out, m, n, k);
}

extern "C" __global__ void __launch_bounds__(128) gemm_bf16(const unsigned short* x, const unsigned short* w,
                                                             float* out, unsigned int m, unsigned int n, unsigned int k)
{
    gemm_tiles<true>(x, w, out, m, n, k);
}
