#include "reduce.cuh"

#include <cuda_fp16.h>

__device__ __forceinline__ float load_cache(const void* cache, unsigned long long index, unsigned int half_cache)
{
    return half_cache != 0 ? __half2float(static_cast<const __half*>(cache)[index]) : static_cast<const float*>(cache)[index];
}

// Four values from index, a multiple of 4.
__device__ __forceinline__ float4 load_cache4(const void* cache, unsigned long long index, unsigned int half_cache)
{
    if (half_cache != 0)
    {
        const uint2 packed = *reinterpret_cast<const uint2*>(static_cast<const __half*>(cache) + index);
        const float2 low = __half22float2(*reinterpret_cast<const __half2*>(&packed.x));
        const float2 high = __half22float2(*reinterpret_cast<const __half2*>(&packed.y));
        return make_float4(low.x, low.y, high.x, high.y);
    }
    return *reinterpret_cast<const float4*>(static_cast<const float*>(cache) + index);
}

extern "C" __global__ void attention(const float* query, const void* keys, const void* values,
                                     float* scratch, float* output, unsigned int n_head, unsigned int n_head_kv,
                                     unsigned int head_dim, unsigned int n_ctx, unsigned int position_value,
                                     const unsigned int* position_at, unsigned int layer, float scale,
                                     unsigned int half_cache)
{
    const unsigned int position = position_at != nullptr ? *position_at : position_value;
    const unsigned int head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int kv_head = head / (n_head / n_head_kv);
    const unsigned int length = position + token + 1;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned long long query_base = (static_cast<unsigned long long>(token) * n_head + head) * head_dim;
    const unsigned long long cache_base = (static_cast<unsigned long long>(layer) * n_head_kv + kv_head) * n_ctx * head_dim;
    float* scores = scratch + (static_cast<unsigned long long>(token) * n_head + head) * n_ctx;
    for (unsigned int past = warp; past < length; past += blockDim.x / 32)
    {
        float sum = 0;
        for (unsigned int column = lane; column < head_dim; column += 32)
        {
            sum = __fmaf_rn(query[query_base + column],
                             load_cache(keys, cache_base + static_cast<unsigned long long>(past) * head_dim + column, half_cache), sum);
        }
        sum = warp_sum(sum);
        if (lane == 0)
        {
            scores[past] = sum * scale;
        }
    }
    __syncthreads();
    float maximum = -__int_as_float(0x7f800000);
    for (unsigned int past = threadIdx.x; past < length; past += blockDim.x)
    {
        maximum = fmaxf(maximum, scores[past]);
    }
    maximum = block_max(maximum);
    float denominator = 0;
    for (unsigned int past = threadIdx.x; past < length; past += blockDim.x)
    {
        const float probability = expf(scores[past] - maximum);
        scores[past] = probability;
        denominator += probability;
    }
    denominator = block_sum(denominator);
    const float inverse = 1.0f / denominator;
    for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x)
    {
        float sum = 0;
        for (unsigned int past = 0; past < length; ++past)
        {
            sum = __fmaf_rn(scores[past] * inverse,
                            load_cache(values, cache_base + static_cast<unsigned long long>(past) * head_dim + column, half_cache), sum);
        }
        output[query_base + column] = sum;
    }
}
#define SPLIT 128

// Merges the used splits of one (token, head) into its output row, every thread of the block
// taking columns; the parts are read past L1 because other blocks wrote them.
__device__ __forceinline__ void combine_head(const float* parts, float* output, unsigned int head_dim,
                                             unsigned int used)
{
    float maximum = -__int_as_float(0x7f800000);
    for (unsigned int split = 0; split < used; ++split)
    {
        maximum = fmaxf(maximum, __ldcg(parts + split * (head_dim + 4) + head_dim));
    }
    float denominator = 0;
    for (unsigned int split = 0; split < used; ++split)
    {
        const float* part = parts + split * (head_dim + 4);
        denominator = __fmaf_rn(__ldcg(part + head_dim + 1), expf(__ldcg(part + head_dim) - maximum), denominator);
    }
    const float inverse = 1.0f / denominator;
    for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x)
    {
        float sum = 0;
        for (unsigned int split = 0; split < used; ++split)
        {
            const float* part = parts + split * (head_dim + 4);
            sum = __fmaf_rn(__ldcg(part + column), expf(__ldcg(part + head_dim) - maximum), sum);
        }
        output[column] = sum * inverse;
    }
}

// Flash-decoding: block (head, split, token) scores up to SPLIT cached positions, keeps the
// unnormalised value sum with its maximum and denominator in scratch, attention_combine merges
// the splits. Needs head_dim % 16 == 0, head_dim <= 256. HEAD_DIM, when not 0, fixes head_dim
// at compile time so the loads below unroll and go out together; the sums keep the same order.
template <unsigned int HEAD_DIM>
__device__ __forceinline__ void attention_split_body(const float* query, const void* keys, const void* values,
                                                     float* scratch, unsigned int n_head, unsigned int n_head_kv,
                                                     unsigned int head_dim_value, unsigned int n_ctx,
                                                     unsigned int position_value, const unsigned int* position_at,
                                                     unsigned int layer, float scale, unsigned int splits,
                                                     unsigned int half_cache, unsigned int* counters, float* output)
{
    constexpr unsigned int BATCH = 16; // loads in flight per thread
    const unsigned int head_dim = HEAD_DIM != 0 ? HEAD_DIM : head_dim_value;
    const unsigned int position = position_at != nullptr ? *position_at : position_value;
    __shared__ float4 shared_query[64];
    __shared__ float probabilities[SPLIT];
    __shared__ float4 partial_sums[SPLIT];
    const unsigned int head = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int token = blockIdx.z;
    const unsigned int length = position + token + 1;
    const unsigned int start = split * SPLIT;
    if (start >= length)
    {
        return;
    }
    const unsigned int end = min(start + SPLIT, length);
    const unsigned int kv_head = head / (n_head / n_head_kv);
    const unsigned int groups = head_dim / 4;
    const unsigned long long query_base = (static_cast<unsigned long long>(token) * n_head + head) * head_dim;
    const unsigned long long cache_base = (static_cast<unsigned long long>(layer) * n_head_kv + kv_head) * n_ctx * head_dim;
    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x)
    {
        shared_query[group] = reinterpret_cast<const float4*>(query + query_base)[group];
    }
    __syncthreads();

    const unsigned int past = start + threadIdx.x;
    float score = -__int_as_float(0x7f800000);
    if (past < end)
    {
        const unsigned long long key = cache_base + static_cast<unsigned long long>(past) * head_dim;
        float sum = 0;
        for (unsigned int first = 0; first < groups; first += BATCH)
        {
            float4 k[BATCH];
#pragma unroll
            for (unsigned int step = 0; step < BATCH; ++step)
            {
                if (first + step < groups)
                {
                    k[step] = load_cache4(keys, key + (first + step) * 4, half_cache);
                }
            }
#pragma unroll
            for (unsigned int step = 0; step < BATCH; ++step)
            {
                if (first + step < groups)
                {
                    const float4 q = shared_query[first + step];
                    sum = __fmaf_rn(q.x, k[step].x, sum);
                    sum = __fmaf_rn(q.y, k[step].y, sum);
                    sum = __fmaf_rn(q.z, k[step].z, sum);
                    sum = __fmaf_rn(q.w, k[step].w, sum);
                }
            }
        }
        score = sum * scale;
    }
    const float maximum = block_max(score);
    const float probability = past < end ? expf(score - maximum) : 0.0f;
    probabilities[threadIdx.x] = probability;
    const float denominator = block_sum(probability);

    const unsigned int group = threadIdx.x % groups;
    const unsigned int lane = threadIdx.x / groups;
    const unsigned int lanes = SPLIT / groups;
    const unsigned int count = SPLIT / lanes; // positions per thread
    float4 sum = make_float4(0, 0, 0, 0);
    if (lane < lanes)
    {
        for (unsigned int first = 0; first < count; first += BATCH)
        {
            float4 v[BATCH];
#pragma unroll
            for (unsigned int step = 0; step < BATCH; ++step)
            {
                const unsigned int index = start + lane + (first + step) * lanes;
                if (first + step < count && index < end)
                {
                    v[step] = load_cache4(values, cache_base + static_cast<unsigned long long>(index) * head_dim + group * 4,
                                          half_cache);
                }
            }
#pragma unroll
            for (unsigned int step = 0; step < BATCH; ++step)
            {
                const unsigned int index = start + lane + (first + step) * lanes;
                if (first + step < count && index < end)
                {
                    const float p = probabilities[index - start];
                    sum.x = __fmaf_rn(p, v[step].x, sum.x);
                    sum.y = __fmaf_rn(p, v[step].y, sum.y);
                    sum.z = __fmaf_rn(p, v[step].z, sum.z);
                    sum.w = __fmaf_rn(p, v[step].w, sum.w);
                }
            }
        }
    }
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    float* out = scratch + ((static_cast<unsigned long long>(token) * n_head + head) * splits + split) * (head_dim + 4);
    if (threadIdx.x < groups)
    {
        float4 total = partial_sums[threadIdx.x];
        for (unsigned int other = 1; other < lanes; ++other)
        {
            const float4 v = partial_sums[other * groups + threadIdx.x];
            total.x += v.x;
            total.y += v.y;
            total.z += v.z;
            total.w += v.w;
        }
        reinterpret_cast<float4*>(out)[threadIdx.x] = total;
    }
    if (threadIdx.x == 0)
    {
        out[head_dim] = maximum;
        out[head_dim + 1] = denominator;
    }
    if (counters == nullptr)
    {
        return;
    }
    // the last split block to finish merges them (attention_combine without its launch)
    __shared__ bool last;
    const unsigned int used = (length + SPLIT - 1) / SPLIT;
    const unsigned long long row = static_cast<unsigned long long>(token) * n_head + head;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        last = atomicAdd(counters + row, 1u) == used - 1;
    }
    __syncthreads();
    if (!last)
    {
        return;
    }
    __threadfence();
    if (threadIdx.x == 0)
    {
        counters[row] = 0;
    }
    combine_head(scratch + row * splits * (head_dim + 4), output + row * head_dim, head_dim, used);
}

#define FAISCA_ATTENTION_SPLIT(name, dim) \
    extern "C" __global__ void name(const float* query, const void* keys, const void* values, float* scratch, \
                                    unsigned int n_head, unsigned int n_head_kv, unsigned int head_dim, \
                                    unsigned int n_ctx, unsigned int position_value, \
                                    const unsigned int* position_at, unsigned int layer, float scale, \
                                    unsigned int splits, unsigned int half_cache, unsigned int* counters, \
                                    float* output) \
    { \
        attention_split_body<dim>(query, keys, values, scratch, n_head, n_head_kv, head_dim, n_ctx, position_value, \
                                  position_at, layer, scale, splits, half_cache, counters, output); \
    }

FAISCA_ATTENTION_SPLIT(attention_split, 0)
FAISCA_ATTENTION_SPLIT(attention_split_64, 64)
FAISCA_ATTENTION_SPLIT(attention_split_128, 128)

extern "C" __global__ void attention_combine(const float* scratch, float* output, unsigned int n_head,
                                             unsigned int head_dim, unsigned int position_value, const unsigned int* position_at,
                                             unsigned int splits)
{
    const unsigned int position = position_at != nullptr ? *position_at : position_value;
    const unsigned int head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int used = (position + token + SPLIT) / SPLIT;
    const unsigned long long row = static_cast<unsigned long long>(token) * n_head + head;
    combine_head(scratch + row * splits * (head_dim + 4), output + row * head_dim, head_dim, used);
}

// Prompt attention, flash style: block (head, 64 queries), TPQ = HEAD_DIM / 16 threads per query
// holding 16 dimensions each; tiles of KT cached positions in shared memory, scores of a tile
// first, then one rescale of the running sum per tile (online softmax), causal mask.
template <unsigned int HEAD_DIM>
__device__ __forceinline__ void attention_prefill_body(const float* query, const void* keys, const void* values,
                                                       float* output, unsigned int n_head, unsigned int n_head_kv,
                                                       unsigned int n_ctx, unsigned int position, unsigned int layer,
                                                       float scale, unsigned int n_tokens, unsigned int half_cache)
{
    constexpr unsigned int TPQ = HEAD_DIM / 16;
    constexpr unsigned int KT = 4096 / HEAD_DIM;
    __shared__ float4 shared_keys[KT][HEAD_DIM / 4];
    __shared__ float4 shared_values[KT][HEAD_DIM / 4];
    const unsigned int head = blockIdx.x;
    const unsigned int kv_head = head / (n_head / n_head_kv);
    const unsigned int first_query = blockIdx.y * 64;
    const unsigned int row = first_query + threadIdx.x / TPQ;
    const unsigned int part = threadIdx.x % TPQ;
    const bool valid = row < n_tokens;
    const unsigned int query_position = position + row;
    const unsigned int last_key = position + min(first_query + 63, n_tokens - 1);
    const unsigned long long cache_base = (static_cast<unsigned long long>(layer) * n_head_kv + kv_head) * n_ctx * HEAD_DIM;

    float4 q[4];
    const float4* query_row =
        reinterpret_cast<const float4*>(query + (static_cast<unsigned long long>(valid ? row : 0) * n_head + head) * HEAD_DIM);
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i)
    {
        q[i] = query_row[part * 4 + i];
    }
    float4 acc[4] = {};
    float maximum = -__int_as_float(0x7f800000);
    float denominator = 0;
    for (unsigned int tile = 0; tile <= last_key; tile += KT)
    {
        __syncthreads();
        for (unsigned int index = threadIdx.x; index < KT * HEAD_DIM / 4; index += blockDim.x)
        {
            const unsigned int key = index / (HEAD_DIM / 4);
            const unsigned int group = index % (HEAD_DIM / 4);
            const unsigned long long at = cache_base + static_cast<unsigned long long>(tile + key) * HEAD_DIM + group * 4;
            const bool inside = tile + key <= last_key;
            shared_keys[key][group] = inside ? load_cache4(keys, at, half_cache) : make_float4(0, 0, 0, 0);
            shared_values[key][group] = inside ? load_cache4(values, at, half_cache) : make_float4(0, 0, 0, 0);
        }
        __syncthreads();
        float scores[KT];
        float tile_maximum = -__int_as_float(0x7f800000);
#pragma unroll
        for (unsigned int key = 0; key < KT; ++key)
        {
            float dot = 0;
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
            {
                const float4 k = shared_keys[key][part * 4 + i];
                dot = __fmaf_rn(q[i].x, k.x, dot);
                dot = __fmaf_rn(q[i].y, k.y, dot);
                dot = __fmaf_rn(q[i].z, k.z, dot);
                dot = __fmaf_rn(q[i].w, k.w, dot);
            }
#pragma unroll
            for (unsigned int offset = 1; offset < TPQ; offset *= 2)
            {
                dot += __shfl_xor_sync(0xffffffffu, dot, offset);
            }
            scores[key] = valid && tile + key <= query_position ? dot * scale : -__int_as_float(0x7f800000);
            tile_maximum = fmaxf(tile_maximum, scores[key]);
        }
        const float next = fmaxf(maximum, tile_maximum);
        if (next == -__int_as_float(0x7f800000))
        {
            continue; // nothing visible yet (or an invalid query)
        }
        const float correction = expf(maximum - next);
        denominator *= correction;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i)
        {
            acc[i].x *= correction;
            acc[i].y *= correction;
            acc[i].z *= correction;
            acc[i].w *= correction;
        }
#pragma unroll
        for (unsigned int key = 0; key < KT; ++key)
        {
            const float p = expf(scores[key] - next);
            denominator += p;
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
            {
                const float4 v = shared_values[key][part * 4 + i];
                acc[i].x = __fmaf_rn(p, v.x, acc[i].x);
                acc[i].y = __fmaf_rn(p, v.y, acc[i].y);
                acc[i].z = __fmaf_rn(p, v.z, acc[i].z);
                acc[i].w = __fmaf_rn(p, v.w, acc[i].w);
            }
        }
        maximum = next;
    }
    if (!valid)
    {
        return;
    }
    const float inverse = 1.0f / denominator;
    float4* out = reinterpret_cast<float4*>(output + (static_cast<unsigned long long>(row) * n_head + head) * HEAD_DIM);
#pragma unroll
    for (unsigned int i = 0; i < 4; ++i)
    {
        out[part * 4 + i] = make_float4(acc[i].x * inverse, acc[i].y * inverse, acc[i].z * inverse, acc[i].w * inverse);
    }
}

#define FAISCA_ATTENTION_PREFILL(name, dim) \
    extern "C" __global__ void name(const float* query, const void* keys, const void* values, float* output, \
                                    unsigned int n_head, unsigned int n_head_kv, unsigned int n_ctx, \
                                    unsigned int position, unsigned int layer, float scale, unsigned int n_tokens, \
                                    unsigned int half_cache) \
    { \
        attention_prefill_body<dim>(query, keys, values, output, n_head, n_head_kv, n_ctx, position, layer, scale, \
                                    n_tokens, half_cache); \
    }

FAISCA_ATTENTION_PREFILL(attention_prefill_64, 64)
FAISCA_ATTENTION_PREFILL(attention_prefill_128, 128)

// Prompt attention on the TF32 tensor cores with float32 accuracy: every operand x is split
// into big = tf32(x) and small = tf32(x - big) and each product summed as small*big + big*small
// + big*big (3xTF32, Ootomo and Yokota 2022, "Recovering single precision accuracy from Tensor
// Cores while surpassing the FP32 theoretical peak performance"). FlashAttention-2 structure:
// block (head, 64 queries), warp: 16 queries; K/V tiles of 64 keys in shared memory (rows of 68
// floats against bank conflicts); S = Q K^T and O += P V with mma.sync m16n8k8 tf32, online
// softmax per tile. Fragments (PTX ISA, m16n8k8 .tf32): g = lane / 4, t = lane % 4; A (g, t),
// (g + 8, t), (g, t + 4), (g + 8, t + 4); B (k t, n g), (k t + 4, n g); C (g, 2t), (g, 2t + 1),
// (g + 8, 2t), (g + 8, 2t + 1). For P V the keys of each group of 8 are taken in the order
// 0, 2, 4, 6, 1, 3, 5, 7, so P's C fragment is already the A fragment.
__device__ __forceinline__ unsigned int to_tf32(float x)
{
    unsigned int r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void split_tf32(float x, unsigned int& big, unsigned int& small)
{
    big = to_tf32(x);
    small = to_tf32(x - __uint_as_float(big));
}

__device__ __forceinline__ void mma_tf32(const unsigned int a[4], unsigned int b0, unsigned int b1, float c[4])
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// c += a * b with a and b already split
__device__ __forceinline__ void mma_3xtf32(const unsigned int a_big[4], const unsigned int a_small[4], unsigned int b0_big,
                                           unsigned int b1_big, unsigned int b0_small, unsigned int b1_small, float c[4])
{
    mma_tf32(a_small, b0_big, b1_big, c);
    mma_tf32(a_big, b0_small, b1_small, c);
    mma_tf32(a_big, b0_big, b1_big, c);
}

extern "C" __global__ void attention_prefill_tc_64(const float* query, const void* keys, const void* values,
                                                  float* output, unsigned int n_head, unsigned int n_head_kv,
                                                  unsigned int n_ctx, unsigned int position, unsigned int layer,
                                                  float scale, unsigned int n_tokens, unsigned int half_cache)
{
    constexpr unsigned int HEAD_DIM = 64;
    constexpr unsigned int KT = 64;
    constexpr unsigned int STRIDE = 68;
    const float negative_infinity = -__int_as_float(0x7f800000);
    __shared__ __align__(16) float shared_keys[KT * STRIDE];
    __shared__ __align__(16) float shared_values[KT * STRIDE];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int g = lane / 4;
    const unsigned int t = lane % 4;
    const unsigned int head = blockIdx.x;
    const unsigned int kv_head = head / (n_head / n_head_kv);
    const unsigned int first_query = blockIdx.y * 64;
    const unsigned int rows[2] = {first_query + warp * 16 + g, first_query + warp * 16 + g + 8};
    const bool valid[2] = {rows[0] < n_tokens, rows[1] < n_tokens};
    const unsigned int last_key = position + min(first_query + 63, n_tokens - 1);
    const unsigned long long cache_base = (static_cast<unsigned long long>(layer) * n_head_kv + kv_head) * n_ctx * HEAD_DIM;

    unsigned int q_big[8][4];
    unsigned int q_small[8][4];
#pragma unroll
    for (unsigned int step = 0; step < 8; ++step)
    {
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i)
        {
            const unsigned int r = i & 1;
            const unsigned int column = step * 8 + t + (i >= 2 ? 4 : 0);
            const float value =
                valid[r] ? query[(static_cast<unsigned long long>(rows[r]) * n_head + head) * HEAD_DIM + column] : 0.0f;
            split_tf32(value, q_big[step][i], q_small[step][i]);
        }
    }
    float o[8][4] = {};
    float maximum[2] = {negative_infinity, negative_infinity};
    float sum[2] = {0, 0};
    for (unsigned int tile = 0; tile <= last_key; tile += KT)
    {
        __syncthreads();
        for (unsigned int index = threadIdx.x; index < KT * HEAD_DIM / 4; index += blockDim.x)
        {
            const unsigned int key = index / (HEAD_DIM / 4);
            const unsigned int group = index % (HEAD_DIM / 4);
            const unsigned long long at = cache_base + static_cast<unsigned long long>(tile + key) * HEAD_DIM + group * 4;
            const bool inside = tile + key <= last_key;
            *reinterpret_cast<float4*>(shared_keys + key * STRIDE + group * 4) =
                inside ? load_cache4(keys, at, half_cache) : make_float4(0, 0, 0, 0);
            *reinterpret_cast<float4*>(shared_values + key * STRIDE + group * 4) =
                inside ? load_cache4(values, at, half_cache) : make_float4(0, 0, 0, 0);
        }
        __syncthreads();

        float s[8][4] = {};
#pragma unroll
        for (unsigned int n = 0; n < 8; ++n)
        {
#pragma unroll
            for (unsigned int step = 0; step < 8; ++step)
            {
                const float* k = shared_keys + (n * 8 + g) * STRIDE + step * 8 + t;
                unsigned int b0_big, b0_small, b1_big, b1_small;
                split_tf32(k[0], b0_big, b0_small);
                split_tf32(k[4], b1_big, b1_small);
                mma_3xtf32(q_big[step], q_small[step], b0_big, b1_big, b0_small, b1_small, s[n]);
            }
        }
        float tile_maximum[2] = {negative_infinity, negative_infinity};
#pragma unroll
        for (unsigned int n = 0; n < 8; ++n)
        {
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
            {
                const unsigned int r = i >= 2 ? 1 : 0;
                const unsigned int key = tile + n * 8 + 2 * t + (i & 1);
                s[n][i] = valid[r] && key <= position + rows[r] ? s[n][i] * scale : negative_infinity;
                tile_maximum[r] = fmaxf(tile_maximum[r], s[n][i]);
            }
        }
        float base[2];
        float correction[2];
#pragma unroll
        for (unsigned int r = 0; r < 2; ++r)
        {
            tile_maximum[r] = fmaxf(tile_maximum[r], __shfl_xor_sync(0xffffffffu, tile_maximum[r], 1));
            tile_maximum[r] = fmaxf(tile_maximum[r], __shfl_xor_sync(0xffffffffu, tile_maximum[r], 2));
            const float next = fmaxf(maximum[r], tile_maximum[r]);
            base[r] = next == negative_infinity ? 0.0f : next;
            correction[r] = expf(maximum[r] - base[r]);
            maximum[r] = next;
            sum[r] *= correction[r];
        }
#pragma unroll
        for (unsigned int n = 0; n < 8; ++n)
        {
            o[n][0] *= correction[0];
            o[n][1] *= correction[0];
            o[n][2] *= correction[1];
            o[n][3] *= correction[1];
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
            {
                const unsigned int r = i >= 2 ? 1 : 0;
                s[n][i] = expf(s[n][i] - base[r]);
                sum[r] += s[n][i];
            }
        }
#pragma unroll
        for (unsigned int j = 0; j < 8; ++j)
        {
            unsigned int p_big[4];
            unsigned int p_small[4];
            split_tf32(s[j][0], p_big[0], p_small[0]);
            split_tf32(s[j][2], p_big[1], p_small[1]);
            split_tf32(s[j][1], p_big[2], p_small[2]);
            split_tf32(s[j][3], p_big[3], p_small[3]);
#pragma unroll
            for (unsigned int n = 0; n < 8; ++n)
            {
                const float* v = shared_values + (j * 8 + 2 * t) * STRIDE + n * 8 + g;
                unsigned int b0_big, b0_small, b1_big, b1_small;
                split_tf32(v[0], b0_big, b0_small);
                split_tf32(v[STRIDE], b1_big, b1_small);
                mma_3xtf32(p_big, p_small, b0_big, b1_big, b0_small, b1_small, o[n]);
            }
        }
    }
#pragma unroll
    for (unsigned int r = 0; r < 2; ++r)
    {
        sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 1);
        sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 2);
    }
#pragma unroll
    for (unsigned int n = 0; n < 8; ++n)
    {
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i)
        {
            const unsigned int r = i >= 2 ? 1 : 0;
            if (valid[r])
            {
                const unsigned int column = n * 8 + 2 * t + (i & 1);
                output[(static_cast<unsigned long long>(rows[r]) * n_head + head) * HEAD_DIM + column] = o[n][i] / sum[r];
            }
        }
    }
}
