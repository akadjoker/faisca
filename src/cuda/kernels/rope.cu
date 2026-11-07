#include <cuda_fp16.h>

// Rotates the first n_rot values of one head in place (pairs of adjacent values); when
// key_cache is not null the rotated pairs and the values past n_rot are also stored there.
__device__ __forceinline__ void rotate_head(float* head, const float* factors, unsigned int has_factors,
                                            unsigned int head_dim, unsigned int n_rot, unsigned int position,
                                            float theta_scale, void* key_cache, unsigned int half_cache)
{
    for (unsigned int pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x)
    {
        float theta = static_cast<float>(position);
        for (unsigned int step = 0; step < pair; ++step)
        {
            theta *= theta_scale;
        }
        if (has_factors != 0)
        {
            theta /= factors[pair];
        }
        const float cosine = cosf(theta);
        const float sine = sinf(theta);
        const float first = head[pair * 2];
        const float second = head[pair * 2 + 1];
        const float rotated_first = first * cosine - second * sine;
        const float rotated_second = first * sine + second * cosine;
        head[pair * 2] = rotated_first;
        head[pair * 2 + 1] = rotated_second;
        if (key_cache != nullptr)
        {
            if (half_cache != 0)
            {
                static_cast<__half*>(key_cache)[pair * 2] = __float2half_rn(rotated_first);
                static_cast<__half*>(key_cache)[pair * 2 + 1] = __float2half_rn(rotated_second);
            }
            else
            {
                static_cast<float*>(key_cache)[pair * 2] = rotated_first;
                static_cast<float*>(key_cache)[pair * 2 + 1] = rotated_second;
            }
        }
    }
}

__device__ __forceinline__ void store_values(const float* source, void* cache, unsigned int first,
                                             unsigned int count, unsigned int half_cache)
{
    for (unsigned int column = first + threadIdx.x; column < count; column += blockDim.x)
    {
        if (half_cache != 0)
        {
            static_cast<__half*>(cache)[column] = __float2half_rn(source[column]);
        }
        else
        {
            static_cast<float*>(cache)[column] = source[column];
        }
    }
}

extern "C" __global__ void rope(float* input, const float* factors, unsigned int has_factors,
                                unsigned int n_head, unsigned int head_dim, unsigned int n_rot,
                                unsigned int position_value, const unsigned int* position_at, float theta_scale)
{
    const unsigned int position = position_at != nullptr ? *position_at : position_value;
    const unsigned long long base = (static_cast<unsigned long long>(blockIdx.x) * n_head + blockIdx.y) * head_dim;
    rotate_head(input + base, factors, has_factors, head_dim, n_rot, position + blockIdx.x, theta_scale, nullptr, 0);
}

// rope on the query and key heads and kv_store in one launch: block (token, head) with the
// n_head query heads first, then the n_head_kv key heads, whose keys and values go to the caches.
extern "C" __global__ void rope_store(float* query, float* keys, const float* values, void* key_cache,
                                      void* value_cache, const float* factors, unsigned int has_factors,
                                      unsigned int n_head, unsigned int n_head_kv, unsigned int head_dim,
                                      unsigned int n_rot, unsigned int n_ctx, unsigned int position_value,
                                      const unsigned int* position_at, unsigned int layer, float theta_scale,
                                      unsigned int half_cache)
{
    const unsigned int position = (position_at != nullptr ? *position_at : position_value) + blockIdx.x;
    if (blockIdx.y < n_head)
    {
        float* head = query + (static_cast<unsigned long long>(blockIdx.x) * n_head + blockIdx.y) * head_dim;
        rotate_head(head, factors, has_factors, head_dim, n_rot, position, theta_scale, nullptr, 0);
        return;
    }
    const unsigned int kv_head = blockIdx.y - n_head;
    const unsigned long long source = (static_cast<unsigned long long>(blockIdx.x) * n_head_kv + kv_head) * head_dim;
    const unsigned long long target = ((static_cast<unsigned long long>(layer) * n_head_kv + kv_head) * n_ctx + position) * head_dim;
    const unsigned int element = half_cache != 0 ? 2 : 4;
    void* key_target = static_cast<char*>(key_cache) + target * element;
    void* value_target = static_cast<char*>(value_cache) + target * element;
    rotate_head(keys + source, factors, has_factors, head_dim, n_rot, position, theta_scale, key_target, half_cache);
    store_values(keys + source, key_target, n_rot, head_dim, half_cache);
    store_values(values + source, value_target, 0, head_dim, half_cache);
}
