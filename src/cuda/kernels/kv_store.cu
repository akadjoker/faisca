#include <cuda_fp16.h>

// half_cache: the caches hold f16 (rounded to nearest), otherwise float32.
extern "C" __global__ void kv_store(const float* keys, const float* values, void* key_cache, void* value_cache,
                                    unsigned int n_head_kv, unsigned int head_dim, unsigned int n_ctx,
                                    unsigned int position_value, const unsigned int* position_at, unsigned int layer,
                                    unsigned int half_cache)
{
    const unsigned int position = position_at != nullptr ? *position_at : position_value;
    for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x)
    {
        const unsigned long long source = (static_cast<unsigned long long>(blockIdx.x) * n_head_kv + blockIdx.y) * head_dim + column;
        const unsigned long long target = ((static_cast<unsigned long long>(layer) * n_head_kv + blockIdx.y) * n_ctx +
                                          position + blockIdx.x) * head_dim + column;
        if (half_cache != 0)
        {
            static_cast<__half*>(key_cache)[target] = __float2half_rn(keys[source]);
            static_cast<__half*>(value_cache)[target] = __float2half_rn(values[source]);
        }
        else
        {
            static_cast<float*>(key_cache)[target] = keys[source];
            static_cast<float*>(value_cache)[target] = values[source];
        }
    }
}
