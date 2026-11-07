#pragma once

__device__ __forceinline__ float warp_sum(float value)
{
    for (unsigned int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value)
{
    __shared__ float partial[32];
    value = warp_sum(value);
    if ((threadIdx.x & 31) == 0)
    {
        partial[threadIdx.x / 32] = value;
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        value = warp_sum(threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0.0f);
        if (threadIdx.x == 0)
        {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float block_max(float value)
{
    __shared__ float partial[32];
    for (unsigned int offset = 16; offset > 0; offset /= 2)
    {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    if ((threadIdx.x & 31) == 0)
    {
        partial[threadIdx.x / 32] = value;
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        value = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : -__int_as_float(0x7f800000);
        for (unsigned int offset = 16; offset > 0; offset /= 2)
        {
            value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
        }
        if (threadIdx.x == 0)
        {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}