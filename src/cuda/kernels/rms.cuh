#pragma once

#include "reduce.cuh"

// rms_norm of one row by the whole block; every thread of the block must call it.
__device__ __forceinline__ void rms_norm_row(const float* input, const float* weight, float* output,
                                             unsigned int width, float epsilon)
{
    float sum = 0;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x)
    {
        const float value = input[column];
        sum = __fmaf_rn(value, value, sum);
    }
    const float scale = rsqrtf(block_sum(sum) / static_cast<float>(width) + epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x)
    {
        output[column] = (input[column] * scale) * weight[column];
    }
}
