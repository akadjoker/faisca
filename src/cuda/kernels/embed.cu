#include <cuda_fp16.h>

extern "C" __global__ void embed(const unsigned char* weight, const __half* scales, const unsigned int* tokens,
                                 float* output, unsigned int n_embd, unsigned int q8)
{
    const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= n_embd)
    {
        return;
    }
    const unsigned long long index = static_cast<unsigned long long>(tokens[blockIdx.y]) * n_embd + column;
    float value;
    if (q8 != 0)
    {
        value = __half2float(scales[index / 32]) * static_cast<float>(static_cast<signed char>(weight[index]));
    }
    else
    {
        value = reinterpret_cast<const float*>(weight)[index];
    }
    output[static_cast<unsigned long long>(blockIdx.y) * n_embd + column] = value;
}
