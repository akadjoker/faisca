#include "rms.cuh"

extern "C" __global__ void rms_norm(const float* input, const float* weight, float* output,
                                    unsigned int width, float epsilon)
{
    const unsigned long long base = static_cast<unsigned long long>(blockIdx.x) * width;
    rms_norm_row(input + base, weight, output + base, width, epsilon);
}
