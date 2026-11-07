extern "C" __global__ void add(float* input, const float* residual, unsigned int count)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        input[index] += residual[index];
    }
}