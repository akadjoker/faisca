extern "C" __global__ void swiglu(const float* gate, const float* up, float* output, unsigned int count)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        const float value = gate[index];
        output[index] = (value / (1.0f + expf(-value))) * up[index];
    }
}