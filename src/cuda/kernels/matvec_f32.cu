extern "C" __global__ void matvec_f32(const float* weight, const float* input, float* output,
                                     unsigned int n_in, unsigned int n_out, unsigned int n_tokens)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int row = blockIdx.x * 4 + threadIdx.x / 32;
    const unsigned int token = blockIdx.y * 4;
    if (row >= n_out)
    {
        return;
    }
    float sums[4] = {};
    for (unsigned int column = lane; column < n_in; column += 32)
    {
        const float value = weight[static_cast<unsigned long long>(row) * n_in + column];
        for (unsigned int slot = 0; slot < 4; ++slot)
        {
            if (token + slot < n_tokens)
            {
                sums[slot] = __fmaf_rn(value, input[static_cast<unsigned long long>(token + slot) * n_in + column], sums[slot]);
            }
        }
    }
    for (unsigned int slot = 0; slot < 4; ++slot)
    {
        for (unsigned int offset = 16; offset > 0; offset /= 2)
        {
            sums[slot] += __shfl_down_sync(0xffffffffu, sums[slot], offset);
        }
        if (lane == 0 && token + slot < n_tokens)
        {
            output[static_cast<unsigned long long>(token + slot) * n_out + row] = sums[slot];
        }
    }
}