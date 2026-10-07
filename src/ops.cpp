#include "faisca/ops.h"

#include <cmath>
#include <cstring>

namespace faisca
{

float f16_to_f32(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
    const uint32_t exponent = (bits >> 10) & 0x1f;
    const uint32_t mantissa = bits & 0x3ff;

    uint32_t out = 0;
    if (exponent == 0)
    {
        // zero or subnormal: mantissa * 2^-24, exact in float
        float value = static_cast<float>(mantissa) * (1.0f / 16777216.0f);
        std::memcpy(&out, &value, 4);
        out |= sign;
    }
    else if (exponent == 31)
    {
        out = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        out = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &out, 4);
    return value;
}

float dot(const float* a, const float* b, size_t n)
{
    float sums[8] = {};
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            sums[j] += a[i + j] * b[i + j];
        }
    }
    for (; i < n; ++i)
    {
        sums[i & 7] += a[i] * b[i];
    }
    return ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
}

void matvec(const float* w, const float* x, size_t n_in, size_t n_out, float* out)
{
    for (size_t o = 0; o < n_out; ++o)
    {
        out[o] = dot(w + o * n_in, x, n_in);
    }
}

void rms_norm(const float* x, const float* weight, size_t n, float eps, float* out)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += static_cast<double>(x[i] * x[i]);
    }
    const float mean = static_cast<float>(sum / static_cast<double>(n));
    const float scale = 1.0f / std::sqrt(mean + eps);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = x[i] * scale * weight[i];
    }
}

void rope_norm(float* x, size_t n_head, size_t head_dim, size_t n_rot, uint32_t pos, float freq_base)
{
    const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));
    for (size_t h = 0; h < n_head; ++h)
    {
        float* head = x + h * head_dim;
        float theta = static_cast<float>(pos);
        for (size_t i = 0; i < n_rot; i += 2)
        {
            const float cos_theta = std::cos(theta);
            const float sin_theta = std::sin(theta);
            const float x0 = head[i];
            const float x1 = head[i + 1];
            head[i] = x0 * cos_theta - x1 * sin_theta;
            head[i + 1] = x0 * sin_theta + x1 * cos_theta;
            theta *= theta_scale;
        }
    }
}

void softmax(float* x, size_t n)
{
    if (n == 0)
    {
        return;
    }
    float max = x[0];
    for (size_t i = 1; i < n; ++i)
    {
        max = x[i] > max ? x[i] : max;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = std::exp(x[i] - max);
        sum += static_cast<double>(x[i]);
    }
    const float inverse = static_cast<float>(1.0 / sum);
    for (size_t i = 0; i < n; ++i)
    {
        x[i] *= inverse;
    }
}

void swiglu(const float* gate, const float* up, size_t n, float* out)
{
    for (size_t i = 0; i < n; ++i)
    {
        const float g = gate[i];
        out[i] = g / (1.0f + std::exp(-g)) * up[i];
    }
}

void add(float* x, const float* y, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        x[i] += y[i];
    }
}

}
