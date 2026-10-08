#include "faisca/ops.h"

#include <cmath>

namespace faisca
{

void layer_norm(const float* x, const float* weight, const float* bias, size_t n, float eps, float* out)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += x[i];
    }
    const float mean = static_cast<float>(sum / static_cast<double>(n));
    double squares = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const float centered = x[i] - mean;
        out[i] = centered;
        squares += static_cast<double>(centered) * static_cast<double>(centered);
    }
    const float variance = static_cast<float>(squares / static_cast<double>(n));
    const float scale = 1.0f / sqrtf(variance + eps);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = out[i] * scale * weight[i] + bias[i];
    }
}

void gelu(float* x, size_t n)
{
    const float coef_a = 0.044715f;
    const float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
    for (size_t i = 0; i < n; ++i)
    {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(sqrt_2_over_pi * v * (1.0f + coef_a * v * v)));
    }
}

void add_bias(float* x, const float* bias, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        x[i] += bias[i];
    }
}

void conv1d_k3(const float* x, size_t n_in, size_t length, const float* w, const float* bias, size_t stride,
               size_t first, size_t last, float* out)
{
    const size_t out_length = length / stride;
    for (size_t oc = first; oc < last; ++oc)
    {
        float* o = out + oc * out_length;
        for (size_t t = 0; t < out_length; ++t)
        {
            o[t] = bias[oc];
        }
        for (size_t ic = 0; ic < n_in; ++ic)
        {
            const float* row = x + ic * length;
            const float w0 = w[(oc * n_in + ic) * 3 + 0];
            const float w1 = w[(oc * n_in + ic) * 3 + 1];
            const float w2 = w[(oc * n_in + ic) * 3 + 2];
            for (size_t t = 0; t < out_length; ++t)
            {
                const size_t center = t * stride;
                float sum = w1 * row[center];
                if (center > 0)
                {
                    sum += w0 * row[center - 1];
                }
                if (center + 1 < length)
                {
                    sum += w2 * row[center + 1];
                }
                o[t] += sum;
            }
        }
    }
}

void attention_rows(const float* q, const float* k, const float* v, size_t stride, size_t head_dim, size_t n_kv,
                    float scale, float* scores, float* out)
{
    for (size_t j = 0; j < n_kv; ++j)
    {
        scores[j] = dot(q, k + j * stride, head_dim) * scale;
    }
    softmax(scores, n_kv);
    for (size_t d = 0; d < head_dim; ++d)
    {
        out[d] = 0.0f;
    }
    for (size_t j = 0; j < n_kv; ++j)
    {
        const float weight = scores[j];
        const float* row = v + j * stride;
        for (size_t d = 0; d < head_dim; ++d)
        {
            out[d] += weight * row[d];
        }
    }
}

}
