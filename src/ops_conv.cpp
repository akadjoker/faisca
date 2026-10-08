#include "faisca/ops.h"

#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace faisca
{

namespace
{

// columns [t0, t1) of rows o in [first, first + rows), every tap in range
template <int ROWS>
void conv_columns_scalar(const float* x, size_t x_stride, size_t ic, const float* w, size_t k, size_t dilation, size_t pad,
                         const float* bias, size_t first, size_t t0, size_t t1, float* out, size_t out_stride)
{
    for (int r = 0; r < ROWS; ++r)
    {
        const size_t o = first + static_cast<size_t>(r);
        float* row = out + o * out_stride;
        const float b = bias != nullptr ? bias[o] : 0.0f;
        for (size_t t = t0; t < t1; ++t)
        {
            float acc = b;
            for (size_t i = 0; i < ic; ++i)
            {
                const float* xi = x + i * x_stride + t - pad;
                const float* wi = w + (o * ic + i) * k;
                for (size_t j = 0; j < k; ++j)
                {
                    acc += wi[j] * xi[j * dilation];
                }
            }
            row[t] = acc;
        }
    }
}

// the columns t < pad, where taps fall before the start of the row and count as zero
template <int ROWS>
void conv_columns_edge(const float* x, size_t x_stride, size_t ic, const float* w, size_t k, size_t dilation, size_t pad,
                       const float* bias, size_t first, size_t t1, float* out, size_t out_stride)
{
    for (int r = 0; r < ROWS; ++r)
    {
        const size_t o = first + static_cast<size_t>(r);
        float* row = out + o * out_stride;
        const float b = bias != nullptr ? bias[o] : 0.0f;
        for (size_t t = 0; t < t1; ++t)
        {
            float acc = b;
            for (size_t i = 0; i < ic; ++i)
            {
                const float* xi = x + i * x_stride;
                const float* wi = w + (o * ic + i) * k;
                for (size_t j = 0; j < k; ++j)
                {
                    const size_t src = t + j * dilation;
                    if (src >= pad)
                    {
                        acc += wi[j] * xi[src - pad];
                    }
                }
            }
            row[t] = acc;
        }
    }
}

#if defined(__AVX512F__)

constexpr size_t CONV_COLUMNS = 64;

template <int ROWS>
void conv_tiles(const float* x, size_t x_stride, size_t ic, const float* w, size_t k, size_t dilation, size_t pad,
                const float* bias, size_t first, size_t t0, size_t t1, float* out, size_t out_stride)
{
    for (size_t t = t0; t + CONV_COLUMNS <= t1; t += CONV_COLUMNS)
    {
        __m512 acc[ROWS][4];
        for (int r = 0; r < ROWS; ++r)
        {
            const __m512 b = _mm512_set1_ps(bias != nullptr ? bias[first + static_cast<size_t>(r)] : 0.0f);
            acc[r][0] = b;
            acc[r][1] = b;
            acc[r][2] = b;
            acc[r][3] = b;
        }
        for (size_t i = 0; i < ic; ++i)
        {
            const float* xi = x + i * x_stride + t - pad;
            const float* wi = w + (first * ic + i) * k;
            for (size_t j = 0; j < k; ++j)
            {
                const float* xs = xi + j * dilation;
                const __m512 x0 = _mm512_loadu_ps(xs);
                const __m512 x1 = _mm512_loadu_ps(xs + 16);
                const __m512 x2 = _mm512_loadu_ps(xs + 32);
                const __m512 x3 = _mm512_loadu_ps(xs + 48);
                for (int r = 0; r < ROWS; ++r)
                {
                    const __m512 wv = _mm512_set1_ps(wi[static_cast<size_t>(r) * ic * k + j]);
                    acc[r][0] = _mm512_fmadd_ps(wv, x0, acc[r][0]);
                    acc[r][1] = _mm512_fmadd_ps(wv, x1, acc[r][1]);
                    acc[r][2] = _mm512_fmadd_ps(wv, x2, acc[r][2]);
                    acc[r][3] = _mm512_fmadd_ps(wv, x3, acc[r][3]);
                }
            }
        }
        for (int r = 0; r < ROWS; ++r)
        {
            float* row = out + (first + static_cast<size_t>(r)) * out_stride + t;
            _mm512_storeu_ps(row, acc[r][0]);
            _mm512_storeu_ps(row + 16, acc[r][1]);
            _mm512_storeu_ps(row + 32, acc[r][2]);
            _mm512_storeu_ps(row + 48, acc[r][3]);
        }
    }
}

#elif defined(__AVX2__) && defined(__FMA__)

constexpr size_t CONV_COLUMNS = 32;

template <int ROWS>
void conv_tiles(const float* x, size_t x_stride, size_t ic, const float* w, size_t k, size_t dilation, size_t pad,
                const float* bias, size_t first, size_t t0, size_t t1, float* out, size_t out_stride)
{
    for (size_t t = t0; t + CONV_COLUMNS <= t1; t += CONV_COLUMNS)
    {
        __m256 acc[ROWS][4];
        for (int r = 0; r < ROWS; ++r)
        {
            const __m256 b = _mm256_set1_ps(bias != nullptr ? bias[first + static_cast<size_t>(r)] : 0.0f);
            acc[r][0] = b;
            acc[r][1] = b;
            acc[r][2] = b;
            acc[r][3] = b;
        }
        for (size_t i = 0; i < ic; ++i)
        {
            const float* xi = x + i * x_stride + t - pad;
            const float* wi = w + (first * ic + i) * k;
            for (size_t j = 0; j < k; ++j)
            {
                const float* xs = xi + j * dilation;
                const __m256 x0 = _mm256_loadu_ps(xs);
                const __m256 x1 = _mm256_loadu_ps(xs + 8);
                const __m256 x2 = _mm256_loadu_ps(xs + 16);
                const __m256 x3 = _mm256_loadu_ps(xs + 24);
                for (int r = 0; r < ROWS; ++r)
                {
                    const __m256 wv = _mm256_set1_ps(wi[static_cast<size_t>(r) * ic * k + j]);
                    acc[r][0] = _mm256_fmadd_ps(wv, x0, acc[r][0]);
                    acc[r][1] = _mm256_fmadd_ps(wv, x1, acc[r][1]);
                    acc[r][2] = _mm256_fmadd_ps(wv, x2, acc[r][2]);
                    acc[r][3] = _mm256_fmadd_ps(wv, x3, acc[r][3]);
                }
            }
        }
        for (int r = 0; r < ROWS; ++r)
        {
            float* row = out + (first + static_cast<size_t>(r)) * out_stride + t;
            _mm256_storeu_ps(row, acc[r][0]);
            _mm256_storeu_ps(row + 8, acc[r][1]);
            _mm256_storeu_ps(row + 16, acc[r][2]);
            _mm256_storeu_ps(row + 24, acc[r][3]);
        }
    }
}

#else

constexpr size_t CONV_COLUMNS = 16;

template <int ROWS>
void conv_tiles(const float* x, size_t x_stride, size_t ic, const float* w, size_t k, size_t dilation, size_t pad,
                const float* bias, size_t first, size_t t0, size_t t1, float* out, size_t out_stride)
{
    const size_t end = t0 + (t1 - t0) / CONV_COLUMNS * CONV_COLUMNS;
    conv_columns_scalar<ROWS>(x, x_stride, ic, w, k, dilation, pad, bias, first, t0, end, out, out_stride);
}

#endif

template <int ROWS>
void conv_rows(const float* x, size_t x_stride, size_t length, size_t ic, const float* w, size_t k, size_t dilation,
               size_t pad, const float* bias, size_t first, size_t t_first, size_t t_last, float* out, size_t out_stride)
{
    const size_t edge = pad < length ? pad : length;
    if (t_first < edge)
    {
        conv_columns_edge<ROWS>(x, x_stride, ic, w, k, dilation, pad, bias, first, edge < t_last ? edge : t_last, out, out_stride);
    }
    const size_t t0 = t_first > edge ? t_first : edge;
    if (t0 >= t_last)
    {
        return;
    }
    const size_t tiles_end = t0 + (t_last - t0) / CONV_COLUMNS * CONV_COLUMNS;
    conv_tiles<ROWS>(x, x_stride, ic, w, k, dilation, pad, bias, first, t0, tiles_end, out, out_stride);
    conv_columns_scalar<ROWS>(x, x_stride, ic, w, k, dilation, pad, bias, first, tiles_end, t_last, out, out_stride);
}

}

void conv1d_rows(const float* x, size_t x_stride, size_t length, size_t ic, const float* w, size_t k, size_t dilation,
                 size_t pad, const float* bias, size_t first, size_t last, size_t t_first, size_t t_last, float* out,
                 size_t out_stride)
{
    size_t o = first;
    for (; o + 4 <= last; o += 4)
    {
        conv_rows<4>(x, x_stride, length, ic, w, k, dilation, pad, bias, o, t_first, t_last, out, out_stride);
    }
    switch (last - o)
    {
    case 3:
        conv_rows<3>(x, x_stride, length, ic, w, k, dilation, pad, bias, o, t_first, t_last, out, out_stride);
        break;
    case 2:
        conv_rows<2>(x, x_stride, length, ic, w, k, dilation, pad, bias, o, t_first, t_last, out, out_stride);
        break;
    case 1:
        conv_rows<1>(x, x_stride, length, ic, w, k, dilation, pad, bias, o, t_first, t_last, out, out_stride);
        break;
    default:
        break;
    }
}

}
