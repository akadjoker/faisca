#include "faisca/ops.h"

#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FAISCA_GEMM_AVX2 1
#endif

namespace faisca
{

void pack_f32_panels(const float* w, size_t n_in, size_t n_out, float* out)
{
    for (size_t p = 0; p < n_out / F32_PANEL; ++p)
    {
        float* panel = out + p * n_in * F32_PANEL;
        for (size_t r = 0; r < F32_PANEL; ++r)
        {
            const float* row = w + (p * F32_PANEL + r) * n_in;
            for (size_t k = 0; k < n_in; ++k)
            {
                panel[k * F32_PANEL + r] = row[k];
            }
        }
    }
}

void gemm_f32_packed_scalar(const float* panels, size_t n_in, const float* x, size_t x_stride, size_t n_tokens,
                            size_t first, size_t last, float* out, size_t out_stride)
{
    for (size_t t = 0; t < n_tokens; ++t)
    {
        const float* xt = x + t * x_stride;
        for (size_t o = first; o < last; ++o)
        {
            const float* panel = panels + (o / F32_PANEL) * n_in * F32_PANEL + o % F32_PANEL;
            float acc = 0.0f;
            for (size_t k = 0; k < n_in; ++k)
            {
                acc = std::fma(xt[k], panel[k * F32_PANEL], acc);
            }
            out[t * out_stride + o] = acc;
        }
    }
}

#if defined(FAISCA_GEMM_AVX2)

namespace
{

constexpr size_t ROWS = 6;

// rows tokens of a, packed [k][ROWS], times one panel of 16 outputs; named accumulators,
// an array of them makes GCC store all twelve to the stack on every k
template <int rows>
void gemm_panel(const float* panel, size_t n_in, const float* a, float* out, size_t out_stride)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    for (size_t k = 0; k < n_in; ++k)
    {
        const __m256 w0 = _mm256_loadu_ps(panel + k * F32_PANEL);
        const __m256 w1 = _mm256_loadu_ps(panel + k * F32_PANEL + 8);
        const float* ak = a + k * ROWS;
        __m256 x = _mm256_broadcast_ss(ak + 0);
        c00 = _mm256_fmadd_ps(x, w0, c00);
        c01 = _mm256_fmadd_ps(x, w1, c01);
        if constexpr (rows > 1)
        {
            x = _mm256_broadcast_ss(ak + 1);
            c10 = _mm256_fmadd_ps(x, w0, c10);
            c11 = _mm256_fmadd_ps(x, w1, c11);
        }
        if constexpr (rows > 2)
        {
            x = _mm256_broadcast_ss(ak + 2);
            c20 = _mm256_fmadd_ps(x, w0, c20);
            c21 = _mm256_fmadd_ps(x, w1, c21);
        }
        if constexpr (rows > 3)
        {
            x = _mm256_broadcast_ss(ak + 3);
            c30 = _mm256_fmadd_ps(x, w0, c30);
            c31 = _mm256_fmadd_ps(x, w1, c31);
        }
        if constexpr (rows > 4)
        {
            x = _mm256_broadcast_ss(ak + 4);
            c40 = _mm256_fmadd_ps(x, w0, c40);
            c41 = _mm256_fmadd_ps(x, w1, c41);
        }
        if constexpr (rows > 5)
        {
            x = _mm256_broadcast_ss(ak + 5);
            c50 = _mm256_fmadd_ps(x, w0, c50);
            c51 = _mm256_fmadd_ps(x, w1, c51);
        }
    }
    const __m256 c[ROWS][2] = {{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}, {c40, c41}, {c50, c51}};
    for (int t = 0; t < rows; ++t)
    {
        _mm256_storeu_ps(out + size_t(t) * out_stride, c[t][0]);
        _mm256_storeu_ps(out + size_t(t) * out_stride + 8, c[t][1]);
    }
}

}

void gemm_f32_packed(const float* panels, size_t n_in, const float* x, size_t x_stride, size_t n_tokens, size_t first,
                     size_t last, float* out, size_t out_stride, float* scratch)
{
    for (size_t t = 0; t < n_tokens; t += ROWS)
    {
        const size_t rows = n_tokens - t < ROWS ? n_tokens - t : ROWS;
        const float* xt = x + t * x_stride;
        for (size_t k = 0; k < n_in; ++k)
        {
            for (size_t r = 0; r < rows; ++r)
            {
                scratch[k * ROWS + r] = xt[r * x_stride + k];
            }
        }
        float* ot = out + t * out_stride;
        for (size_t o = first; o < last; o += F32_PANEL)
        {
            const float* panel = panels + (o / F32_PANEL) * n_in * F32_PANEL;
            switch (rows)
            {
            case 6:
                gemm_panel<6>(panel, n_in, scratch, ot + o, out_stride);
                break;
            case 5:
                gemm_panel<5>(panel, n_in, scratch, ot + o, out_stride);
                break;
            case 4:
                gemm_panel<4>(panel, n_in, scratch, ot + o, out_stride);
                break;
            case 3:
                gemm_panel<3>(panel, n_in, scratch, ot + o, out_stride);
                break;
            case 2:
                gemm_panel<2>(panel, n_in, scratch, ot + o, out_stride);
                break;
            default:
                gemm_panel<1>(panel, n_in, scratch, ot + o, out_stride);
                break;
            }
        }
    }
}

#else

void gemm_f32_packed(const float* panels, size_t n_in, const float* x, size_t x_stride, size_t n_tokens, size_t first,
                     size_t last, float* out, size_t out_stride, float*)
{
    gemm_f32_packed_scalar(panels, n_in, x, x_stride, n_tokens, first, last, out, out_stride);
}

#endif

}
