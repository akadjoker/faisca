#include "faisca/ops.h"

#include <cmath>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define FAISCA_AVX2 1
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512BW__)
#define FAISCA_AVX512_VNNI 1
#elif defined(__AVXVNNI__)
#define FAISCA_AVX_VNNI 1
#endif
#endif

namespace faisca
{

const char* simd_name()
{
#if defined(FAISCA_AVX512_VNNI)
    return "AVX-512 VNNI";
#elif defined(FAISCA_AVX_VNNI)
    return "AVX2+FMA+F16C+VNNI";
#elif defined(FAISCA_AVX2)
    return "AVX2+FMA+F16C";
#else
    return "scalar";
#endif
}

bool q8_0_packed_supported()
{
#if defined(FAISCA_AVX512_VNNI) || defined(FAISCA_AVX_VNNI)
    return true;
#else
    return false;
#endif
}

size_t q8_0_padded_rows(size_t n_out)
{
    return (n_out + Q8_0_GROUP - 1) / Q8_0_GROUP * Q8_0_GROUP;
}

size_t q8_0_packed_bytes(size_t n_in, size_t n_out)
{
    return q8_0_padded_rows(n_out) / Q8_0_GROUP * (n_in / Q8_0_BLOCK) * Q8_0_GROUP_BLOCK_BYTES;
}

void pack_q8_0(const uint8_t* rows, size_t n_in, size_t n_out, uint8_t* out)
{
    const size_t blocks = n_in / Q8_0_BLOCK;
    const size_t row_bytes = blocks * Q8_0_BLOCK_BYTES;
    const size_t groups = q8_0_padded_rows(n_out) / Q8_0_GROUP;

    for (size_t g = 0; g < groups; ++g)
    {
        for (size_t b = 0; b < blocks; ++b)
        {
            uint8_t* scales = out;
            uint8_t* quants = out + 2 * Q8_0_GROUP;
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                const size_t row = g * Q8_0_GROUP + r;
                if (row >= n_out)
                {
                    scales[2 * r] = 0;
                    scales[2 * r + 1] = 0;
                    for (size_t k = 0; k < Q8_0_BLOCK; ++k)
                    {
                        quants[(k / 4) * 64 + r * 4 + (k % 4)] = 128;
                    }
                    continue;
                }
                const uint8_t* block = rows + row * row_bytes + b * Q8_0_BLOCK_BYTES;
                scales[2 * r] = block[0];
                scales[2 * r + 1] = block[1];
                for (size_t k = 0; k < Q8_0_BLOCK; ++k)
                {
                    quants[(k / 4) * 64 + r * 4 + (k % 4)] = static_cast<uint8_t>(block[2 + k] + 128);
                }
            }
            out += Q8_0_GROUP_BLOCK_BYTES;
        }
    }
}

void row_q8_0_packed(const uint8_t* packed, size_t n_in, size_t row, float* out)
{
    const size_t blocks = n_in / Q8_0_BLOCK;
    const size_t g = row / Q8_0_GROUP;
    const size_t r = row % Q8_0_GROUP;
    const uint8_t* group = packed + g * blocks * Q8_0_GROUP_BLOCK_BYTES;
    for (size_t b = 0; b < blocks; ++b)
    {
        const uint8_t* block = group + b * Q8_0_GROUP_BLOCK_BYTES;
        uint16_t bits = 0;
        std::memcpy(&bits, block + 2 * r, 2);
        const float d = f16_to_f32(bits);
        const uint8_t* quants = block + 2 * Q8_0_GROUP;
        for (size_t k = 0; k < Q8_0_BLOCK; ++k)
        {
            const int value = static_cast<int>(quants[(k / 4) * 64 + r * 4 + (k % 4)]) - 128;
            out[b * Q8_0_BLOCK + k] = static_cast<float>(value) * d;
        }
    }
}

void q8_0_block_sums(const uint8_t* xq, size_t n, int32_t* out)
{
    for (size_t b = 0; b < n / Q8_0_BLOCK; ++b)
    {
        const int8_t* q = reinterpret_cast<const int8_t*>(xq + b * Q8_0_BLOCK_BYTES + 2);
        int32_t sum = 0;
        for (size_t k = 0; k < Q8_0_BLOCK; ++k)
        {
            sum += q[k];
        }
        out[b] = 128 * sum;
    }
}

void matvec_q8_0_packed_scalar(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in,
                               size_t n_out, size_t first, size_t last, float* out)
{
    const size_t blocks = n_in / Q8_0_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* group = packed + g * blocks * Q8_0_GROUP_BLOCK_BYTES;
        float acc[Q8_0_GROUP] = {};
        for (size_t b = 0; b < blocks; ++b)
        {
            const uint8_t* block = group + b * Q8_0_GROUP_BLOCK_BYTES;
            const uint8_t* quants = block + 2 * Q8_0_GROUP;
            const int8_t* x = reinterpret_cast<const int8_t*>(xq + b * Q8_0_BLOCK_BYTES + 2);
            uint16_t dx_bits = 0;
            std::memcpy(&dx_bits, xq + b * Q8_0_BLOCK_BYTES, 2);
            const float dx = f16_to_f32(dx_bits);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                int32_t sum = 0;
                for (size_t k = 0; k < Q8_0_BLOCK; ++k)
                {
                    sum += static_cast<int32_t>(quants[(k / 4) * 64 + r * 4 + (k % 4)]) * x[k];
                }
                sum -= x_sums[b];
                uint16_t bits = 0;
                std::memcpy(&bits, block + 2 * r, 2);
                acc[r] += static_cast<float>(sum) * (f16_to_f32(bits) * dx);
            }
        }
        for (size_t r = 0; r < Q8_0_GROUP && g * Q8_0_GROUP + r < last; ++r)
        {
            out[g * Q8_0_GROUP + r] = acc[r];
        }
    }
    (void)n_out;
}

#if defined(FAISCA_AVX512_VNNI)

namespace
{

constexpr size_t GEMM_TOKENS = 16;

// one group of 16 rows against up to 16 inputs, the 8 chunks of every block loaded once
template <size_t TOKENS>
void gemm_group_avx512(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                       size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    __m512 acc[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc[t] = _mm512_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        const uint8_t* quants = block + 2 * Q8_0_GROUP;
        __m512i w[8];
        for (size_t j = 0; j < 8; ++j)
        {
            w[j] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quants + 64 * j));
        }
        const __m512 scales = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block)));
        for (size_t t = 0; t < TOKENS; ++t)
        {
            const uint8_t* x = xq + t * xq_stride + b * Q8_0_BLOCK_BYTES;
            __m512i sum = _mm512_setzero_si512();
            for (size_t j = 0; j < 8; ++j)
            {
                int32_t x4 = 0;
                std::memcpy(&x4, x + 2 + 4 * j, 4);
                sum = _mm512_dpbusd_epi32(sum, w[j], _mm512_set1_epi32(x4));
            }
            sum = _mm512_sub_epi32(sum, _mm512_set1_epi32(x_sums[t * sums_stride + b]));
            uint16_t dx_bits = 0;
            std::memcpy(&dx_bits, x, 2);
            const __m512 d = _mm512_mul_ps(scales, _mm512_set1_ps(f16_to_f32(dx_bits)));
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), d, acc[t]);
        }
        block += Q8_0_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        float* o = out + t * out_stride + row;
        if (row + Q8_0_GROUP <= last)
        {
            _mm512_storeu_ps(o, acc[t]);
        }
        else
        {
            _mm512_mask_storeu_ps(o, static_cast<__mmask16>((1u << (last - row)) - 1), acc[t]);
        }
    }
}

}

void matmul_q8_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    const size_t blocks = n_in / Q8_0_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* block = packed + g * blocks * Q8_0_GROUP_BLOCK_BYTES;
        const size_t row = g * Q8_0_GROUP;
        size_t t = 0;
        for (; t + GEMM_TOKENS <= n_tokens; t += GEMM_TOKENS)
        {
            gemm_group_avx512<GEMM_TOKENS>(block, xq + t * xq_stride, xq_stride, x_sums + t * sums_stride, sums_stride,
                                           blocks, row, last, out + t * out_stride, out_stride);
        }
        for (; t < n_tokens; ++t)
        {
            gemm_group_avx512<1>(block, xq + t * xq_stride, xq_stride, x_sums + t * sums_stride, sums_stride, blocks,
                                 row, last, out + t * out_stride, out_stride);
        }
    }
    (void)n_out;
}

void matvec_q8_0_packed(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in, size_t n_out,
                        size_t first, size_t last, float* out)
{
    const size_t blocks = n_in / Q8_0_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* block = packed + g * blocks * Q8_0_GROUP_BLOCK_BYTES;
        const uint8_t* x = xq;
        __m512 acc = _mm512_setzero_ps();
        for (size_t b = 0; b < blocks; ++b)
        {
            const uint8_t* quants = block + 2 * Q8_0_GROUP;
            __m512i sum = _mm512_setzero_si512();
            for (size_t j = 0; j < 8; ++j)
            {
                int32_t x4 = 0;
                std::memcpy(&x4, x + 2 + 4 * j, 4);
                sum = _mm512_dpbusd_epi32(sum, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quants + 64 * j)),
                                          _mm512_set1_epi32(x4));
            }
            sum = _mm512_sub_epi32(sum, _mm512_set1_epi32(x_sums[b]));

            uint16_t dx_bits = 0;
            std::memcpy(&dx_bits, x, 2);
            const __m512 scales = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block)));
            const __m512 d = _mm512_mul_ps(scales, _mm512_set1_ps(f16_to_f32(dx_bits)));
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), d, acc);

            block += Q8_0_GROUP_BLOCK_BYTES;
            x += Q8_0_BLOCK_BYTES;
        }
        const size_t row = g * Q8_0_GROUP;
        if (row + Q8_0_GROUP <= last)
        {
            _mm512_storeu_ps(out + row, acc);
        }
        else
        {
            _mm512_mask_storeu_ps(out + row, static_cast<__mmask16>((1u << (last - row)) - 1), acc);
        }
    }
    (void)n_out;
}

#elif defined(FAISCA_AVX_VNNI)

void matmul_q8_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    for (size_t t = 0; t < n_tokens; ++t)
    {
        matvec_q8_0_packed(packed, xq + t * xq_stride, x_sums + t * sums_stride, n_in, n_out, first, last,
                           out + t * out_stride);
    }
}

void matvec_q8_0_packed(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in, size_t n_out,
                        size_t first, size_t last, float* out)
{
    const size_t blocks = n_in / Q8_0_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* block = packed + g * blocks * Q8_0_GROUP_BLOCK_BYTES;
        const uint8_t* x = xq;
        __m256 acc_lo = _mm256_setzero_ps();
        __m256 acc_hi = _mm256_setzero_ps();
        for (size_t b = 0; b < blocks; ++b)
        {
            const uint8_t* quants = block + 2 * Q8_0_GROUP;
            __m256i sum_lo = _mm256_setzero_si256();
            __m256i sum_hi = _mm256_setzero_si256();
            for (size_t j = 0; j < 8; ++j)
            {
                int32_t x4 = 0;
                std::memcpy(&x4, x + 2 + 4 * j, 4);
                const __m256i xb = _mm256_set1_epi32(x4);
                sum_lo = _mm256_dpbusd_avx_epi32(sum_lo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * j)), xb);
                sum_hi = _mm256_dpbusd_avx_epi32(sum_hi, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * j + 32)), xb);
            }
            const __m256i correction = _mm256_set1_epi32(x_sums[b]);
            sum_lo = _mm256_sub_epi32(sum_lo, correction);
            sum_hi = _mm256_sub_epi32(sum_hi, correction);

            uint16_t dx_bits = 0;
            std::memcpy(&dx_bits, x, 2);
            const __m256 dx = _mm256_set1_ps(f16_to_f32(dx_bits));
            const __m256 d_lo = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block))), dx);
            const __m256 d_hi = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 16))), dx);
            acc_lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum_lo), d_lo, acc_lo);
            acc_hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum_hi), d_hi, acc_hi);

            block += Q8_0_GROUP_BLOCK_BYTES;
            x += Q8_0_BLOCK_BYTES;
        }
        const size_t row = g * Q8_0_GROUP;
        float values[Q8_0_GROUP];
        _mm256_storeu_ps(values, acc_lo);
        _mm256_storeu_ps(values + 8, acc_hi);
        const size_t count = row + Q8_0_GROUP <= last ? Q8_0_GROUP : last - row;
        std::memcpy(out + row, values, count * sizeof(float));
    }
    (void)n_out;
}

#else

void matmul_q8_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    for (size_t t = 0; t < n_tokens; ++t)
    {
        matvec_q8_0_packed_scalar(packed, xq + t * xq_stride, x_sums + t * sums_stride, n_in, n_out, first, last,
                                  out + t * out_stride);
    }
}

void matvec_q8_0_packed(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in, size_t n_out,
                        size_t first, size_t last, float* out)
{
    matvec_q8_0_packed_scalar(packed, xq, x_sums, n_in, n_out, first, last, out);
}

#endif

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

uint16_t f32_to_f16(float value)
{
    uint32_t w = 0;
    std::memcpy(&w, &value, 4);
    const uint32_t sign = w & 0x80000000u;
    const uint32_t magnitude = w & 0x7fffffffu;

    if (magnitude >= 0x7f800000u)
    {
        // infinity keeps its exponent, NaN keeps a payload bit
        const uint32_t mantissa = magnitude & 0x7fffffu;
        return static_cast<uint16_t>((sign >> 16) | 0x7c00u | (mantissa != 0 ? 0x200u : 0));
    }
    if (magnitude >= 0x477ff000u)
    {
        // rounds to or beyond the largest half
        return static_cast<uint16_t>((sign >> 16) | 0x7c00u);
    }
    if (magnitude < 0x33000000u)
    {
        // below half the smallest subnormal
        return static_cast<uint16_t>(sign >> 16);
    }

    uint32_t exponent = magnitude >> 23;
    uint32_t mantissa = magnitude & 0x7fffffu;
    uint32_t shift = 13;
    if (exponent < 113)
    {
        // subnormal half: the hidden bit joins the mantissa and more bits are dropped
        mantissa |= 0x800000u;
        shift += 113 - exponent;
        exponent = 0;
    }
    else
    {
        exponent -= 112;
    }

    const uint32_t dropped = mantissa & ((1u << shift) - 1);
    const uint32_t half_way = 1u << (shift - 1);
    uint32_t result = (exponent << 10) | (mantissa >> shift);
    if (dropped > half_way || (dropped == half_way && (result & 1u) != 0))
    {
        ++result; // a carry moves into the exponent on its own
    }
    return static_cast<uint16_t>((sign >> 16) | result);
}

void quantize_q8_0(const float* x, size_t n, uint8_t* out)
{
    for (size_t i = 0; i < n; i += Q8_0_BLOCK)
    {
        float amax = 0.0f;
        for (size_t j = 0; j < Q8_0_BLOCK; ++j)
        {
            const float a = std::fabs(x[i + j]);
            amax = a > amax ? a : amax;
        }
        const float d = amax / 127.0f;
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;

        const uint16_t bits = f32_to_f16(d);
        std::memcpy(out, &bits, 2);
        int8_t* q = reinterpret_cast<int8_t*>(out + 2);
        for (size_t j = 0; j < Q8_0_BLOCK; ++j)
        {
            q[j] = static_cast<int8_t>(std::nearbyint(x[i + j] * id));
        }
        out += Q8_0_BLOCK_BYTES;
    }
}

void dequantize_q8_0(const uint8_t* blocks, size_t n, float* out)
{
    for (size_t i = 0; i < n; i += Q8_0_BLOCK)
    {
        uint16_t bits = 0;
        std::memcpy(&bits, blocks, 2);
        const float d = f16_to_f32(bits);
        const int8_t* q = reinterpret_cast<const int8_t*>(blocks + 2);
        for (size_t j = 0; j < Q8_0_BLOCK; ++j)
        {
            out[i + j] = static_cast<float>(q[j]) * d;
        }
        blocks += Q8_0_BLOCK_BYTES;
    }
}

float dot_q8_0_scalar(const uint8_t* a, const uint8_t* b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i += Q8_0_BLOCK)
    {
        uint16_t da = 0;
        uint16_t db = 0;
        std::memcpy(&da, a, 2);
        std::memcpy(&db, b, 2);
        const int8_t* qa = reinterpret_cast<const int8_t*>(a + 2);
        const int8_t* qb = reinterpret_cast<const int8_t*>(b + 2);
        int32_t sumi = 0;
        for (size_t j = 0; j < Q8_0_BLOCK; ++j)
        {
            sumi += static_cast<int32_t>(qa[j]) * qb[j];
        }
        sum += static_cast<float>(sumi) * (f16_to_f32(da) * f16_to_f32(db));
        a += Q8_0_BLOCK_BYTES;
        b += Q8_0_BLOCK_BYTES;
    }
    return sum;
}

#if defined(FAISCA_AVX2)

namespace
{

// the 8 lanes each hold the sum of 4 products, as ggml's mul_sum_i8_pairs_float
inline __m256 sum_i8_products(__m256i x, __m256i y)
{
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
#if defined(__AVXVNNI__)
    return _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy));
#else
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, dot));
#endif
}

inline float scale_of(const uint8_t* block)
{
    uint16_t bits = 0;
    std::memcpy(&bits, block, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

inline float horizontal_sum(__m256 x)
{
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// product of the two f16 block scales, converted with F16C
inline __m256 block_scale(const uint8_t* a, const uint8_t* b)
{
    uint16_t bits_a = 0;
    uint16_t bits_b = 0;
    std::memcpy(&bits_a, a, 2);
    std::memcpy(&bits_b, b, 2);
    const __m128 scales = _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(bits_a) | (static_cast<int>(bits_b) << 16)));
    const __m128 product = _mm_mul_ss(scales, _mm_shuffle_ps(scales, scales, 1));
    return _mm256_broadcastss_ps(product);
}

}

float dot_q8_0(const uint8_t* a, const uint8_t* b, size_t n)
{
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i < n; i += Q8_0_BLOCK)
    {
        const __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 2));
        const __m256i qb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 2));
        acc = _mm256_fmadd_ps(block_scale(a, b), sum_i8_products(qa, qb), acc);
        a += Q8_0_BLOCK_BYTES;
        b += Q8_0_BLOCK_BYTES;
    }
    return horizontal_sum(acc);
}

#else

float dot_q8_0(const uint8_t* a, const uint8_t* b, size_t n)
{
    return dot_q8_0_scalar(a, b, n);
}

#endif

#if defined(FAISCA_AVX2)

// four rows share every load of the vector; each row accumulates exactly as dot_q8_0 does
void matvec_q8_0(const uint8_t* w, const uint8_t* xq, size_t n_in, size_t n_out, float* out)
{
    const size_t blocks = n_in / Q8_0_BLOCK;
    const size_t stride = blocks * Q8_0_BLOCK_BYTES;

    float x_scales[8192 / Q8_0_BLOCK];
    if (blocks > sizeof(x_scales) / sizeof(x_scales[0]))
    {
        for (size_t o = 0; o < n_out; ++o)
        {
            out[o] = dot_q8_0(w + o * stride, xq, n_in);
        }
        return;
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        x_scales[b] = scale_of(xq + b * Q8_0_BLOCK_BYTES);
    }

    size_t o = 0;
    for (; o + 4 <= n_out; o += 4)
    {
        const uint8_t* w0 = w + o * stride;
        const uint8_t* w1 = w0 + stride;
        const uint8_t* w2 = w1 + stride;
        const uint8_t* w3 = w2 + stride;
        const uint8_t* x = xq;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (size_t b = 0; b < blocks; ++b)
        {
            const __m256i qx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + 2));
            const float dx = x_scales[b];
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(scale_of(w0) * dx),
                                   sum_i8_products(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w0 + 2)), qx), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(scale_of(w1) * dx),
                                   sum_i8_products(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w1 + 2)), qx), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(scale_of(w2) * dx),
                                   sum_i8_products(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w2 + 2)), qx), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(scale_of(w3) * dx),
                                   sum_i8_products(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w3 + 2)), qx), acc3);
            x += Q8_0_BLOCK_BYTES;
            w0 += Q8_0_BLOCK_BYTES;
            w1 += Q8_0_BLOCK_BYTES;
            w2 += Q8_0_BLOCK_BYTES;
            w3 += Q8_0_BLOCK_BYTES;
        }
        out[o] = horizontal_sum(acc0);
        out[o + 1] = horizontal_sum(acc1);
        out[o + 2] = horizontal_sum(acc2);
        out[o + 3] = horizontal_sum(acc3);
    }
    for (; o < n_out; ++o)
    {
        out[o] = dot_q8_0(w + o * stride, xq, n_in);
    }
}

#else

void matvec_q8_0(const uint8_t* w, const uint8_t* xq, size_t n_in, size_t n_out, float* out)
{
    const size_t stride = n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
    for (size_t o = 0; o < n_out; ++o)
    {
        out[o] = dot_q8_0(w + o * stride, xq, n_in);
    }
}

#endif

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
