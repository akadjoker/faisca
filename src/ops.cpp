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
        out[b] = sum;
    }
}

void dequantize_q4_0(const uint8_t* blocks, size_t n, float* out)
{
    for (size_t i = 0; i < n; i += Q4_0_BLOCK)
    {
        uint16_t bits = 0;
        std::memcpy(&bits, blocks, 2);
        const float d = f16_to_f32(bits);
        for (size_t j = 0; j < Q4_0_BLOCK / 2; ++j)
        {
            out[i + j] = static_cast<float>(static_cast<int>(blocks[2 + j] & 0x0f) - 8) * d;
            out[i + j + Q4_0_BLOCK / 2] = static_cast<float>(static_cast<int>(blocks[2 + j] >> 4) - 8) * d;
        }
        blocks += Q4_0_BLOCK_BYTES;
    }
}

size_t q4_0_packed_bytes(size_t n_in, size_t n_out)
{
    return q8_0_padded_rows(n_out) / Q8_0_GROUP * (n_in / Q4_0_BLOCK) * Q4_0_GROUP_BLOCK_BYTES;
}

namespace
{

// nibble k of a Q4_0 block: low nibbles hold values 0 to 15, high nibbles 16 to 31
inline uint8_t q4_0_value(const uint8_t* block, size_t k)
{
    const uint8_t byte = block[2 + (k % 16)];
    return k < 16 ? static_cast<uint8_t>(byte & 0x0f) : static_cast<uint8_t>(byte >> 4);
}

// where value k of row r lives in a packed Q4_0 group block: byte index and whether it is the high nibble
inline size_t q4_0_packed_byte(size_t r, size_t k, bool& high)
{
    const size_t chunk = k / 8;
    const size_t within = k % 8;
    high = within >= 4;
    return 2 * Q8_0_GROUP + chunk * 64 + r * 4 + (within % 4);
}

}

void pack_q4_0(const uint8_t* rows, size_t n_in, size_t n_out, uint8_t* out)
{
    const size_t blocks = n_in / Q4_0_BLOCK;
    const size_t row_bytes = blocks * Q4_0_BLOCK_BYTES;
    const size_t groups = q8_0_padded_rows(n_out) / Q8_0_GROUP;

    for (size_t g = 0; g < groups; ++g)
    {
        for (size_t b = 0; b < blocks; ++b)
        {
            std::memset(out, 0, Q4_0_GROUP_BLOCK_BYTES);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                const size_t row = g * Q8_0_GROUP + r;
                if (row >= n_out)
                {
                    // zero scale and the value 8, which is 0 after the offset
                    for (size_t k = 0; k < Q4_0_BLOCK; ++k)
                    {
                        bool high = false;
                        const size_t at = q4_0_packed_byte(r, k, high);
                        out[at] = static_cast<uint8_t>(out[at] | (high ? 0x80 : 0x08));
                    }
                    continue;
                }
                const uint8_t* block = rows + row * row_bytes + b * Q4_0_BLOCK_BYTES;
                out[2 * r] = block[0];
                out[2 * r + 1] = block[1];
                for (size_t k = 0; k < Q4_0_BLOCK; ++k)
                {
                    bool high = false;
                    const size_t at = q4_0_packed_byte(r, k, high);
                    const uint8_t value = q4_0_value(block, k);
                    out[at] = static_cast<uint8_t>(out[at] | (high ? value << 4 : value));
                }
            }
            out += Q4_0_GROUP_BLOCK_BYTES;
        }
    }
}

void row_q4_0_packed(const uint8_t* packed, size_t n_in, size_t row, float* out)
{
    const size_t blocks = n_in / Q4_0_BLOCK;
    const size_t g = row / Q8_0_GROUP;
    const size_t r = row % Q8_0_GROUP;
    const uint8_t* group = packed + g * blocks * Q4_0_GROUP_BLOCK_BYTES;
    for (size_t b = 0; b < blocks; ++b)
    {
        const uint8_t* block = group + b * Q4_0_GROUP_BLOCK_BYTES;
        uint16_t bits = 0;
        std::memcpy(&bits, block + 2 * r, 2);
        const float d = f16_to_f32(bits);
        for (size_t k = 0; k < Q4_0_BLOCK; ++k)
        {
            bool high = false;
            const size_t at = q4_0_packed_byte(r, k, high);
            const int value = high ? block[at] >> 4 : block[at] & 0x0f;
            out[b * Q4_0_BLOCK + k] = static_cast<float>(value - 8) * d;
        }
    }
}

void matvec_q4_0_packed_scalar(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in,
                               size_t n_out, size_t first, size_t last, float* out)
{
    const size_t blocks = n_in / Q4_0_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* group = packed + g * blocks * Q4_0_GROUP_BLOCK_BYTES;
        float acc[Q8_0_GROUP] = {};
        for (size_t b = 0; b < blocks; ++b)
        {
            const uint8_t* block = group + b * Q4_0_GROUP_BLOCK_BYTES;
            const int8_t* x = reinterpret_cast<const int8_t*>(xq + b * Q8_0_BLOCK_BYTES + 2);
            uint16_t dx_bits = 0;
            std::memcpy(&dx_bits, xq + b * Q8_0_BLOCK_BYTES, 2);
            const float dx = f16_to_f32(dx_bits);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                int32_t sum = 0;
                for (size_t k = 0; k < Q4_0_BLOCK; ++k)
                {
                    bool high = false;
                    const size_t at = q4_0_packed_byte(r, k, high);
                    const int value = high ? block[at] >> 4 : block[at] & 0x0f;
                    sum += value * x[k];
                }
                sum -= 8 * x_sums[b];
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
                sum -= 128 * x_sums[b];
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
            sum = _mm512_sub_epi32(sum, _mm512_set1_epi32(128 * x_sums[t * sums_stride + b]));
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
            sum = _mm512_sub_epi32(sum, _mm512_set1_epi32(128 * x_sums[b]));

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

namespace
{

template <size_t TOKENS>
void gemm_group_q4_avx512(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                          size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    const __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512 acc[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc[t] = _mm512_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        const uint8_t* quants = block + 2 * Q8_0_GROUP;
        __m512i w[8];
        for (size_t c = 0; c < 4; ++c)
        {
            const __m512i bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quants + 64 * c));
            w[2 * c] = _mm512_and_si512(bytes, low_mask);
            w[2 * c + 1] = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), low_mask);
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
            sum = _mm512_sub_epi32(sum, _mm512_set1_epi32(8 * x_sums[t * sums_stride + b]));
            uint16_t dx_bits = 0;
            std::memcpy(&dx_bits, x, 2);
            const __m512 d = _mm512_mul_ps(scales, _mm512_set1_ps(f16_to_f32(dx_bits)));
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), d, acc[t]);
        }
        block += Q4_0_GROUP_BLOCK_BYTES;
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

void matmul_q4_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    const size_t blocks = n_in / Q4_0_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* block = packed + g * blocks * Q4_0_GROUP_BLOCK_BYTES;
        const size_t row = g * Q8_0_GROUP;
        size_t t = 0;
        for (; t + GEMM_TOKENS <= n_tokens; t += GEMM_TOKENS)
        {
            gemm_group_q4_avx512<GEMM_TOKENS>(block, xq + t * xq_stride, xq_stride, x_sums + t * sums_stride,
                                              sums_stride, blocks, row, last, out + t * out_stride, out_stride);
        }
        for (; t < n_tokens; ++t)
        {
            gemm_group_q4_avx512<1>(block, xq + t * xq_stride, xq_stride, x_sums + t * sums_stride, sums_stride,
                                    blocks, row, last, out + t * out_stride, out_stride);
        }
    }
    (void)n_out;
}

#elif defined(FAISCA_AVX_VNNI)

void matmul_q4_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    const size_t blocks = n_in / Q4_0_BLOCK;
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    for (size_t t = 0; t < n_tokens; ++t)
    {
        const uint8_t* xt = xq + t * xq_stride;
        for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
        {
            const uint8_t* block = packed + g * blocks * Q4_0_GROUP_BLOCK_BYTES;
            __m256 acc_lo = _mm256_setzero_ps();
            __m256 acc_hi = _mm256_setzero_ps();
            for (size_t b = 0; b < blocks; ++b)
            {
                const uint8_t* quants = block + 2 * Q8_0_GROUP;
                const uint8_t* x = xt + b * Q8_0_BLOCK_BYTES;
                __m256i sum_lo = _mm256_setzero_si256();
                __m256i sum_hi = _mm256_setzero_si256();
                for (size_t c = 0; c < 4; ++c)
                {
                    const __m256i bytes_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * c));
                    const __m256i bytes_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * c + 32));
                    int32_t x4 = 0;
                    int32_t x5 = 0;
                    std::memcpy(&x4, x + 2 + 8 * c, 4);
                    std::memcpy(&x5, x + 6 + 8 * c, 4);
                    sum_lo = _mm256_dpbusd_avx_epi32(sum_lo, _mm256_and_si256(bytes_lo, low_mask), _mm256_set1_epi32(x4));
                    sum_hi = _mm256_dpbusd_avx_epi32(sum_hi, _mm256_and_si256(bytes_hi, low_mask), _mm256_set1_epi32(x4));
                    sum_lo = _mm256_dpbusd_avx_epi32(sum_lo, _mm256_and_si256(_mm256_srli_epi16(bytes_lo, 4), low_mask), _mm256_set1_epi32(x5));
                    sum_hi = _mm256_dpbusd_avx_epi32(sum_hi, _mm256_and_si256(_mm256_srli_epi16(bytes_hi, 4), low_mask), _mm256_set1_epi32(x5));
                }
                const __m256i correction = _mm256_set1_epi32(8 * x_sums[t * sums_stride + b]);
                sum_lo = _mm256_sub_epi32(sum_lo, correction);
                sum_hi = _mm256_sub_epi32(sum_hi, correction);
                uint16_t dx_bits = 0;
                std::memcpy(&dx_bits, x, 2);
                const __m256 dx = _mm256_set1_ps(f16_to_f32(dx_bits));
                const __m256 d_lo = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block))), dx);
                const __m256 d_hi = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 16))), dx);
                acc_lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum_lo), d_lo, acc_lo);
                acc_hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum_hi), d_hi, acc_hi);
                block += Q4_0_GROUP_BLOCK_BYTES;
            }
            const size_t row = g * Q8_0_GROUP;
            float values[Q8_0_GROUP];
            _mm256_storeu_ps(values, acc_lo);
            _mm256_storeu_ps(values + 8, acc_hi);
            const size_t count = row + Q8_0_GROUP <= last ? Q8_0_GROUP : last - row;
            std::memcpy(out + t * out_stride + row, values, count * sizeof(float));
        }
    }
    (void)n_out;
}

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
            const __m256i correction = _mm256_set1_epi32(128 * x_sums[b]);
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

void matmul_q4_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    for (size_t t = 0; t < n_tokens; ++t)
    {
        matvec_q4_0_packed_scalar(packed, xq + t * xq_stride, x_sums + t * sums_stride, n_in, n_out, first, last,
                                  out + t * out_stride);
    }
}

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

void exp_values_scalar(float* x, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = std::exp(x[i]);
    }
}

void attention_head_scalar(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                           size_t n_past, float scale, float* scores, float* out)
{
    for (size_t p = 0; p < n_past; ++p)
    {
        float s = 0.0f;
        for (size_t d = 0; d < head_dim; ++d)
        {
            s += q[d] * k_tiles[key_index(head_dim, p, d)];
        }
        scores[p] = s * scale;
    }
    softmax(scores, n_past);
    for (size_t d = 0; d < head_dim; ++d)
    {
        out[d] = 0.0f;
    }
    for (size_t p = 0; p < n_past; ++p)
    {
        const float weight = scores[p];
        const float* row = v + p * v_stride;
        for (size_t d = 0; d < head_dim; ++d)
        {
            out[d] += weight * row[d];
        }
    }
}

#if defined(FAISCA_AVX2)

namespace
{

// e^x = 2^n * e^r with x = n ln2 + r, |r| <= ln2 / 2, e^r by a degree 7 polynomial
constexpr float EXP_LO = -104.0f;
constexpr float EXP_HI = 88.72283f;
constexpr float EXP_ZERO_BELOW = -87.33655f; // smaller results are subnormal
constexpr float EXP_LOG2E = 1.44269504088896341f;
constexpr float EXP_LN2_HI = 0.693145751953125f;
constexpr float EXP_LN2_LO = 1.428606765330187e-06f;
constexpr float EXP_C2 = 0.5f;
constexpr float EXP_C3 = 0.16666667163372039795f;
constexpr float EXP_C4 = 0.041666667908430099487f;
constexpr float EXP_C5 = 0.0083333337679505348206f;
constexpr float EXP_C6 = 0.0013888889225199818611f;
constexpr float EXP_C7 = 0.00019841270113829523325f;

inline __m256 exp_avx2(__m256 x)
{
    const __m256 keep = _mm256_cmp_ps(x, _mm256_set1_ps(EXP_ZERO_BELOW), _CMP_GE_OQ);
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(EXP_LO)), _mm256_set1_ps(EXP_HI));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(EXP_LOG2E)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(EXP_LN2_HI), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(EXP_LN2_LO), r);
    __m256 p = _mm256_set1_ps(EXP_C7);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C6));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C4));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    // 2^n in two halves, n may reach 128
    const __m256i ni = _mm256_cvtps_epi32(n);
    const __m256i half = _mm256_srai_epi32(ni, 1);
    const __m256i rest = _mm256_sub_epi32(ni, half);
    const __m256 scale_a = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(half, _mm256_set1_epi32(127)), 23));
    const __m256 scale_b = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(rest, _mm256_set1_epi32(127)), 23));
    return _mm256_and_ps(_mm256_mul_ps(_mm256_mul_ps(p, scale_a), scale_b), keep);
}

#if defined(__AVX512F__)
inline __m512 exp_avx512(__m512 x)
{
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(EXP_LO)), _mm512_set1_ps(EXP_HI));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(EXP_LOG2E)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(EXP_LN2_HI), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(EXP_LN2_LO), r);
    __m512 p = _mm512_set1_ps(EXP_C7);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(EXP_C6));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(EXP_C5));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(EXP_C4));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(EXP_C3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(EXP_C2));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}
#endif

}

void exp_values(float* x, size_t n)
{
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16)
    {
        _mm512_storeu_ps(x + i, exp_avx512(_mm512_loadu_ps(x + i)));
    }
    if (i < n)
    {
        const __mmask16 mask = static_cast<__mmask16>((1u << (n - i)) - 1);
        _mm512_mask_storeu_ps(x + i, mask, exp_avx512(_mm512_maskz_loadu_ps(mask, x + i)));
        i = n;
    }
#else
    for (; i + 8 <= n; i += 8)
    {
        _mm256_storeu_ps(x + i, exp_avx2(_mm256_loadu_ps(x + i)));
    }
    for (; i < n; ++i)
    {
        x[i] = std::exp(x[i]);
    }
#endif
}

#if defined(__AVX512F__)

void attention_head(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                    size_t n_past, float scale, float* scores, float* out)
{
    if (head_dim % 16 != 0)
    {
        attention_head_scalar(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out);
        return;
    }

    // scores for 16 positions at a time: head_dim broadcast FMAs over one contiguous tile, no reductions
    const __m512 vscale = _mm512_set1_ps(scale);
    __m512 vmax = _mm512_set1_ps(-INFINITY);
    for (size_t p = 0; p < n_past; p += 16)
    {
        const __mmask16 mask = static_cast<__mmask16>(p + 16 <= n_past ? 0xffffu : (1u << (n_past - p)) - 1u);
        const float* tile = k_tiles + (p / KEY_TILE) * head_dim * KEY_TILE;
        __m512 s = _mm512_setzero_ps();
        for (size_t d = 0; d < head_dim; ++d)
        {
            s = _mm512_fmadd_ps(_mm512_set1_ps(q[d]), _mm512_loadu_ps(tile + d * KEY_TILE), s);
        }
        s = _mm512_mul_ps(s, vscale);
        _mm512_storeu_ps(scores + p, s);
        vmax = _mm512_mask_max_ps(vmax, mask, vmax, s);
    }
    const float max = _mm512_reduce_max_ps(vmax);

    __m512 vsum = _mm512_setzero_ps();
    for (size_t p = 0; p < n_past; p += 16)
    {
        const __mmask16 mask = static_cast<__mmask16>(p + 16 <= n_past ? 0xffffu : (1u << (n_past - p)) - 1u);
        const __m512 e = exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(scores + p), _mm512_set1_ps(max)));
        _mm512_storeu_ps(scores + p, e);
        vsum = _mm512_mask_add_ps(vsum, mask, vsum, e);
    }
    const float inverse = 1.0f / _mm512_reduce_add_ps(vsum);

    for (size_t d0 = 0; d0 < head_dim; d0 += 64)
    {
        const size_t width = head_dim - d0 < 64 ? head_dim - d0 : 64;
        __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
        for (size_t p = 0; p < n_past; ++p)
        {
            const __m512 weight = _mm512_set1_ps(scores[p]);
            const float* row = v + p * v_stride + d0;
            for (size_t c = 0; c < width / 16; ++c)
            {
                acc[c] = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 16 * c), acc[c]);
            }
        }
        for (size_t c = 0; c < width / 16; ++c)
        {
            _mm512_storeu_ps(out + d0 + 16 * c, _mm512_mul_ps(acc[c], _mm512_set1_ps(inverse)));
        }
    }
}

#else

void attention_head(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                    size_t n_past, float scale, float* scores, float* out)
{
    if (head_dim % 8 != 0)
    {
        attention_head_scalar(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out);
        return;
    }

    // 8 positions at a time: a tile holds 16, each d row of the tile is two ymm halves
    const size_t padded = (n_past + 7) / 8 * 8;
    const __m256 vscale = _mm256_set1_ps(scale);
    for (size_t p = 0; p < padded; p += 8)
    {
        const float* tile = k_tiles + (p / KEY_TILE) * head_dim * KEY_TILE + (p % KEY_TILE);
        __m256 s = _mm256_setzero_ps();
        for (size_t d = 0; d < head_dim; ++d)
        {
            s = _mm256_fmadd_ps(_mm256_set1_ps(q[d]), _mm256_loadu_ps(tile + d * KEY_TILE), s);
        }
        _mm256_storeu_ps(scores + p, _mm256_mul_ps(s, vscale));
    }
    float max = scores[0];
    for (size_t p = 1; p < n_past; ++p)
    {
        max = scores[p] > max ? scores[p] : max;
    }
    for (size_t p = 0; p < padded; p += 8)
    {
        _mm256_storeu_ps(scores + p, exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(scores + p), _mm256_set1_ps(max))));
    }
    double sum = 0.0;
    for (size_t p = 0; p < n_past; ++p)
    {
        sum += static_cast<double>(scores[p]);
    }
    const float inverse = static_cast<float>(1.0 / sum);

    for (size_t d0 = 0; d0 < head_dim; d0 += 64)
    {
        const size_t width = head_dim - d0 < 64 ? head_dim - d0 : 64;
        __m256 acc[8];
        for (size_t c = 0; c < 8; ++c)
        {
            acc[c] = _mm256_setzero_ps();
        }
        for (size_t p = 0; p < n_past; ++p)
        {
            const __m256 weight = _mm256_set1_ps(scores[p]);
            const float* row = v + p * v_stride + d0;
            for (size_t c = 0; c < width / 8; ++c)
            {
                acc[c] = _mm256_fmadd_ps(weight, _mm256_loadu_ps(row + 8 * c), acc[c]);
            }
        }
        for (size_t c = 0; c < width / 8; ++c)
        {
            _mm256_storeu_ps(out + d0 + 8 * c, _mm256_mul_ps(acc[c], _mm256_set1_ps(inverse)));
        }
    }
}

#endif

#else

void exp_values(float* x, size_t n)
{
    exp_values_scalar(x, n);
}

void attention_head(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                    size_t n_past, float scale, float* scores, float* out)
{
    attention_head_scalar(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out);
}

#endif

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
