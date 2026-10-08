#include "faisca/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(__F16C__)
#include <immintrin.h>
#endif

using namespace faisca;

namespace
{

int g_checks = 0;
int g_failures = 0;

#define CHECK(condition)                                                              \
    do                                                                                \
    {                                                                                 \
        ++g_checks;                                                                   \
        if (!(condition))                                                             \
        {                                                                             \
            ++g_failures;                                                             \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        }                                                                             \
    } while (false)

bool close(double a, double b, double tolerance)
{
    return std::fabs(a - b) <= tolerance * (1.0 + std::fabs(b));
}

uint32_t bits_of(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    return bits;
}

std::vector<float> random_values(std::mt19937& rng, size_t n, float scale)
{
    std::normal_distribution<float> normal(0.0f, scale);
    std::vector<float> out(n);
    for (float& value : out)
    {
        value = normal(rng);
    }
    return out;
}

void test_f16()
{
    CHECK(f16_to_f32(0x0000) == 0.0f && !std::signbit(f16_to_f32(0x0000)));
    CHECK(f16_to_f32(0x8000) == 0.0f && std::signbit(f16_to_f32(0x8000)));
    CHECK(f16_to_f32(0x3c00) == 1.0f);
    CHECK(f16_to_f32(0xc000) == -2.0f);
    CHECK(f16_to_f32(0x3555) == 0.333251953125f);
    CHECK(f16_to_f32(0x7bff) == 65504.0f);
    CHECK(f16_to_f32(0x0400) == 6.103515625e-05f);
    CHECK(f16_to_f32(0x0001) == 5.960464477539063e-08f);
    CHECK(f16_to_f32(0x03ff) == 6.097555160522461e-05f);
    CHECK(std::isinf(f16_to_f32(0x7c00)) && f16_to_f32(0x7c00) > 0.0f);
    CHECK(std::isinf(f16_to_f32(0xfc00)) && f16_to_f32(0xfc00) < 0.0f);
    CHECK(std::isnan(f16_to_f32(0x7e00)));
    CHECK(bits_of(f16_to_f32(0x3555)) == 0x3eaaa000u);

    // every finite half round trips through float exactly
    bool exact = true;
    for (uint32_t h = 0; h < 0x10000; ++h)
    {
        const uint32_t exponent = (h >> 10) & 0x1f;
        if (exponent == 31)
        {
            continue;
        }
        const double sign = (h & 0x8000) != 0 ? -1.0 : 1.0;
        const double mantissa = static_cast<double>(h & 0x3ff);
        const double expected = exponent == 0 ? sign * mantissa * std::ldexp(1.0, -24)
                                              : sign * (1.0 + mantissa / 1024.0) * std::ldexp(1.0, static_cast<int>(exponent) - 15);
        exact = exact && static_cast<double>(f16_to_f32(static_cast<uint16_t>(h))) == expected;
    }
    CHECK(exact);
}

void test_f32_to_f16()
{
    CHECK(f32_to_f16(0.0f) == 0x0000);
    CHECK(f32_to_f16(-0.0f) == 0x8000);
    CHECK(f32_to_f16(1.0f) == 0x3c00);
    CHECK(f32_to_f16(-2.0f) == 0xc000);
    CHECK(f32_to_f16(65504.0f) == 0x7bff);
    CHECK(f32_to_f16(65519.0f) == 0x7bff);
    CHECK(f32_to_f16(65520.0f) == 0x7c00);
    CHECK(f32_to_f16(1e10f) == 0x7c00);
    CHECK(f32_to_f16(-1e10f) == 0xfc00);
    CHECK(f32_to_f16(5.960464477539063e-08f) == 0x0001);
    CHECK(f32_to_f16(2.9802322387695312e-08f) == 0x0000);
    CHECK(f32_to_f16(2.98023224e-08f * 1.5f) == 0x0001);
    CHECK(f32_to_f16(6.103515625e-05f) == 0x0400);
    CHECK((f32_to_f16(std::nanf("")) & 0x7c00) == 0x7c00 && (f32_to_f16(std::nanf("")) & 0x3ff) != 0);
    CHECK(f32_to_f16(1.0f + 1.0f / 2048.0f) == 0x3c00);
    CHECK(f32_to_f16(1.0f + 3.0f / 2048.0f) == 0x3c02);
    CHECK(f32_to_f16(1.0f + 1.0f / 1024.0f) == 0x3c01);

    bool round_trip = true;
    for (uint32_t h = 0; h < 0x10000; ++h)
    {
        if (((h >> 10) & 0x1f) == 31)
        {
            continue;
        }
        round_trip = round_trip && f32_to_f16(f16_to_f32(static_cast<uint16_t>(h))) == h;
    }
    CHECK(round_trip);

#if defined(__F16C__)
    std::mt19937 rng(23);
    std::uniform_int_distribution<uint32_t> any_bits(0, 0xffffffffu);
    std::uniform_real_distribution<float> small(-70000.0f, 70000.0f);
    bool same_as_f16c = true;
    for (int i = 0; i < 2000000 && same_as_f16c; ++i)
    {
        float value = 0.0f;
        if (i % 2 == 0)
        {
            const uint32_t bits = any_bits(rng);
            std::memcpy(&value, &bits, 4);
        }
        else
        {
            value = small(rng);
        }
        if (std::isnan(value))
        {
            continue;
        }
        same_as_f16c = f32_to_f16(value) == static_cast<uint16_t>(_cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        if (!same_as_f16c)
        {
            std::fprintf(stderr, "  f32_to_f16(%a) = %04x, F16C gives %04x\n", static_cast<double>(value),
                         f32_to_f16(value), _cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }
    CHECK(same_as_f16c);
#endif
}

void test_q8_0()
{
    std::mt19937 rng(29);
    const size_t n = 2560;
    std::vector<float> x = random_values(rng, n, 1.5f);
    x[5] = 0.0f;
    for (size_t i = 64; i < 96; ++i)
    {
        x[i] = 0.0f; // a block of zeros
    }
    x[100] = 3.0f;
    x[101] = -3.0f; // ties round to even: 127 * 3 / 3 = 127 exactly, no tie, but amax decides the scale

    std::vector<uint8_t> xq(n / Q8_0_BLOCK * Q8_0_BLOCK_BYTES);
    quantize_q8_0(x.data(), n, xq.data());

    bool blocks_ok = true;
    for (size_t b = 0; b < n / Q8_0_BLOCK; ++b)
    {
        const uint8_t* block = xq.data() + b * Q8_0_BLOCK_BYTES;
        uint16_t bits = 0;
        std::memcpy(&bits, block, 2);
        float amax = 0.0f;
        for (size_t j = 0; j < Q8_0_BLOCK; ++j)
        {
            amax = std::fabs(x[b * Q8_0_BLOCK + j]) > amax ? std::fabs(x[b * Q8_0_BLOCK + j]) : amax;
        }
        blocks_ok = blocks_ok && bits == f32_to_f16(amax / 127.0f);
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        for (size_t j = 0; j < Q8_0_BLOCK; ++j)
        {
            const int8_t q = static_cast<int8_t>(block[2 + j]);
            blocks_ok = blocks_ok && q == static_cast<int8_t>(std::nearbyint(x[b * Q8_0_BLOCK + j] * id));
            blocks_ok = blocks_ok && q != -128;
        }
    }
    CHECK(blocks_ok);

    std::vector<float> back(n);
    dequantize_q8_0(xq.data(), n, back.data());
    double error = 0.0;
    double energy = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        error += (static_cast<double>(back[i]) - x[i]) * (static_cast<double>(back[i]) - x[i]);
        energy += static_cast<double>(x[i]) * x[i];
    }
    CHECK(std::sqrt(error / energy) < 0.01);
    CHECK(back[64] == 0.0f && back[90] == 0.0f);

    const std::vector<float> w = random_values(rng, n, 0.05f);
    std::vector<uint8_t> wq(xq.size());
    quantize_q8_0(w.data(), n, wq.data());
    std::vector<float> w_back(n);
    dequantize_q8_0(wq.data(), n, w_back.data());

    double expected = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        expected += static_cast<double>(w_back[i]) * back[i];
    }
    const float scalar = dot_q8_0_scalar(wq.data(), xq.data(), n);
    const float fast = dot_q8_0(wq.data(), xq.data(), n);
    CHECK(close(scalar, expected, 1e-5));
    CHECK(close(fast, expected, 1e-5));
    CHECK(close(fast, scalar, 1e-6));

    const size_t n_out = 17;
    std::vector<uint8_t> matrix;
    std::vector<float> rows_back;
    for (size_t o = 0; o < n_out; ++o)
    {
        const std::vector<float> row = random_values(rng, n, 0.05f);
        std::vector<uint8_t> rq(xq.size());
        quantize_q8_0(row.data(), n, rq.data());
        matrix.insert(matrix.end(), rq.begin(), rq.end());
        std::vector<float> rb(n);
        dequantize_q8_0(rq.data(), n, rb.data());
        rows_back.insert(rows_back.end(), rb.begin(), rb.end());
    }
    std::vector<float> out(n_out);
    matvec_q8_0(matrix.data(), xq.data(), n, n_out, out.data());
    bool all_close = true;
    for (size_t o = 0; o < n_out; ++o)
    {
        double e = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            e += static_cast<double>(rows_back[o * n + i]) * back[i];
        }
        all_close = all_close && close(out[o], e, 1e-5);
    }
    CHECK(all_close);
}

void test_q8_0_packed()
{
    std::mt19937 rng(31);
    const size_t n_in = 960;
    for (const size_t n_out : {16, 37, 64})
    {
        const size_t row_bytes = n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
        std::vector<uint8_t> rows(n_out * row_bytes);
        std::vector<float> dense;
        for (size_t o = 0; o < n_out; ++o)
        {
            const std::vector<float> row = random_values(rng, n_in, 0.1f);
            quantize_q8_0(row.data(), n_in, rows.data() + o * row_bytes);
            std::vector<float> back(n_in);
            dequantize_q8_0(rows.data() + o * row_bytes, n_in, back.data());
            dense.insert(dense.end(), back.begin(), back.end());
        }

        std::vector<uint8_t> packed(q8_0_packed_bytes(n_in, n_out));
        CHECK(packed.size() == q8_0_padded_rows(n_out) / Q8_0_GROUP * (n_in / Q8_0_BLOCK) * Q8_0_GROUP_BLOCK_BYTES);
        pack_q8_0(rows.data(), n_in, n_out, packed.data());

        bool rows_back = true;
        std::vector<float> row(n_in);
        for (size_t o = 0; o < n_out; ++o)
        {
            row_q8_0_packed(packed.data(), n_in, o, row.data());
            rows_back = rows_back && std::memcmp(row.data(), dense.data() + o * n_in, n_in * sizeof(float)) == 0;
        }
        CHECK(rows_back);

        const std::vector<float> x = random_values(rng, n_in, 1.0f);
        std::vector<uint8_t> xq(n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES);
        quantize_q8_0(x.data(), n_in, xq.data());
        std::vector<int32_t> sums(n_in / Q8_0_BLOCK);
        q8_0_block_sums(xq.data(), n_in, sums.data());
        bool sums_ok = true;
        for (size_t b = 0; b < sums.size(); ++b)
        {
            int32_t expected = 0;
            for (size_t k = 0; k < Q8_0_BLOCK; ++k)
            {
                expected += static_cast<int8_t>(xq[b * Q8_0_BLOCK_BYTES + 2 + k]);
            }
            sums_ok = sums_ok && sums[b] == expected;
        }
        CHECK(sums_ok);

        std::vector<float> reference(n_out);
        for (size_t o = 0; o < n_out; ++o)
        {
            reference[o] = dot_q8_0_scalar(rows.data() + o * row_bytes, xq.data(), n_in);
        }
        std::vector<float> scalar(n_out, 999.0f);
        std::vector<float> fast(n_out, 999.0f);
        matvec_q8_0_packed_scalar(packed.data(), xq.data(), sums.data(), n_in, n_out, 0, n_out, scalar.data());
        matvec_q8_0_packed(packed.data(), xq.data(), sums.data(), n_in, n_out, 0, n_out, fast.data());
        bool all_close = true;
        for (size_t o = 0; o < n_out; ++o)
        {
            all_close = all_close && close(scalar[o], reference[o], 1e-6) && close(fast[o], reference[o], 1e-6);
        }
        CHECK(all_close);

        // the batched kernel gives every input exactly what the single kernel gives
        const size_t n_tokens = 21;
        std::vector<uint8_t> xqs(n_tokens * xq.size());
        std::vector<int32_t> sums_all(n_tokens * sums.size());
        std::vector<float> expected_all(n_tokens * n_out);
        for (size_t t = 0; t < n_tokens; ++t)
        {
            const std::vector<float> xt = random_values(rng, n_in, 1.0f + static_cast<float>(t));
            quantize_q8_0(xt.data(), n_in, xqs.data() + t * xq.size());
            q8_0_block_sums(xqs.data() + t * xq.size(), n_in, sums_all.data() + t * sums.size());
            matvec_q8_0_packed(packed.data(), xqs.data() + t * xq.size(), sums_all.data() + t * sums.size(), n_in,
                               n_out, 0, n_out, expected_all.data() + t * n_out);
        }
        std::vector<float> batched(n_tokens * n_out, 999.0f);
        matmul_q8_0_packed(packed.data(), xqs.data(), xq.size(), sums_all.data(), sums.size(), n_tokens, n_in, n_out, 0,
                           n_out, batched.data(), n_out);
        CHECK(std::memcmp(batched.data(), expected_all.data(), batched.size() * sizeof(float)) == 0);

        // a partial range writes only its rows
        std::vector<float> partial(n_out, 999.0f);
        matvec_q8_0_packed(packed.data(), xq.data(), sums.data(), n_in, n_out, 16, n_out, partial.data());
        bool partial_ok = true;
        for (size_t o = 0; o < n_out; ++o)
        {
            partial_ok = partial_ok && (o < 16 ? partial[o] == 999.0f : close(partial[o], reference[o], 1e-6));
        }
        CHECK(partial_ok);
    }
}

void test_q4_0_packed()
{
    std::mt19937 rng(41);
    const size_t n_in = 960;
    for (const size_t n_out : {16, 37})
    {
        // rows quantized by hand to Q4_0: scale = max / -8 as llama.cpp, nibble = round(x / scale) + 8
        const size_t row_bytes = n_in / Q4_0_BLOCK * Q4_0_BLOCK_BYTES;
        std::vector<uint8_t> rows(n_out * row_bytes);
        std::vector<float> dense;
        for (size_t o = 0; o < n_out; ++o)
        {
            const std::vector<float> row = random_values(rng, n_in, 0.1f);
            for (size_t b = 0; b < n_in / Q4_0_BLOCK; ++b)
            {
                float max = 0.0f;
                float amax = 0.0f;
                for (size_t k = 0; k < Q4_0_BLOCK; ++k)
                {
                    const float value = row[b * Q4_0_BLOCK + k];
                    if (std::fabs(value) > amax)
                    {
                        amax = std::fabs(value);
                        max = value;
                    }
                }
                const float d = max / -8.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                uint8_t* block = rows.data() + o * row_bytes + b * Q4_0_BLOCK_BYTES;
                const uint16_t bits = f32_to_f16(d);
                std::memcpy(block, &bits, 2);
                for (size_t j = 0; j < Q4_0_BLOCK / 2; ++j)
                {
                    const float v0 = row[b * Q4_0_BLOCK + j] * id + 8.5f;
                    const float v1 = row[b * Q4_0_BLOCK + j + 16] * id + 8.5f;
                    const uint8_t q0 = static_cast<uint8_t>(std::min(15.0f, std::max(0.0f, std::floor(v0))));
                    const uint8_t q1 = static_cast<uint8_t>(std::min(15.0f, std::max(0.0f, std::floor(v1))));
                    block[2 + j] = static_cast<uint8_t>(q0 | (q1 << 4));
                }
            }
            std::vector<float> back(n_in);
            dequantize_q4_0(rows.data() + o * row_bytes, n_in, back.data());
            dense.insert(dense.end(), back.begin(), back.end());
        }

        std::vector<uint8_t> packed(q4_0_packed_bytes(n_in, n_out));
        pack_q4_0(rows.data(), n_in, n_out, packed.data());
        bool rows_back = true;
        std::vector<float> row(n_in);
        for (size_t o = 0; o < n_out; ++o)
        {
            row_q4_0_packed(packed.data(), n_in, o, row.data());
            rows_back = rows_back && std::memcmp(row.data(), dense.data() + o * n_in, n_in * sizeof(float)) == 0;
        }
        CHECK(rows_back);

        const size_t n_tokens = 19;
        std::vector<uint8_t> xqs(n_tokens * n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES);
        std::vector<int32_t> sums(n_tokens * n_in / Q8_0_BLOCK);
        std::vector<float> xs_back(n_tokens * n_in);
        const size_t xq_stride = n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
        const size_t sums_stride = n_in / Q8_0_BLOCK;
        for (size_t t = 0; t < n_tokens; ++t)
        {
            const std::vector<float> x = random_values(rng, n_in, 1.0f);
            quantize_q8_0(x.data(), n_in, xqs.data() + t * xq_stride);
            q8_0_block_sums(xqs.data() + t * xq_stride, n_in, sums.data() + t * sums_stride);
            dequantize_q8_0(xqs.data() + t * xq_stride, n_in, xs_back.data() + t * n_in);
        }
        std::vector<float> scalar(n_tokens * n_out), fast(n_tokens * n_out, 999.0f);
        for (size_t t = 0; t < n_tokens; ++t)
        {
            matvec_q4_0_packed_scalar(packed.data(), xqs.data() + t * xq_stride, sums.data() + t * sums_stride, n_in,
                                      n_out, 0, n_out, scalar.data() + t * n_out);
        }
        matmul_q4_0_packed(packed.data(), xqs.data(), xq_stride, sums.data(), sums_stride, n_tokens, n_in, n_out, 0,
                           n_out, fast.data(), n_out);
        bool all_close = true;
        for (size_t t = 0; t < n_tokens; ++t)
        {
            for (size_t o = 0; o < n_out; ++o)
            {
                double expected = 0.0;
                for (size_t i = 0; i < n_in; ++i)
                {
                    expected += static_cast<double>(dense[o * n_in + i]) * xs_back[t * n_in + i];
                }
                all_close = all_close && close(scalar[t * n_out + o], expected, 1e-5) &&
                            close(fast[t * n_out + o], expected, 1e-5);
            }
        }
        CHECK(all_close);
    }
}

void test_q8_k()
{
    std::mt19937 rng(43);
    const size_t n = 5 * Q8_K_BLOCK;
    std::vector<float> x = random_values(rng, n, 2.0f);
    for (size_t i = 0; i < Q8_K_BLOCK; ++i)
    {
        x[2 * Q8_K_BLOCK + i] = 0.0f;
    }
    x[3 * Q8_K_BLOCK + 7] = -9.0f; // the largest magnitude of its block is negative
    std::vector<uint8_t> xq(n / Q8_K_BLOCK * Q8_K_BLOCK_BYTES);
    std::vector<int32_t> sums(n / 16);
    std::vector<float> back(n);
    quantize_q8_k(x.data(), n, xq.data());
    q8_k_block_sums(xq.data(), n, sums.data());
    dequantize_q8_k(xq.data(), n, back.data());
    bool scales_ok = true;
    bool values_ok = true;
    bool extreme_ok = true;
    bool sums_ok = true;
    for (size_t b = 0; b < n / Q8_K_BLOCK; ++b)
    {
        const uint8_t* block = xq.data() + b * Q8_K_BLOCK_BYTES;
        const int8_t* q = reinterpret_cast<const int8_t*>(block + 4);
        float d = 0.0f;
        std::memcpy(&d, block, 4);
        float max = 0.0f;
        for (size_t j = 0; j < Q8_K_BLOCK; ++j)
        {
            max = std::fabs(x[b * Q8_K_BLOCK + j]) > std::fabs(max) ? x[b * Q8_K_BLOCK + j] : max;
        }
        // llama.cpp: scale = 1 / (-127 / max), the extreme value becomes -127
        scales_ok = scales_ok && (max == 0.0f ? d == 0.0f : d == 1.0f / (-127.0f / max));
        for (size_t j = 0; j < Q8_K_BLOCK; ++j)
        {
            const float value = x[b * Q8_K_BLOCK + j];
            values_ok = values_ok && (max != 0.0f && value == max ? q[j] == -127 : std::fabs(back[b * Q8_K_BLOCK + j] - value) <= std::fabs(d) * 0.5f + 1e-6f);
            extreme_ok = extreme_ok && q[j] != -128;
        }
        for (size_t s = 0; s < Q8_K_BLOCK / 16; ++s)
        {
            int32_t sum = 0;
            for (size_t k = 0; k < 16; ++k)
            {
                sum += q[s * 16 + k];
            }
            sums_ok = sums_ok && sums[b * (Q8_K_BLOCK / 16) + s] == sum;
        }
    }
    CHECK(scales_ok);
    CHECK(values_ok);
    CHECK(extreme_ok);
    CHECK(sums_ok);
    CHECK(back[2 * Q8_K_BLOCK + 5] == 0.0f);
    CHECK(sums[2 * Q8_K_BLOCK / 16 + 3] == 0);
}

struct PackedFormat
{
    const char* name;
    size_t block;
    size_t block_bytes;
    bool k_input;
    void (*make_block)(std::mt19937&, uint8_t*);
    void (*dequantize)(const uint8_t*, size_t, float*);
    size_t (*packed_bytes)(size_t, size_t);
    void (*pack)(const uint8_t*, size_t, size_t, uint8_t*);
    void (*row)(const uint8_t*, size_t, size_t, float*);
    void (*matvec_scalar)(const uint8_t*, const uint8_t*, const int32_t*, size_t, size_t, size_t, size_t, float*);
    void (*matmul)(const uint8_t*, const uint8_t*, size_t, const int32_t*, size_t, size_t, size_t, size_t, size_t,
                   size_t, float*, size_t);
};

uint8_t random_byte(std::mt19937& rng)
{
    return static_cast<uint8_t>(std::uniform_int_distribution<int>(0, 255)(rng));
}

uint16_t random_half(std::mt19937& rng, float scale, bool positive)
{
    float value = std::normal_distribution<float>(0.0f, scale)(rng);
    value = positive ? std::fabs(value) : value;
    return f32_to_f16(value);
}

void make_q5_0_block(std::mt19937& rng, uint8_t* block)
{
    const uint16_t d = random_half(rng, 0.05f, false);
    std::memcpy(block, &d, 2);
    for (size_t i = 2; i < Q5_0_BLOCK_BYTES; ++i)
    {
        block[i] = random_byte(rng);
    }
}

void make_q4_k_block(std::mt19937& rng, uint8_t* block)
{
    const uint16_t d = random_half(rng, 0.01f, true);
    const uint16_t dmin = random_half(rng, 0.01f, true);
    std::memcpy(block, &d, 2);
    std::memcpy(block + 2, &dmin, 2);
    for (size_t i = 4; i < Q4_K_BLOCK_BYTES; ++i)
    {
        block[i] = random_byte(rng);
    }
}

void make_q6_k_block(std::mt19937& rng, uint8_t* block)
{
    for (size_t i = 0; i < 208; ++i)
    {
        block[i] = random_byte(rng);
    }
    const uint16_t d = random_half(rng, 0.002f, false);
    std::memcpy(block + 208, &d, 2);
}

void check_packed_format(const PackedFormat& f)
{
    std::mt19937 rng(47);
    const size_t n_in = 1280;
    for (const size_t n_out : {16, 37})
    {
        const size_t blocks = n_in / f.block;
        const size_t row_bytes = blocks * f.block_bytes;
        std::vector<uint8_t> rows(n_out * row_bytes);
        for (size_t i = 0; i < n_out * blocks; ++i)
        {
            f.make_block(rng, rows.data() + i * f.block_bytes);
        }
        std::vector<float> dense(n_out * n_in);
        for (size_t o = 0; o < n_out; ++o)
        {
            f.dequantize(rows.data() + o * row_bytes, n_in, dense.data() + o * n_in);
        }

        std::vector<uint8_t> packed(f.packed_bytes(n_in, n_out));
        f.pack(rows.data(), n_in, n_out, packed.data());
        bool rows_back = true;
        std::vector<float> row(n_in);
        for (size_t o = 0; o < n_out; ++o)
        {
            f.row(packed.data(), n_in, o, row.data());
            rows_back = rows_back && std::memcmp(row.data(), dense.data() + o * n_in, n_in * sizeof(float)) == 0;
        }
        if (!rows_back)
        {
            std::fprintf(stderr, "  %s: packed rows differ from the dequantized rows\n", f.name);
        }
        CHECK(rows_back);

        const size_t n_tokens = 19;
        const size_t xq_stride = f.k_input ? n_in / Q8_K_BLOCK * Q8_K_BLOCK_BYTES : n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
        const size_t sums_stride = f.k_input ? n_in / 16 : n_in / Q8_0_BLOCK;
        std::vector<uint8_t> xqs(n_tokens * xq_stride);
        std::vector<int32_t> sums(n_tokens * sums_stride);
        std::vector<float> xs_back(n_tokens * n_in);
        for (size_t t = 0; t < n_tokens; ++t)
        {
            const std::vector<float> x = random_values(rng, n_in, 1.0f);
            if (f.k_input)
            {
                quantize_q8_k(x.data(), n_in, xqs.data() + t * xq_stride);
                q8_k_block_sums(xqs.data() + t * xq_stride, n_in, sums.data() + t * sums_stride);
                dequantize_q8_k(xqs.data() + t * xq_stride, n_in, xs_back.data() + t * n_in);
            }
            else
            {
                quantize_q8_0(x.data(), n_in, xqs.data() + t * xq_stride);
                q8_0_block_sums(xqs.data() + t * xq_stride, n_in, sums.data() + t * sums_stride);
                dequantize_q8_0(xqs.data() + t * xq_stride, n_in, xs_back.data() + t * n_in);
            }
        }
        std::vector<float> scalar(n_tokens * n_out), fast(n_tokens * n_out, 999.0f);
        for (size_t t = 0; t < n_tokens; ++t)
        {
            f.matvec_scalar(packed.data(), xqs.data() + t * xq_stride, sums.data() + t * sums_stride, n_in, n_out, 0,
                            n_out, scalar.data() + t * n_out);
        }
        f.matmul(packed.data(), xqs.data(), xq_stride, sums.data(), sums_stride, n_tokens, n_in, n_out, 0, n_out,
                 fast.data(), n_out);
        bool scalar_close = true;
        bool same_bits = true;
        double worst = 0.0;
        for (size_t t = 0; t < n_tokens; ++t)
        {
            for (size_t o = 0; o < n_out; ++o)
            {
                double expected = 0.0;
                double norm = 0.0;
                for (size_t i = 0; i < n_in; ++i)
                {
                    expected += static_cast<double>(dense[o * n_in + i]) * xs_back[t * n_in + i];
                    norm += std::fabs(static_cast<double>(dense[o * n_in + i]) * xs_back[t * n_in + i]);
                }
                const double rel = std::fabs(scalar[t * n_out + o] - expected) / (norm + 1e-30);
                worst = rel > worst ? rel : worst;
                scalar_close = scalar_close && rel < 1e-6;
                same_bits = same_bits && bits_of(fast[t * n_out + o]) == bits_of(scalar[t * n_out + o]);
            }
        }
        if (!scalar_close || !same_bits)
        {
            std::fprintf(stderr, "  %s n_out %zu: worst scalar error %.3e, vector %s\n", f.name, n_out, worst,
                         same_bits ? "identical" : "DIFFERENT");
        }
        CHECK(scalar_close);
        CHECK(same_bits);
    }
}

void test_k_quants_packed()
{
    const PackedFormat formats[] = {
        {"Q5_0", Q5_0_BLOCK, Q5_0_BLOCK_BYTES, false, make_q5_0_block, dequantize_q5_0, q5_0_packed_bytes, pack_q5_0,
         row_q5_0_packed, matvec_q5_0_packed_scalar, matmul_q5_0_packed},
        {"Q4_K", Q4_K_BLOCK, Q4_K_BLOCK_BYTES, true, make_q4_k_block, dequantize_q4_k, q4_k_packed_bytes, pack_q4_k,
         row_q4_k_packed, matvec_q4_k_packed_scalar, matmul_q4_k_packed},
        {"Q6_K", Q6_K_BLOCK, Q6_K_BLOCK_BYTES, true, make_q6_k_block, dequantize_q6_k, q6_k_packed_bytes, pack_q6_k,
         row_q6_k_packed, matvec_q6_k_packed_scalar, matmul_q6_k_packed},
    };
    for (const PackedFormat& f : formats)
    {
        check_packed_format(f);
    }
}

void test_dot_and_matvec()
{
    std::mt19937 rng(7);
    for (const size_t n : {1, 7, 8, 9, 63, 64, 65, 960, 2560})
    {
        const std::vector<float> a = random_values(rng, n, 1.0f);
        const std::vector<float> b = random_values(rng, n, 1.0f);
        double expected = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            expected += static_cast<double>(a[i]) * b[i];
        }
        CHECK(close(dot(a.data(), b.data(), n), expected, 1e-5));
    }

    const size_t n_in = 96;
    const size_t n_out = 40;
    const std::vector<float> w = random_values(rng, n_in * n_out, 0.1f);
    const std::vector<float> x = random_values(rng, n_in, 1.0f);
    std::vector<float> out(n_out, 123.0f);
    matvec(w.data(), x.data(), n_in, n_out, out.data());
    bool all_close = true;
    for (size_t o = 0; o < n_out; ++o)
    {
        double expected = 0.0;
        for (size_t i = 0; i < n_in; ++i)
        {
            expected += static_cast<double>(w[o * n_in + i]) * x[i];
        }
        all_close = all_close && close(out[o], expected, 1e-5);
    }
    CHECK(all_close);

    const float identity[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float v[2] = {3.0f, -4.0f};
    float r[2] = {};
    matvec(identity, v, 2, 2, r);
    CHECK(r[0] == 3.0f && r[1] == -4.0f);
}

void test_rms_norm()
{
    std::mt19937 rng(11);
    const size_t n = 960;
    const std::vector<float> x = random_values(rng, n, 2.0f);
    const std::vector<float> w = random_values(rng, n, 1.0f);
    std::vector<float> out(n);
    rms_norm(x.data(), w.data(), n, 1e-5f, out.data());

    double sum = 0.0;
    for (const float value : x)
    {
        sum += static_cast<double>(value) * value;
    }
    const double scale = 1.0 / std::sqrt(sum / static_cast<double>(n) + 1e-5);
    bool all_close = true;
    for (size_t i = 0; i < n; ++i)
    {
        all_close = all_close && close(out[i], x[i] * scale * w[i], 1e-5);
    }
    CHECK(all_close);

    const float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float constant[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float r[4] = {};
    rms_norm(constant, ones, 4, 0.0f, r);
    CHECK(r[0] == 1.0f && r[3] == 1.0f);

    const float zeros[4] = {};
    rms_norm(zeros, ones, 4, 1e-5f, r);
    CHECK(r[0] == 0.0f && !std::isnan(r[0]));
}

void test_rope()
{
    std::mt19937 rng(13);
    const size_t n_head = 3;
    const size_t head_dim = 8;
    const std::vector<float> original = random_values(rng, n_head * head_dim, 1.0f);

    std::vector<float> x = original;
    rope_norm(x.data(), n_head, head_dim, head_dim, 0, 10000.0f);
    CHECK(x == original);

    x = original;
    rope_norm(x.data(), n_head, head_dim, head_dim, 17, 10000.0f);
    bool all_close = true;
    for (size_t h = 0; h < n_head; ++h)
    {
        for (size_t i = 0; i < head_dim; i += 2)
        {
            const double theta = 17.0 * std::pow(10000.0, -static_cast<double>(i) / static_cast<double>(head_dim));
            const double x0 = original[h * head_dim + i];
            const double x1 = original[h * head_dim + i + 1];
            all_close = all_close && close(x[h * head_dim + i], x0 * std::cos(theta) - x1 * std::sin(theta), 1e-4);
            all_close = all_close && close(x[h * head_dim + i + 1], x0 * std::sin(theta) + x1 * std::cos(theta), 1e-4);
        }
    }
    CHECK(all_close);

    double before = 0.0;
    double after = 0.0;
    for (size_t i = 0; i < original.size(); ++i)
    {
        before += static_cast<double>(original[i]) * original[i];
        after += static_cast<double>(x[i]) * x[i];
    }
    CHECK(close(after, before, 1e-5));

    // only the first n_rot values of each head rotate
    x = original;
    rope_norm(x.data(), n_head, head_dim, 4, 5, 10000.0f);
    bool tail_untouched = true;
    bool head_rotated = true;
    for (size_t h = 0; h < n_head; ++h)
    {
        for (size_t i = 4; i < head_dim; ++i)
        {
            tail_untouched = tail_untouched && x[h * head_dim + i] == original[h * head_dim + i];
        }
        head_rotated = head_rotated && x[h * head_dim] != original[h * head_dim];
    }
    CHECK(tail_untouched);
    CHECK(head_rotated);

    // frequency factors divide the angle of every pair
    const float factors[4] = {1.0f, 2.0f, 8.0f, 0.5f};
    x = original;
    rope_norm(x.data(), n_head, head_dim, head_dim, 17, 10000.0f, factors);
    bool factors_ok = true;
    for (size_t h = 0; h < n_head; ++h)
    {
        for (size_t i = 0; i < head_dim; i += 2)
        {
            const double theta = 17.0 * std::pow(10000.0, -static_cast<double>(i) / static_cast<double>(head_dim)) / factors[i / 2];
            const double x0 = original[h * head_dim + i];
            const double x1 = original[h * head_dim + i + 1];
            factors_ok = factors_ok && close(x[h * head_dim + i], x0 * std::cos(theta) - x1 * std::sin(theta), 1e-4);
            factors_ok = factors_ok && close(x[h * head_dim + i + 1], x0 * std::sin(theta) + x1 * std::cos(theta), 1e-4);
        }
    }
    CHECK(factors_ok);
}

void test_softmax()
{
    std::mt19937 rng(17);
    for (const size_t n : {1, 2, 5, 64, 1000})
    {
        std::vector<float> x = random_values(rng, n, 3.0f);
        const std::vector<float> copy = x;
        softmax(x.data(), n);

        double max = copy[0];
        for (const float value : copy)
        {
            max = value > max ? value : max;
        }
        double sum = 0.0;
        for (const float value : copy)
        {
            sum += std::exp(static_cast<double>(value) - max);
        }
        bool all_close = true;
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            all_close = all_close && close(x[i], std::exp(static_cast<double>(copy[i]) - max) / sum, 1e-5);
            total += x[i];
        }
        CHECK(all_close);
        CHECK(close(total, 1.0, 1e-5));
    }

    float large[3] = {1000.0f, 1000.0f, -1000.0f};
    softmax(large, 3);
    CHECK(large[0] == 0.5f && large[1] == 0.5f && large[2] == 0.0f);
}

void test_exp_and_attention()
{
    std::vector<float> x;
    for (float value = -100.0f; value <= 88.7f; value += 0.0137f)
    {
        x.push_back(value);
    }
    x.push_back(-87.0f);
    x.push_back(-87.4f);
    x.push_back(0.0f);
    x.push_back(88.72f);
    std::vector<float> fast = x;
    exp_values(fast.data(), fast.size());
    double worst = 0.0;
    bool finite = true;
    bool tiny_ok = true;
    for (size_t i = 0; i < x.size(); ++i)
    {
        const double expected = std::exp(static_cast<double>(x[i]));
        finite = finite && std::isfinite(fast[i]) && fast[i] >= 0.0f;
        if (expected >= 1.17549435e-38)
        {
            const double rel = std::fabs(static_cast<double>(fast[i]) - expected) / expected;
            worst = rel > worst ? rel : worst;
        }
        else
        {
            tiny_ok = tiny_ok && fast[i] <= 1.17549435e-38f;
        }
    }
    CHECK(finite);
    CHECK(tiny_ok);
    CHECK(worst < 2.5e-7);
    CHECK(fast[x.size() - 2] == 1.0f);
    CHECK(fast[x.size() - 1] > 3.3e38f);
    std::vector<float> large = {89.0f, 200.0f};
    exp_values(large.data(), large.size());
    CHECK(std::isinf(large[0]) || large[0] > 3.3e38f);
    CHECK(std::isinf(large[1]) || large[1] > 3.3e38f);

    std::mt19937 rng(37);
    const size_t n_ctx = 80;
    const size_t head_dim = 64;
    const size_t n_kv = 320;
    std::vector<float> k_t = random_values(rng, head_dim * n_ctx, 1.0f); // in tiles, see key_index
    std::vector<float> v = random_values(rng, n_ctx * n_kv, 1.0f);
    std::vector<float> q = random_values(rng, head_dim, 1.0f);
    for (float& value : q)
    {
        value *= 3.0f;
    }
    for (const size_t n_past : {1, 2, 15, 16, 17, 33, 64, 79})
    {
        std::vector<float> scores_a(n_ctx + 16), scores_b(n_ctx + 16), out_a(head_dim), out_b(head_dim);
        attention_head_scalar(q.data(), k_t.data(), v.data() + 64, n_kv, head_dim, n_past, 0.125f, scores_a.data(),
                              out_a.data());
        attention_head(q.data(), k_t.data(), v.data() + 64, n_kv, head_dim, n_past, 0.125f, scores_b.data(),
                       out_b.data());

        // the scalar kernel against a plain double loop over the tiled layout
        std::vector<double> plain(head_dim, 0.0);
        std::vector<double> weights(n_past);
        double max = -1e30;
        for (size_t p = 0; p < n_past; ++p)
        {
            double s = 0.0;
            for (size_t d = 0; d < head_dim; ++d)
            {
                s += static_cast<double>(q[d]) * k_t[key_index(head_dim, p, d)];
            }
            weights[p] = s * 0.125;
            max = weights[p] > max ? weights[p] : max;
        }
        double sum = 0.0;
        for (size_t p = 0; p < n_past; ++p)
        {
            weights[p] = std::exp(weights[p] - max);
            sum += weights[p];
        }
        for (size_t p = 0; p < n_past; ++p)
        {
            for (size_t d = 0; d < head_dim; ++d)
            {
                plain[d] += weights[p] / sum * v[64 + p * n_kv + d];
            }
        }
        bool scalar_close = true;
        for (size_t d = 0; d < head_dim; ++d)
        {
            scalar_close = scalar_close && close(out_a[d], plain[d], 1e-5);
        }
        CHECK(scalar_close);
        bool all_close = true;
        for (size_t d = 0; d < head_dim; ++d)
        {
            all_close = all_close && close(out_b[d], out_a[d], 1e-5);
        }
        CHECK(all_close);
    }
}

void test_f16_cache()
{
    std::mt19937 rng(41);
    std::vector<float> x = random_values(rng, 1000, 3.0f);
    x[3] = 1.0f + 1.0f / 2048.0f;
    x[4] = 1.0f + 3.0f / 2048.0f;
    x[5] = 2.98023224e-08f * 1.5f;
    x[6] = -65520.0f;
    x[7] = 1e-6f;
    x[8] = 0.0f;
    x[9] = -0.0f;
    for (const size_t n : {1, 7, 8, 9, 16, 23, 24, 31, 32, 40, 1000})
    {
        std::vector<uint16_t> fast(n, 0xaaaa), reference(n, 0x5555);
        f32_to_f16_values(x.data(), n, fast.data());
        f32_to_f16_values_scalar(x.data(), n, reference.data());
        bool same = true;
        for (size_t i = 0; i < n; ++i)
        {
            same = same && fast[i] == reference[i] && reference[i] == f32_to_f16(x[i]);
        }
        CHECK(same);
    }

    const size_t n_ctx = 80;
    const size_t head_dim = 64;
    const size_t n_kv = 320;
    std::vector<float> k_t = random_values(rng, head_dim * n_ctx, 1.0f);
    std::vector<float> v = random_values(rng, n_ctx * n_kv, 1.0f);
    std::vector<float> q = random_values(rng, head_dim, 3.0f);
    std::vector<uint16_t> k_half(k_t.size()), v_half(v.size());
    f32_to_f16_values(k_t.data(), k_t.size(), k_half.data());
    f32_to_f16_values(v.data(), v.size(), v_half.data());
    // the float32 kernels fed with the values the halves hold
    std::vector<float> k_round(k_t.size()), v_round(v.size());
    for (size_t i = 0; i < k_t.size(); ++i)
    {
        k_round[i] = f16_to_f32(k_half[i]);
    }
    for (size_t i = 0; i < v.size(); ++i)
    {
        v_round[i] = f16_to_f32(v_half[i]);
    }
    for (const size_t n_past : {1, 2, 15, 16, 17, 33, 64, 79})
    {
        std::vector<float> scores(n_ctx + 16), out_half(head_dim), out_round(head_dim), out_scalar(head_dim),
            out_scalar_round(head_dim);
        attention_head_f16(q.data(), k_half.data(), v_half.data() + 64, n_kv, head_dim, n_past, 0.125f, scores.data(),
                           out_half.data());
        attention_head(q.data(), k_round.data(), v_round.data() + 64, n_kv, head_dim, n_past, 0.125f, scores.data(),
                       out_round.data());
        attention_head_f16_scalar(q.data(), k_half.data(), v_half.data() + 64, n_kv, head_dim, n_past, 0.125f,
                                  scores.data(), out_scalar.data());
        attention_head_scalar(q.data(), k_round.data(), v_round.data() + 64, n_kv, head_dim, n_past, 0.125f,
                              scores.data(), out_scalar_round.data());
        bool same_fast = true;
        bool same_scalar = true;
        bool all_close = true;
        for (size_t d = 0; d < head_dim; ++d)
        {
            same_fast = same_fast && bits_of(out_half[d]) == bits_of(out_round[d]);
            same_scalar = same_scalar && bits_of(out_scalar[d]) == bits_of(out_scalar_round[d]);
            all_close = all_close && close(out_half[d], out_scalar[d], 1e-5);
        }
        CHECK(same_fast);
        CHECK(same_scalar);
        CHECK(all_close);
    }
}

void test_vector_quantizers_and_swiglu()
{
    std::mt19937 rng(53);
    const size_t n = 8 * Q8_K_BLOCK;
    std::vector<float> x = random_values(rng, n, 2.0f);
    for (size_t i = 0; i < Q8_K_BLOCK; ++i)
    {
        x[Q8_K_BLOCK + i] = 0.0f;
    }
    // a tie between a positive and a negative extreme: the first one wins
    x[2 * Q8_K_BLOCK + 40] = -7.5f;
    x[2 * Q8_K_BLOCK + 41] = 7.5f;
    x[3 * Q8_K_BLOCK + 9] = 7.5f;
    x[3 * Q8_K_BLOCK + 8] = -7.5f;
    x[4 * Q8_K_BLOCK + 5] = 1e-30f;
    std::vector<uint8_t> a(n / Q8_0_BLOCK * Q8_0_BLOCK_BYTES), b(a.size());
    quantize_q8_0(x.data(), n, a.data());
    quantize_q8_0_scalar(x.data(), n, b.data());
    CHECK(std::memcmp(a.data(), b.data(), a.size()) == 0);
    std::vector<uint8_t> c(n / Q8_K_BLOCK * Q8_K_BLOCK_BYTES), d(c.size());
    quantize_q8_k(x.data(), n, c.data());
    quantize_q8_k_scalar(x.data(), n, d.data());
    CHECK(std::memcmp(c.data(), d.data(), c.size()) == 0);

    std::vector<float> gate = random_values(rng, 2560 + 5, 4.0f);
    std::vector<float> up = random_values(rng, 2560 + 5, 1.0f);
    gate[0] = -200.0f;
    gate[1] = 200.0f;
    gate[2] = 0.0f;
    gate[3] = -88.5f;
    std::vector<float> fast(gate.size()), reference(gate.size());
    swiglu(gate.data(), up.data(), gate.size(), fast.data());
    swiglu_scalar(gate.data(), up.data(), gate.size(), reference.data());
    bool all_close = true;
    for (size_t i = 0; i < gate.size(); ++i)
    {
        all_close = all_close && close(fast[i], reference[i], 4e-7);
    }
    CHECK(all_close);
}

void test_swiglu_and_add()
{
    const float gate[4] = {0.0f, 1.0f, -1.0f, 20.0f};
    const float up[4] = {5.0f, 2.0f, 3.0f, 0.5f};
    float out[4] = {};
    swiglu(gate, up, 4, out);
    CHECK(out[0] == 0.0f);
    CHECK(close(out[1], 2.0 / (1.0 + std::exp(-1.0)), 1e-6));
    CHECK(close(out[2], -3.0 / (1.0 + std::exp(1.0)), 1e-6));
    CHECK(close(out[3], 10.0 / (1.0 + std::exp(-20.0)), 1e-6));

    float x[3] = {1.0f, 2.0f, 3.0f};
    const float y[3] = {0.5f, -2.0f, 10.0f};
    add(x, y, 3);
    CHECK(x[0] == 1.5f && x[1] == 0.0f && x[2] == 13.0f);
}

void test_whisper_ops()
{
    std::mt19937 rng(91);
    const size_t n = 96;
    std::vector<float> x = random_values(rng, n, 2.0f);
    std::vector<float> w = random_values(rng, n, 1.0f);
    std::vector<float> b = random_values(rng, n, 0.5f);
    std::vector<float> out(n);
    layer_norm(x.data(), w.data(), b.data(), n, 1e-5f, out.data());
    double mean = 0.0;
    for (float v : x)
    {
        mean += v;
    }
    mean /= n;
    double var = 0.0;
    for (float v : x)
    {
        var += (v - mean) * (v - mean);
    }
    var /= n;
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double expected = (x[i] - mean) / std::sqrt(var + 1e-5) * w[i] + b[i];
        worst = std::max(worst, std::fabs(expected - out[i]));
    }
    CHECK(worst < 1e-5);

    std::vector<float> g = random_values(rng, n, 4.0f);
    std::vector<float> g_ref = g;
    gelu(g.data(), n);
    worst = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double v = g_ref[i];
        const double expected = 0.5 * v * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (v + 0.044715 * v * v * v)));
        worst = std::max(worst, std::fabs(expected - g[i]));
    }
    CHECK(worst < 1e-5);

    // conv1d with kernel 3 and one zero of padding against a plain double loop, stride 1 and 2
    const size_t n_in = 5;
    const size_t n_out = 7;
    const size_t length = 20;
    std::vector<float> cx = random_values(rng, n_in * length, 1.0f);
    std::vector<float> cw = random_values(rng, n_out * n_in * 3, 1.0f);
    std::vector<float> cb = random_values(rng, n_out, 1.0f);
    for (size_t stride : {size_t(1), size_t(2)})
    {
        std::vector<float> co(n_out * (length / stride), 999.0f);
        conv1d_k3(cx.data(), n_in, length, cw.data(), cb.data(), stride, 0, n_out, co.data());
        worst = 0.0;
        for (size_t oc = 0; oc < n_out; ++oc)
        {
            for (size_t t = 0; t < length / stride; ++t)
            {
                double sum = cb[oc];
                for (size_t ic = 0; ic < n_in; ++ic)
                {
                    for (size_t k = 0; k < 3; ++k)
                    {
                        const long pos = static_cast<long>(t * stride + k) - 1;
                        if (pos >= 0 && pos < static_cast<long>(length))
                        {
                            sum += static_cast<double>(cw[(oc * n_in + ic) * 3 + k]) * cx[ic * length + static_cast<size_t>(pos)];
                        }
                    }
                }
                worst = std::max(worst, std::fabs(sum - co[oc * (length / stride) + t]));
            }
        }
        CHECK(worst < 1e-5);
    }

    // one attention row against a double softmax
    const size_t head_dim = 8;
    const size_t n_kv = 13;
    const size_t stride = 24;
    std::vector<float> q = random_values(rng, head_dim, 1.0f);
    std::vector<float> k = random_values(rng, n_kv * stride, 1.0f);
    std::vector<float> v = random_values(rng, n_kv * stride, 1.0f);
    std::vector<float> scores(n_kv);
    std::vector<float> ao(head_dim);
    attention_rows(q.data(), k.data(), v.data(), stride, head_dim, n_kv, 0.35f, scores.data(), ao.data());
    std::vector<double> p(n_kv);
    double pmax = -1e30;
    for (size_t j = 0; j < n_kv; ++j)
    {
        double s = 0.0;
        for (size_t d = 0; d < head_dim; ++d)
        {
            s += static_cast<double>(q[d]) * k[j * stride + d];
        }
        p[j] = s * 0.35;
        pmax = std::max(pmax, p[j]);
    }
    double psum = 0.0;
    for (size_t j = 0; j < n_kv; ++j)
    {
        p[j] = std::exp(p[j] - pmax);
        psum += p[j];
    }
    worst = 0.0;
    for (size_t d = 0; d < head_dim; ++d)
    {
        double expected = 0.0;
        for (size_t j = 0; j < n_kv; ++j)
        {
            expected += p[j] / psum * v[j * stride + d];
        }
        worst = std::max(worst, std::fabs(expected - ao[d]));
    }
    CHECK(worst < 1e-5);
}

}

void test_rope_neox()
{
    std::mt19937 rng(29);
    const size_t n_head = 2;
    const size_t head_dim = 8;
    const std::vector<float> original = random_values(rng, n_head * head_dim, 1.0f);
    for (const float freq_scale : {1.0f, 0.125f})
    {
        std::vector<float> x = original;
        rope_neox(x.data(), n_head, head_dim, head_dim, 23, 10000.0f, freq_scale);
        bool all_close = true;
        for (size_t h = 0; h < n_head; ++h)
        {
            for (size_t i = 0; i < head_dim / 2; ++i)
            {
                const double theta = freq_scale * 23.0 * std::pow(10000.0, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
                const double x0 = original[h * head_dim + i];
                const double x1 = original[h * head_dim + i + head_dim / 2];
                all_close = all_close && close(x[h * head_dim + i], x0 * std::cos(theta) - x1 * std::sin(theta), 1e-4);
                all_close = all_close && close(x[h * head_dim + i + head_dim / 2], x0 * std::sin(theta) + x1 * std::cos(theta), 1e-4);
            }
        }
        CHECK(all_close);
    }
}

void test_attention_skip()
{
    std::mt19937 rng(23);
    const size_t head_dim = 64;
    const size_t n_kv = 2 * head_dim;
    const size_t n_past = 45;
    std::vector<float> q = random_values(rng, head_dim, 1.0f);
    std::vector<float> k = random_values(rng, n_past * head_dim, 1.0f);
    std::vector<float> v = random_values(rng, (n_past + 1) * n_kv, 1.0f);
    std::vector<float> k_t(((n_past + 15) / 16) * 16 * head_dim, 0.0f);
    for (size_t p = 0; p < n_past; ++p)
    {
        for (size_t d = 0; d < head_dim; ++d)
        {
            k_t[key_index(head_dim, p, d)] = k[p * head_dim + d];
        }
    }
    double worst = 0.0;
    for (size_t skip = 0; skip < 16; ++skip)
    {
        std::vector<float> scores_a(64), scores_b(64), out_a(head_dim), out_b(head_dim);
        attention_head_scalar(q.data(), k_t.data(), v.data() + 64, n_kv, head_dim, n_past, 0.125f, scores_a.data(),
                              out_a.data(), skip);
        attention_head(q.data(), k_t.data(), v.data() + 64, n_kv, head_dim, n_past, 0.125f, scores_b.data(), out_b.data(),
                       skip);
        std::vector<double> w(n_past, 0.0);
        double max = -1e30;
        for (size_t p = skip; p < n_past; ++p)
        {
            double s = 0.0;
            for (size_t d = 0; d < head_dim; ++d)
            {
                s += static_cast<double>(q[d]) * k[p * head_dim + d];
            }
            w[p] = s * 0.125;
            max = std::max(max, w[p]);
        }
        double sum = 0.0;
        for (size_t p = skip; p < n_past; ++p)
        {
            w[p] = std::exp(w[p] - max);
            sum += w[p];
        }
        for (size_t d = 0; d < head_dim; ++d)
        {
            double expected = 0.0;
            for (size_t p = skip; p < n_past; ++p)
            {
                expected += w[p] / sum * v[64 + p * n_kv + d];
            }
            worst = std::max(worst, std::fabs(expected - out_a[d]));
            worst = std::max(worst, std::fabs(expected - out_b[d]));
        }
    }
    CHECK(worst < 1e-5);
    std::printf("attention_head with skip: worst error %.2e\n", worst);
}

void test_conv1d_rows()
{
    std::mt19937 rng(77);
    const size_t cases[][5] = {{512, 7, 1, 4, 100}, {64, 3, 3, 5, 200}, {32, 1, 1, 3, 70}, {8, 2, 1, 1, 150}, {16, 3, 1, 6, 15}};
    double worst = 0.0;
    for (const size_t* c : cases)
    {
        const size_t ic = c[0];
        const size_t k = c[1];
        const size_t dilation = c[2];
        const size_t oc = c[3];
        const size_t length = c[4];
        const size_t pad = (k - 1) * dilation;
        std::vector<float> x = random_values(rng, ic * length, 1.0f);
        std::vector<float> w = random_values(rng, oc * ic * k, 0.2f);
        std::vector<float> b = random_values(rng, oc, 0.5f);
        std::vector<float> out(oc * length, 0.0f);
        const size_t t_split = length / 3;
        conv1d_rows(x.data(), length, length, ic, w.data(), k, dilation, pad, b.data(), 0, oc, 0, t_split, out.data(), length);
        conv1d_rows(x.data(), length, length, ic, w.data(), k, dilation, pad, b.data(), 0, oc, t_split, length, out.data(), length);
        for (size_t o = 0; o < oc; ++o)
        {
            for (size_t t = 0; t < length; ++t)
            {
                double expected = b[o];
                for (size_t i = 0; i < ic; ++i)
                {
                    for (size_t j = 0; j < k; ++j)
                    {
                        const ptrdiff_t src = static_cast<ptrdiff_t>(t + j * dilation) - static_cast<ptrdiff_t>(pad);
                        if (src >= 0)
                        {
                            expected += static_cast<double>(w[(o * ic + i) * k + j]) * x[i * length + static_cast<size_t>(src)];
                        }
                    }
                }
                worst = std::max(worst, std::fabs(expected - out[o * length + t]) / (std::fabs(expected) + 1.0));
            }
        }
    }
    CHECK(worst < 1e-5);
    std::printf("conv1d_rows: worst error %.2e\n", worst);
}

int main()
{
    test_f16();
    test_f32_to_f16();
    test_q8_0();
    test_q8_0_packed();
    test_q4_0_packed();
    test_q8_k();
    test_k_quants_packed();
    test_dot_and_matvec();
    test_rms_norm();
    test_rope();
    test_softmax();
    test_exp_and_attention();
    test_f16_cache();
    test_vector_quantizers_and_swiglu();
    test_swiglu_and_add();
    test_whisper_ops();
    test_conv1d_rows();
    test_attention_skip();
    test_rope_neox();

    std::printf("%s ops_unit: %d checks, %d failed\n", g_failures == 0 ? "ok  " : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
