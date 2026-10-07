#include "faisca/ops.h"

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

}

int main()
{
    test_f16();
    test_f32_to_f16();
    test_q8_0();
    test_dot_and_matvec();
    test_rms_norm();
    test_rope();
    test_softmax();
    test_swiglu_and_add();

    std::printf("%s ops_unit: %d checks, %d failed\n", g_failures == 0 ? "ok  " : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
