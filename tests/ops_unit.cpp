#include "faisca/ops.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

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
    test_dot_and_matvec();
    test_rms_norm();
    test_rope();
    test_softmax();
    test_swiglu_and_add();

    std::printf("%s ops_unit: %d checks, %d failed\n", g_failures == 0 ? "ok  " : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
