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

namespace
{

// value k of a Q5_0 block: nibbles as Q4_0, bit k of the 4 high bit bytes
inline uint8_t q5_0_value(const uint8_t* block, size_t k)
{
    uint32_t high = 0;
    std::memcpy(&high, block + 2, 4);
    const uint8_t byte = block[6 + (k % 16)];
    const uint8_t low = k < 16 ? static_cast<uint8_t>(byte & 0x0f) : static_cast<uint8_t>(byte >> 4);
    return static_cast<uint8_t>(low | (((high >> k) & 1u) << 4));
}

// byte of the packed high bits of value k of row r and the bit it lives at
inline size_t q5_0_packed_high_byte(size_t r, size_t k, unsigned& bit)
{
    bit = static_cast<unsigned>(k / 4);
    return 2 * Q8_0_GROUP + 4 * 64 + r * 4 + (k % 4);
}

// scale and min of sub block j of a Q4_K super block, 6 bits each from 12 bytes
inline void q4_k_scale_min(const uint8_t* scales, size_t j, uint8_t& scale, uint8_t& min)
{
    if (j < 4)
    {
        scale = static_cast<uint8_t>(scales[j] & 63);
        min = static_cast<uint8_t>(scales[j + 4] & 63);
    }
    else
    {
        scale = static_cast<uint8_t>((scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4));
        min = static_cast<uint8_t>((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }
}

// value i of a Q4_K super block: 64 values per 32 bytes, low nibbles then high
inline uint8_t q4_k_value(const uint8_t* block, size_t i)
{
    const uint8_t byte = block[16 + (i / 64) * 32 + (i % 32)];
    return (i / 32) % 2 == 0 ? static_cast<uint8_t>(byte & 0x0f) : static_cast<uint8_t>(byte >> 4);
}

// value i of a Q6_K super block, 0 to 63, as ggml's dequantize_row_q6_K reads it
inline uint8_t q6_k_value(const uint8_t* block, size_t i)
{
    const uint8_t* ql = block + (i / 128) * 64;
    const uint8_t* qh = block + 128 + (i / 128) * 32;
    const size_t l = i % 32;
    const size_t part = (i % 128) / 32;
    const uint8_t low = part < 2 ? static_cast<uint8_t>(ql[l + 32 * part] & 0x0f)
                                 : static_cast<uint8_t>(ql[l + 32 * (part - 2)] >> 4);
    const uint8_t high = static_cast<uint8_t>((qh[l] >> (2 * part)) & 3);
    return static_cast<uint8_t>(low | (high << 4));
}

inline float half_at(const uint8_t* at)
{
    uint16_t bits = 0;
    std::memcpy(&bits, at, 2);
    return f16_to_f32(bits);
}

// ggml's nearest_int: round to nearest even for |x| < 2^22
inline int32_t nearest_int(float value)
{
    const float shifted = value + 12582912.0f;
    int32_t bits = 0;
    std::memcpy(&bits, &shifted, 4);
    return (bits & 0x007fffff) - 0x00400000;
}

// where value k (0 to 255) of row r lives in the nibble chunks of a packed K-quant block
inline size_t k_packed_nibble_byte(size_t r, size_t k, bool& high)
{
    const size_t chunk = k / 8;
    const size_t within = k % 8;
    high = within >= 4;
    return chunk * 64 + r * 4 + (within % 4);
}

}

void dequantize_q5_0(const uint8_t* blocks, size_t n, float* out)
{
    for (size_t i = 0; i < n; i += Q5_0_BLOCK)
    {
        const float d = half_at(blocks);
        for (size_t k = 0; k < Q5_0_BLOCK; ++k)
        {
            out[i + k] = static_cast<float>(static_cast<int>(q5_0_value(blocks, k)) - 16) * d;
        }
        blocks += Q5_0_BLOCK_BYTES;
    }
}

size_t q5_0_packed_bytes(size_t n_in, size_t n_out)
{
    return q8_0_padded_rows(n_out) / Q8_0_GROUP * (n_in / Q5_0_BLOCK) * Q5_0_GROUP_BLOCK_BYTES;
}

void pack_q5_0(const uint8_t* rows, size_t n_in, size_t n_out, uint8_t* out)
{
    const size_t blocks = n_in / Q5_0_BLOCK;
    const size_t row_bytes = blocks * Q5_0_BLOCK_BYTES;
    const size_t groups = q8_0_padded_rows(n_out) / Q8_0_GROUP;

    for (size_t g = 0; g < groups; ++g)
    {
        for (size_t b = 0; b < blocks; ++b)
        {
            std::memset(out, 0, Q5_0_GROUP_BLOCK_BYTES);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                const size_t row = g * Q8_0_GROUP + r;
                for (size_t k = 0; k < Q5_0_BLOCK; ++k)
                {
                    // padding rows get the value 16, which is 0 after the offset, and a zero scale
                    uint8_t value = 16;
                    if (row < n_out)
                    {
                        const uint8_t* block = rows + row * row_bytes + b * Q5_0_BLOCK_BYTES;
                        out[2 * r] = block[0];
                        out[2 * r + 1] = block[1];
                        value = q5_0_value(block, k);
                    }
                    bool high = false;
                    const size_t at = 2 * Q8_0_GROUP + k_packed_nibble_byte(r, k, high);
                    out[at] = static_cast<uint8_t>(out[at] | (high ? (value & 0x0f) << 4 : (value & 0x0f)));
                    unsigned bit = 0;
                    const size_t high_at = q5_0_packed_high_byte(r, k, bit);
                    out[high_at] = static_cast<uint8_t>(out[high_at] | ((value >> 4) << bit));
                }
            }
            out += Q5_0_GROUP_BLOCK_BYTES;
        }
    }
}

namespace
{

inline int q5_0_packed_value(const uint8_t* block, size_t r, size_t k)
{
    bool high = false;
    const size_t at = 2 * Q8_0_GROUP + k_packed_nibble_byte(r, k, high);
    unsigned bit = 0;
    const size_t high_at = q5_0_packed_high_byte(r, k, bit);
    const int low = high ? block[at] >> 4 : block[at] & 0x0f;
    return low | (((block[high_at] >> bit) & 1) << 4);
}

}

void row_q5_0_packed(const uint8_t* packed, size_t n_in, size_t row, float* out)
{
    const size_t blocks = n_in / Q5_0_BLOCK;
    const size_t g = row / Q8_0_GROUP;
    const size_t r = row % Q8_0_GROUP;
    const uint8_t* group = packed + g * blocks * Q5_0_GROUP_BLOCK_BYTES;
    for (size_t b = 0; b < blocks; ++b)
    {
        const uint8_t* block = group + b * Q5_0_GROUP_BLOCK_BYTES;
        const float d = half_at(block + 2 * r);
        for (size_t k = 0; k < Q5_0_BLOCK; ++k)
        {
            out[b * Q5_0_BLOCK + k] = static_cast<float>(q5_0_packed_value(block, r, k) - 16) * d;
        }
    }
}

void matvec_q5_0_packed_scalar(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in,
                               size_t n_out, size_t first, size_t last, float* out)
{
    const size_t blocks = n_in / Q5_0_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* group = packed + g * blocks * Q5_0_GROUP_BLOCK_BYTES;
        float acc[Q8_0_GROUP] = {};
        for (size_t b = 0; b < blocks; ++b)
        {
            const uint8_t* block = group + b * Q5_0_GROUP_BLOCK_BYTES;
            const int8_t* x = reinterpret_cast<const int8_t*>(xq + b * Q8_0_BLOCK_BYTES + 2);
            const float dx = half_at(xq + b * Q8_0_BLOCK_BYTES);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                int32_t sum = 0;
                for (size_t k = 0; k < Q5_0_BLOCK; ++k)
                {
                    sum += q5_0_packed_value(block, r, k) * x[k];
                }
                sum -= 16 * x_sums[b];
                const float d = half_at(block + 2 * r) * dx;
                acc[r] = std::fma(static_cast<float>(sum), d, acc[r]);
            }
        }
        for (size_t r = 0; r < Q8_0_GROUP && g * Q8_0_GROUP + r < last; ++r)
        {
            out[g * Q8_0_GROUP + r] = acc[r];
        }
    }
    (void)n_out;
}

void quantize_q8_k_scalar(const float* x, size_t n, uint8_t* out)
{
    for (size_t i = 0; i < n; i += Q8_K_BLOCK)
    {
        float amax = 0.0f;
        float max = 0.0f;
        for (size_t j = 0; j < Q8_K_BLOCK; ++j)
        {
            const float a = std::fabs(x[i + j]);
            if (a > amax)
            {
                amax = a;
                max = x[i + j];
            }
        }
        int8_t* q = reinterpret_cast<int8_t*>(out + 4);
        if (amax == 0.0f)
        {
            const float zero = 0.0f;
            std::memcpy(out, &zero, 4);
            std::memset(q, 0, Q8_K_BLOCK);
            out += Q8_K_BLOCK_BYTES;
            continue;
        }
        const float iscale = -127.0f / max;
        for (size_t j = 0; j < Q8_K_BLOCK; ++j)
        {
            const int32_t v = nearest_int(iscale * x[i + j]);
            q[j] = static_cast<int8_t>(v < 127 ? v : 127);
        }
        const float d = 1.0f / iscale;
        std::memcpy(out, &d, 4);
        out += Q8_K_BLOCK_BYTES;
    }
}

void dequantize_q8_k(const uint8_t* blocks, size_t n, float* out)
{
    for (size_t i = 0; i < n; i += Q8_K_BLOCK)
    {
        float d = 0.0f;
        std::memcpy(&d, blocks, 4);
        const int8_t* q = reinterpret_cast<const int8_t*>(blocks + 4);
        for (size_t j = 0; j < Q8_K_BLOCK; ++j)
        {
            out[i + j] = d * static_cast<float>(q[j]);
        }
        blocks += Q8_K_BLOCK_BYTES;
    }
}

void q8_k_block_sums(const uint8_t* xq, size_t n, int32_t* out)
{
    for (size_t b = 0; b < n / Q8_K_BLOCK; ++b)
    {
        const int8_t* q = reinterpret_cast<const int8_t*>(xq + b * Q8_K_BLOCK_BYTES + 4);
        for (size_t s = 0; s < Q8_K_BLOCK / 16; ++s)
        {
            int32_t sum = 0;
            for (size_t k = 0; k < 16; ++k)
            {
                sum += q[s * 16 + k];
            }
            out[b * (Q8_K_BLOCK / 16) + s] = sum;
        }
    }
}

void dequantize_q4_k(const uint8_t* blocks, size_t n, float* out)
{
    for (size_t i = 0; i < n; i += Q4_K_BLOCK)
    {
        const float d = half_at(blocks);
        const float dmin = half_at(blocks + 2);
        for (size_t j = 0; j < Q4_K_BLOCK / 32; ++j)
        {
            uint8_t scale = 0;
            uint8_t min = 0;
            q4_k_scale_min(blocks + 4, j, scale, min);
            const float dj = d * static_cast<float>(scale);
            const float mj = dmin * static_cast<float>(min);
            for (size_t l = 0; l < 32; ++l)
            {
                out[i + j * 32 + l] = dj * static_cast<float>(q4_k_value(blocks, j * 32 + l)) - mj;
            }
        }
        blocks += Q4_K_BLOCK_BYTES;
    }
}

void dequantize_q6_k(const uint8_t* blocks, size_t n, float* out)
{
    for (size_t i = 0; i < n; i += Q6_K_BLOCK)
    {
        const float d = half_at(blocks + 208);
        const int8_t* scales = reinterpret_cast<const int8_t*>(blocks + 192);
        for (size_t k = 0; k < Q6_K_BLOCK; ++k)
        {
            const int q = static_cast<int>(q6_k_value(blocks, k)) - 32;
            out[i + k] = d * static_cast<float>(scales[k / 16]) * static_cast<float>(q);
        }
        blocks += Q6_K_BLOCK_BYTES;
    }
}

bool k_quants_packed_supported()
{
#if defined(FAISCA_AVX512_VNNI) || defined(FAISCA_AVX_VNNI)
    return true;
#else
    return false;
#endif
}

size_t q4_k_packed_bytes(size_t n_in, size_t n_out)
{
    return q8_0_padded_rows(n_out) / Q8_0_GROUP * (n_in / Q4_K_BLOCK) * Q4_K_GROUP_BLOCK_BYTES;
}

size_t q6_k_packed_bytes(size_t n_in, size_t n_out)
{
    return q8_0_padded_rows(n_out) / Q8_0_GROUP * (n_in / Q6_K_BLOCK) * Q6_K_GROUP_BLOCK_BYTES;
}

namespace
{

constexpr size_t Q4_K_PACKED_SCALES = 4 * Q8_0_GROUP;
constexpr size_t Q4_K_PACKED_MINS = Q4_K_PACKED_SCALES + 8 * Q8_0_GROUP;
constexpr size_t Q4_K_PACKED_QUANTS = Q4_K_PACKED_MINS + 8 * Q8_0_GROUP;
constexpr size_t Q6_K_PACKED_SCALES = 2 * Q8_0_GROUP;
constexpr size_t Q6_K_PACKED_QUANTS = Q6_K_PACKED_SCALES + 16 * Q8_0_GROUP;
constexpr size_t Q6_K_PACKED_HIGH = Q6_K_PACKED_QUANTS + Q8_0_GROUP * Q6_K_BLOCK / 2;

inline int q4_k_packed_value(const uint8_t* block, size_t r, size_t k)
{
    bool high = false;
    const size_t at = Q4_K_PACKED_QUANTS + k_packed_nibble_byte(r, k, high);
    return high ? block[at] >> 4 : block[at] & 0x0f;
}

inline int q6_k_packed_value(const uint8_t* block, size_t r, size_t k)
{
    bool high = false;
    const size_t at = Q6_K_PACKED_QUANTS + k_packed_nibble_byte(r, k, high);
    const int low = high ? block[at] >> 4 : block[at] & 0x0f;
    const size_t high_at = Q6_K_PACKED_HIGH + (k / 16) * 64 + r * 4 + (k % 4);
    const unsigned shift = 2 * static_cast<unsigned>((k % 16) / 4);
    return low | (((block[high_at] >> shift) & 3) << 4);
}

}

void pack_q4_k(const uint8_t* rows, size_t n_in, size_t n_out, uint8_t* out)
{
    const size_t blocks = n_in / Q4_K_BLOCK;
    const size_t row_bytes = blocks * Q4_K_BLOCK_BYTES;
    const size_t groups = q8_0_padded_rows(n_out) / Q8_0_GROUP;

    for (size_t g = 0; g < groups; ++g)
    {
        for (size_t b = 0; b < blocks; ++b)
        {
            // padding rows are all zero: zero scales and mins make any nibble worth nothing
            std::memset(out, 0, Q4_K_GROUP_BLOCK_BYTES);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                const size_t row = g * Q8_0_GROUP + r;
                if (row >= n_out)
                {
                    continue;
                }
                const uint8_t* block = rows + row * row_bytes + b * Q4_K_BLOCK_BYTES;
                std::memcpy(out + 2 * r, block, 2);
                std::memcpy(out + 2 * Q8_0_GROUP + 2 * r, block + 2, 2);
                for (size_t s = 0; s < 8; ++s)
                {
                    uint8_t scale = 0;
                    uint8_t min = 0;
                    q4_k_scale_min(block + 4, s, scale, min);
                    out[Q4_K_PACKED_SCALES + s * Q8_0_GROUP + r] = scale;
                    out[Q4_K_PACKED_MINS + s * Q8_0_GROUP + r] = min;
                }
                for (size_t k = 0; k < Q4_K_BLOCK; ++k)
                {
                    bool high = false;
                    const size_t at = Q4_K_PACKED_QUANTS + k_packed_nibble_byte(r, k, high);
                    const uint8_t value = q4_k_value(block, k);
                    out[at] = static_cast<uint8_t>(out[at] | (high ? value << 4 : value));
                }
            }
            out += Q4_K_GROUP_BLOCK_BYTES;
        }
    }
}

void row_q4_k_packed(const uint8_t* packed, size_t n_in, size_t row, float* out)
{
    const size_t blocks = n_in / Q4_K_BLOCK;
    const size_t g = row / Q8_0_GROUP;
    const size_t r = row % Q8_0_GROUP;
    const uint8_t* group = packed + g * blocks * Q4_K_GROUP_BLOCK_BYTES;
    for (size_t b = 0; b < blocks; ++b)
    {
        const uint8_t* block = group + b * Q4_K_GROUP_BLOCK_BYTES;
        const float d = half_at(block + 2 * r);
        const float dmin = half_at(block + 2 * Q8_0_GROUP + 2 * r);
        for (size_t s = 0; s < 8; ++s)
        {
            const float ds = d * static_cast<float>(block[Q4_K_PACKED_SCALES + s * Q8_0_GROUP + r]);
            const float ms = dmin * static_cast<float>(block[Q4_K_PACKED_MINS + s * Q8_0_GROUP + r]);
            for (size_t l = 0; l < 32; ++l)
            {
                out[b * Q4_K_BLOCK + s * 32 + l] = ds * static_cast<float>(q4_k_packed_value(block, r, s * 32 + l)) - ms;
            }
        }
    }
}

void matvec_q4_k_packed_scalar(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in,
                               size_t n_out, size_t first, size_t last, float* out)
{
    const size_t blocks = n_in / Q4_K_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* group = packed + g * blocks * Q4_K_GROUP_BLOCK_BYTES;
        float acc[Q8_0_GROUP] = {};
        for (size_t b = 0; b < blocks; ++b)
        {
            const uint8_t* block = group + b * Q4_K_GROUP_BLOCK_BYTES;
            const int8_t* x = reinterpret_cast<const int8_t*>(xq + b * Q8_K_BLOCK_BYTES + 4);
            float dx = 0.0f;
            std::memcpy(&dx, xq + b * Q8_K_BLOCK_BYTES, 4);
            const int32_t* sums = x_sums + b * (Q8_K_BLOCK / 16);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                int32_t sumi = 0;
                int32_t summ = 0;
                for (size_t s = 0; s < 8; ++s)
                {
                    int32_t dot = 0;
                    for (size_t l = 0; l < 32; ++l)
                    {
                        dot += q4_k_packed_value(block, r, s * 32 + l) * x[s * 32 + l];
                    }
                    sumi += block[Q4_K_PACKED_SCALES + s * Q8_0_GROUP + r] * dot;
                    summ += block[Q4_K_PACKED_MINS + s * Q8_0_GROUP + r] * (sums[2 * s] + sums[2 * s + 1]);
                }
                const float d = half_at(block + 2 * r) * dx;
                const float dmin = half_at(block + 2 * Q8_0_GROUP + 2 * r) * dx;
                acc[r] = std::fma(static_cast<float>(sumi), d, acc[r]);
                acc[r] = std::fma(-static_cast<float>(summ), dmin, acc[r]);
            }
        }
        for (size_t r = 0; r < Q8_0_GROUP && g * Q8_0_GROUP + r < last; ++r)
        {
            out[g * Q8_0_GROUP + r] = acc[r];
        }
    }
    (void)n_out;
}

void pack_q6_k(const uint8_t* rows, size_t n_in, size_t n_out, uint8_t* out)
{
    const size_t blocks = n_in / Q6_K_BLOCK;
    const size_t row_bytes = blocks * Q6_K_BLOCK_BYTES;
    const size_t groups = q8_0_padded_rows(n_out) / Q8_0_GROUP;

    for (size_t g = 0; g < groups; ++g)
    {
        for (size_t b = 0; b < blocks; ++b)
        {
            // padding rows are all zero: zero scales make any value worth nothing
            std::memset(out, 0, Q6_K_GROUP_BLOCK_BYTES);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                const size_t row = g * Q8_0_GROUP + r;
                if (row >= n_out)
                {
                    continue;
                }
                const uint8_t* block = rows + row * row_bytes + b * Q6_K_BLOCK_BYTES;
                std::memcpy(out + 2 * r, block + 208, 2);
                for (size_t s = 0; s < 16; ++s)
                {
                    out[Q6_K_PACKED_SCALES + s * Q8_0_GROUP + r] = block[192 + s];
                }
                for (size_t k = 0; k < Q6_K_BLOCK; ++k)
                {
                    const uint8_t value = q6_k_value(block, k);
                    bool high = false;
                    const size_t at = Q6_K_PACKED_QUANTS + k_packed_nibble_byte(r, k, high);
                    out[at] = static_cast<uint8_t>(out[at] | (high ? (value & 0x0f) << 4 : (value & 0x0f)));
                    const size_t high_at = Q6_K_PACKED_HIGH + (k / 16) * 64 + r * 4 + (k % 4);
                    const unsigned shift = 2 * static_cast<unsigned>((k % 16) / 4);
                    out[high_at] = static_cast<uint8_t>(out[high_at] | ((value >> 4) << shift));
                }
            }
            out += Q6_K_GROUP_BLOCK_BYTES;
        }
    }
}

void row_q6_k_packed(const uint8_t* packed, size_t n_in, size_t row, float* out)
{
    const size_t blocks = n_in / Q6_K_BLOCK;
    const size_t g = row / Q8_0_GROUP;
    const size_t r = row % Q8_0_GROUP;
    const uint8_t* group = packed + g * blocks * Q6_K_GROUP_BLOCK_BYTES;
    for (size_t b = 0; b < blocks; ++b)
    {
        const uint8_t* block = group + b * Q6_K_GROUP_BLOCK_BYTES;
        const float d = half_at(block + 2 * r);
        for (size_t k = 0; k < Q6_K_BLOCK; ++k)
        {
            const float scale = static_cast<float>(static_cast<int8_t>(block[Q6_K_PACKED_SCALES + (k / 16) * Q8_0_GROUP + r]));
            out[b * Q6_K_BLOCK + k] = d * scale * static_cast<float>(q6_k_packed_value(block, r, k) - 32);
        }
    }
}

void matvec_q6_k_packed_scalar(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in,
                               size_t n_out, size_t first, size_t last, float* out)
{
    const size_t blocks = n_in / Q6_K_BLOCK;
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* group = packed + g * blocks * Q6_K_GROUP_BLOCK_BYTES;
        float acc[Q8_0_GROUP] = {};
        for (size_t b = 0; b < blocks; ++b)
        {
            const uint8_t* block = group + b * Q6_K_GROUP_BLOCK_BYTES;
            const int8_t* x = reinterpret_cast<const int8_t*>(xq + b * Q8_K_BLOCK_BYTES + 4);
            float dx = 0.0f;
            std::memcpy(&dx, xq + b * Q8_K_BLOCK_BYTES, 4);
            const int32_t* sums = x_sums + b * (Q8_K_BLOCK / 16);
            for (size_t r = 0; r < Q8_0_GROUP; ++r)
            {
                int32_t sumi = 0;
                for (size_t s = 0; s < 16; ++s)
                {
                    int32_t dot = 0;
                    for (size_t l = 0; l < 16; ++l)
                    {
                        dot += q6_k_packed_value(block, r, s * 16 + l) * x[s * 16 + l];
                    }
                    dot -= 32 * sums[s];
                    sumi += static_cast<int8_t>(block[Q6_K_PACKED_SCALES + s * Q8_0_GROUP + r]) * dot;
                }
                const float d = half_at(block + 2 * r) * dx;
                acc[r] = std::fma(static_cast<float>(sumi), d, acc[r]);
            }
        }
        for (size_t r = 0; r < Q8_0_GROUP && g * Q8_0_GROUP + r < last; ++r)
        {
            out[g * Q8_0_GROUP + r] = acc[r];
        }
    }
    (void)n_out;
}

#if defined(FAISCA_AVX512_VNNI) || defined(FAISCA_AVX_VNNI)

namespace
{

// one input at a time the kernels are bound by the misses in flight, not by compute:
// asking for the group block 8 KiB ahead keeps the memory pipeline full (bench/README.md)
constexpr size_t PREFETCH_AHEAD = 8192;

inline void prefetch_group_block(const uint8_t* block, size_t bytes)
{
    for (size_t line = 0; line < bytes; line += 64)
    {
        _mm_prefetch(reinterpret_cast<const char*>(block + PREFETCH_AHEAD + line), _MM_HINT_T0);
    }
}

}

#endif

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
        prefetch_group_block(block, Q8_0_GROUP_BLOCK_BYTES);
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
            prefetch_group_block(block, Q8_0_GROUP_BLOCK_BYTES);
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
        prefetch_group_block(block, Q4_0_GROUP_BLOCK_BYTES);
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

namespace
{

inline void store_group(float* o, size_t row, size_t last, __m512 acc)
{
    if (row + Q8_0_GROUP <= last)
    {
        _mm512_storeu_ps(o, acc);
    }
    else
    {
        _mm512_mask_storeu_ps(o, static_cast<__mmask16>((1u << (last - row)) - 1), acc);
    }
}

template <size_t TOKENS>
void gemm_group_q5_avx512(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                          size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    const __m512i low_mask = _mm512_set1_epi8(0x0f);
    const __m512i one = _mm512_set1_epi8(0x01);
    __m512 acc[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc[t] = _mm512_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        prefetch_group_block(block, Q5_0_GROUP_BLOCK_BYTES);
        const uint8_t* quants = block + 2 * Q8_0_GROUP;
        const __m512i high = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quants + 4 * 64));
        __m512i w[8];
        for (size_t c = 0; c < 4; ++c)
        {
            const __m512i bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quants + 64 * c));
            const __m512i h_lo = _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(high, static_cast<int>(2 * c)), one), 4);
            const __m512i h_hi = _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(high, static_cast<int>(2 * c + 1)), one), 4);
            w[2 * c] = _mm512_or_si512(_mm512_and_si512(bytes, low_mask), h_lo);
            w[2 * c + 1] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bytes, 4), low_mask), h_hi);
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
            sum = _mm512_sub_epi32(sum, _mm512_set1_epi32(16 * x_sums[t * sums_stride + b]));
            uint16_t dx_bits = 0;
            std::memcpy(&dx_bits, x, 2);
            const __m512 d = _mm512_mul_ps(scales, _mm512_set1_ps(f16_to_f32(dx_bits)));
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), d, acc[t]);
        }
        block += Q5_0_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        store_group(out + t * out_stride + row, row, last, acc[t]);
    }
}

// 8 dot products of 4 consecutive values against one input, the integer part of a sub block of 32
inline __m512i dot_chunks(const __m512i* w, const int8_t* x)
{
    __m512i sum = _mm512_setzero_si512();
    for (size_t j = 0; j < 8; ++j)
    {
        int32_t x4 = 0;
        std::memcpy(&x4, x + 4 * j, 4);
        sum = _mm512_dpbusd_epi32(sum, w[j], _mm512_set1_epi32(x4));
    }
    return sum;
}

template <size_t TOKENS>
void gemm_group_q4_k_avx512(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
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
        prefetch_group_block(block, Q4_K_GROUP_BLOCK_BYTES);
        const uint8_t* quants = block + Q4_K_PACKED_QUANTS;
        __m512i sumi[TOKENS];
        __m512i summ[TOKENS];
        for (size_t t = 0; t < TOKENS; ++t)
        {
            sumi[t] = _mm512_setzero_si512();
            summ[t] = _mm512_setzero_si512();
        }
        for (size_t s = 0; s < 8; ++s)
        {
            __m512i w[8];
            for (size_t c = 0; c < 4; ++c)
            {
                const __m512i bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quants + 64 * (4 * s + c)));
                w[2 * c] = _mm512_and_si512(bytes, low_mask);
                w[2 * c + 1] = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), low_mask);
            }
            const __m512i scale = _mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + Q4_K_PACKED_SCALES + s * Q8_0_GROUP)));
            const __m512i min = _mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + Q4_K_PACKED_MINS + s * Q8_0_GROUP)));
            for (size_t t = 0; t < TOKENS; ++t)
            {
                const int8_t* x = reinterpret_cast<const int8_t*>(xq + t * xq_stride + b * Q8_K_BLOCK_BYTES + 4);
                const int32_t* sums = x_sums + t * sums_stride + b * (Q8_K_BLOCK / 16);
                sumi[t] = _mm512_add_epi32(sumi[t], _mm512_mullo_epi32(dot_chunks(w, x + 32 * s), scale));
                summ[t] = _mm512_add_epi32(summ[t], _mm512_mullo_epi32(min, _mm512_set1_epi32(sums[2 * s] + sums[2 * s + 1])));
            }
        }
        const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block)));
        const __m512 dmin = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 2 * Q8_0_GROUP)));
        for (size_t t = 0; t < TOKENS; ++t)
        {
            float dx = 0.0f;
            std::memcpy(&dx, xq + t * xq_stride + b * Q8_K_BLOCK_BYTES, 4);
            const __m512 vdx = _mm512_set1_ps(dx);
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sumi[t]), _mm512_mul_ps(d, vdx), acc[t]);
            acc[t] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(summ[t]), _mm512_mul_ps(dmin, vdx), acc[t]);
        }
        block += Q4_K_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        store_group(out + t * out_stride + row, row, last, acc[t]);
    }
}

template <size_t TOKENS>
void gemm_group_q6_k_avx512(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                            size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    const __m512i low_mask = _mm512_set1_epi8(0x0f);
    const __m512i mask_03 = _mm512_set1_epi8(0x03);
    const __m512i mask_0c = _mm512_set1_epi8(0x0c);
    const __m512i mask_30 = _mm512_set1_epi8(0x30);
    const __m512i mask_c0 = _mm512_set1_epi8(static_cast<char>(0xc0));
    __m512 acc[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc[t] = _mm512_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        prefetch_group_block(block, Q6_K_GROUP_BLOCK_BYTES);
        const uint8_t* quants = block + Q6_K_PACKED_QUANTS;
        const uint8_t* high = block + Q6_K_PACKED_HIGH;
        __m512i sumi[TOKENS];
        for (size_t t = 0; t < TOKENS; ++t)
        {
            sumi[t] = _mm512_setzero_si512();
        }
        for (size_t s = 0; s < 16; ++s)
        {
            const __m512i h = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(high + 64 * s));
            const __m512i bytes0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quants + 64 * (2 * s)));
            const __m512i bytes1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quants + 64 * (2 * s + 1)));
            __m512i w[4];
            w[0] = _mm512_or_si512(_mm512_and_si512(bytes0, low_mask), _mm512_slli_epi16(_mm512_and_si512(h, mask_03), 4));
            w[1] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bytes0, 4), low_mask),
                                   _mm512_slli_epi16(_mm512_and_si512(h, mask_0c), 2));
            w[2] = _mm512_or_si512(_mm512_and_si512(bytes1, low_mask), _mm512_and_si512(h, mask_30));
            w[3] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bytes1, 4), low_mask),
                                   _mm512_srli_epi16(_mm512_and_si512(h, mask_c0), 2));
            const __m512i scale = _mm512_cvtepi8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + Q6_K_PACKED_SCALES + s * Q8_0_GROUP)));
            for (size_t t = 0; t < TOKENS; ++t)
            {
                const int8_t* x = reinterpret_cast<const int8_t*>(xq + t * xq_stride + b * Q8_K_BLOCK_BYTES + 4) + 16 * s;
                const int32_t* sums = x_sums + t * sums_stride + b * (Q8_K_BLOCK / 16);
                __m512i dot = _mm512_setzero_si512();
                for (size_t j = 0; j < 4; ++j)
                {
                    int32_t x4 = 0;
                    std::memcpy(&x4, x + 4 * j, 4);
                    dot = _mm512_dpbusd_epi32(dot, w[j], _mm512_set1_epi32(x4));
                }
                dot = _mm512_sub_epi32(dot, _mm512_set1_epi32(32 * sums[s]));
                sumi[t] = _mm512_add_epi32(sumi[t], _mm512_mullo_epi32(dot, scale));
            }
        }
        const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block)));
        for (size_t t = 0; t < TOKENS; ++t)
        {
            float dx = 0.0f;
            std::memcpy(&dx, xq + t * xq_stride + b * Q8_K_BLOCK_BYTES, 4);
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sumi[t]), _mm512_mul_ps(d, _mm512_set1_ps(dx)), acc[t]);
        }
        block += Q6_K_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        store_group(out + t * out_stride + row, row, last, acc[t]);
    }
}

template <void (*GEMM)(const uint8_t*, const uint8_t*, size_t, const int32_t*, size_t, size_t, size_t, size_t, float*, size_t),
          void (*GEMV)(const uint8_t*, const uint8_t*, size_t, const int32_t*, size_t, size_t, size_t, size_t, float*, size_t)>
void matmul_groups(size_t group_block_bytes, const uint8_t* packed, const uint8_t* xq, size_t xq_stride,
                   const int32_t* x_sums, size_t sums_stride, size_t n_tokens, size_t blocks, size_t first, size_t last,
                   float* out, size_t out_stride)
{
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* block = packed + g * blocks * group_block_bytes;
        const size_t row = g * Q8_0_GROUP;
        size_t t = 0;
        for (; t + GEMM_TOKENS <= n_tokens; t += GEMM_TOKENS)
        {
            GEMM(block, xq + t * xq_stride, xq_stride, x_sums + t * sums_stride, sums_stride, blocks, row, last,
                 out + t * out_stride, out_stride);
        }
        for (; t < n_tokens; ++t)
        {
            GEMV(block, xq + t * xq_stride, xq_stride, x_sums + t * sums_stride, sums_stride, blocks, row, last,
                 out + t * out_stride, out_stride);
        }
    }
}

}

void matmul_q5_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    matmul_groups<gemm_group_q5_avx512<GEMM_TOKENS>, gemm_group_q5_avx512<1>>(
        Q5_0_GROUP_BLOCK_BYTES, packed, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in / Q5_0_BLOCK, first, last,
        out, out_stride);
    (void)n_out;
}

void matmul_q4_k_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    matmul_groups<gemm_group_q4_k_avx512<GEMM_TOKENS>, gemm_group_q4_k_avx512<1>>(
        Q4_K_GROUP_BLOCK_BYTES, packed, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in / Q4_K_BLOCK, first, last,
        out, out_stride);
    (void)n_out;
}

void matmul_q6_k_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    matmul_groups<gemm_group_q6_k_avx512<GEMM_TOKENS>, gemm_group_q6_k_avx512<1>>(
        Q6_K_GROUP_BLOCK_BYTES, packed, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in / Q6_K_BLOCK, first, last,
        out, out_stride);
    (void)n_out;
}

#elif defined(FAISCA_AVX_VNNI)

namespace
{

// group block chunks unpacked once and reused for this many inputs; 16 ymm registers
// hold the two halves of the sums for 4 inputs plus the unpacked chunk
constexpr size_t GEMM_TOKENS_AVX = 4;

inline void store_group_halves(float* o, size_t row, size_t last, __m256 lo, __m256 hi)
{
    float values[Q8_0_GROUP];
    _mm256_storeu_ps(values, lo);
    _mm256_storeu_ps(values + 8, hi);
    const size_t count = row + Q8_0_GROUP <= last ? Q8_0_GROUP : last - row;
    std::memcpy(o, values, count * sizeof(float));
}

inline __m256 half_scales(const uint8_t* at)
{
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(at)));
}

inline __m256i broadcast_4(const uint8_t* at)
{
    int32_t x4 = 0;
    std::memcpy(&x4, at, 4);
    return _mm256_set1_epi32(x4);
}

inline float f16_at(const uint8_t* at)
{
    uint16_t bits = 0;
    std::memcpy(&bits, at, 2);
    return f16_to_f32(bits);
}

inline float f32_at(const uint8_t* at)
{
    float value = 0.0f;
    std::memcpy(&value, at, 4);
    return value;
}

inline void accumulate_block(__m256& acc_lo, __m256& acc_hi, __m256i sum_lo, __m256i sum_hi, int32_t correction,
                             __m256 scales_lo, __m256 scales_hi, float dx)
{
    const __m256i c = _mm256_set1_epi32(correction);
    const __m256 vdx = _mm256_set1_ps(dx);
    acc_lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(sum_lo, c)), _mm256_mul_ps(scales_lo, vdx), acc_lo);
    acc_hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(sum_hi, c)), _mm256_mul_ps(scales_hi, vdx), acc_hi);
}

template <size_t TOKENS>
void gemm_group_q8_avx(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                       size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    __m256 acc_lo[TOKENS];
    __m256 acc_hi[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc_lo[t] = _mm256_setzero_ps();
        acc_hi[t] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        prefetch_group_block(block, Q8_0_GROUP_BLOCK_BYTES);
        const uint8_t* quants = block + 2 * Q8_0_GROUP;
        const uint8_t* x = xq + b * Q8_0_BLOCK_BYTES;
        __m256i sum_lo[TOKENS];
        __m256i sum_hi[TOKENS];
        for (size_t t = 0; t < TOKENS; ++t)
        {
            sum_lo[t] = _mm256_setzero_si256();
            sum_hi[t] = _mm256_setzero_si256();
        }
        for (size_t j = 0; j < 8; ++j)
        {
            const __m256i w_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * j));
            const __m256i w_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * j + 32));
            for (size_t t = 0; t < TOKENS; ++t)
            {
                const __m256i xb = broadcast_4(x + t * xq_stride + 2 + 4 * j);
                sum_lo[t] = _mm256_dpbusd_avx_epi32(sum_lo[t], w_lo, xb);
                sum_hi[t] = _mm256_dpbusd_avx_epi32(sum_hi[t], w_hi, xb);
            }
        }
        const __m256 scales_lo = half_scales(block);
        const __m256 scales_hi = half_scales(block + 16);
        for (size_t t = 0; t < TOKENS; ++t)
        {
            accumulate_block(acc_lo[t], acc_hi[t], sum_lo[t], sum_hi[t], 128 * x_sums[t * sums_stride + b], scales_lo,
                             scales_hi, f16_at(x + t * xq_stride));
        }
        block += Q8_0_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        store_group_halves(out + t * out_stride + row, row, last, acc_lo[t], acc_hi[t]);
    }
}

template <size_t TOKENS>
void gemm_group_q4_avx(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                       size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256 acc_lo[TOKENS];
    __m256 acc_hi[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc_lo[t] = _mm256_setzero_ps();
        acc_hi[t] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        prefetch_group_block(block, Q4_0_GROUP_BLOCK_BYTES);
        const uint8_t* quants = block + 2 * Q8_0_GROUP;
        const uint8_t* x = xq + b * Q8_0_BLOCK_BYTES;
        __m256i sum_lo[TOKENS];
        __m256i sum_hi[TOKENS];
        for (size_t t = 0; t < TOKENS; ++t)
        {
            sum_lo[t] = _mm256_setzero_si256();
            sum_hi[t] = _mm256_setzero_si256();
        }
        for (size_t c = 0; c < 4; ++c)
        {
            const __m256i bytes_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * c));
            const __m256i bytes_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * c + 32));
            const __m256i w_lo_a = _mm256_and_si256(bytes_lo, low_mask);
            const __m256i w_hi_a = _mm256_and_si256(bytes_hi, low_mask);
            const __m256i w_lo_b = _mm256_and_si256(_mm256_srli_epi16(bytes_lo, 4), low_mask);
            const __m256i w_hi_b = _mm256_and_si256(_mm256_srli_epi16(bytes_hi, 4), low_mask);
            for (size_t t = 0; t < TOKENS; ++t)
            {
                const __m256i xa = broadcast_4(x + t * xq_stride + 2 + 8 * c);
                const __m256i xb = broadcast_4(x + t * xq_stride + 6 + 8 * c);
                sum_lo[t] = _mm256_dpbusd_avx_epi32(sum_lo[t], w_lo_a, xa);
                sum_hi[t] = _mm256_dpbusd_avx_epi32(sum_hi[t], w_hi_a, xa);
                sum_lo[t] = _mm256_dpbusd_avx_epi32(sum_lo[t], w_lo_b, xb);
                sum_hi[t] = _mm256_dpbusd_avx_epi32(sum_hi[t], w_hi_b, xb);
            }
        }
        const __m256 scales_lo = half_scales(block);
        const __m256 scales_hi = half_scales(block + 16);
        for (size_t t = 0; t < TOKENS; ++t)
        {
            accumulate_block(acc_lo[t], acc_hi[t], sum_lo[t], sum_hi[t], 8 * x_sums[t * sums_stride + b], scales_lo,
                             scales_hi, f16_at(x + t * xq_stride));
        }
        block += Q4_0_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        store_group_halves(out + t * out_stride + row, row, last, acc_lo[t], acc_hi[t]);
    }
}

template <size_t TOKENS>
void gemm_group_q5_avx(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                       size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i one = _mm256_set1_epi8(0x01);
    __m256 acc_lo[TOKENS];
    __m256 acc_hi[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc_lo[t] = _mm256_setzero_ps();
        acc_hi[t] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        prefetch_group_block(block, Q5_0_GROUP_BLOCK_BYTES);
        const uint8_t* quants = block + 2 * Q8_0_GROUP;
        const uint8_t* x = xq + b * Q8_0_BLOCK_BYTES;
        const __m256i high_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 4 * 64));
        const __m256i high_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 4 * 64 + 32));
        __m256i sum_lo[TOKENS];
        __m256i sum_hi[TOKENS];
        for (size_t t = 0; t < TOKENS; ++t)
        {
            sum_lo[t] = _mm256_setzero_si256();
            sum_hi[t] = _mm256_setzero_si256();
        }
        for (size_t c = 0; c < 4; ++c)
        {
            const __m256i bytes_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * c));
            const __m256i bytes_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * c + 32));
            const int shift_a = static_cast<int>(2 * c);
            const int shift_b = shift_a + 1;
            const __m256i w_lo_a = _mm256_or_si256(_mm256_and_si256(bytes_lo, low_mask),
                                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high_lo, shift_a), one), 4));
            const __m256i w_hi_a = _mm256_or_si256(_mm256_and_si256(bytes_hi, low_mask),
                                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high_hi, shift_a), one), 4));
            const __m256i w_lo_b = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bytes_lo, 4), low_mask),
                                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high_lo, shift_b), one), 4));
            const __m256i w_hi_b = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bytes_hi, 4), low_mask),
                                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high_hi, shift_b), one), 4));
            for (size_t t = 0; t < TOKENS; ++t)
            {
                const __m256i xa = broadcast_4(x + t * xq_stride + 2 + 8 * c);
                const __m256i xb = broadcast_4(x + t * xq_stride + 6 + 8 * c);
                sum_lo[t] = _mm256_dpbusd_avx_epi32(sum_lo[t], w_lo_a, xa);
                sum_hi[t] = _mm256_dpbusd_avx_epi32(sum_hi[t], w_hi_a, xa);
                sum_lo[t] = _mm256_dpbusd_avx_epi32(sum_lo[t], w_lo_b, xb);
                sum_hi[t] = _mm256_dpbusd_avx_epi32(sum_hi[t], w_hi_b, xb);
            }
        }
        const __m256 scales_lo = half_scales(block);
        const __m256 scales_hi = half_scales(block + 16);
        for (size_t t = 0; t < TOKENS; ++t)
        {
            accumulate_block(acc_lo[t], acc_hi[t], sum_lo[t], sum_hi[t], 16 * x_sums[t * sums_stride + b], scales_lo,
                             scales_hi, f16_at(x + t * xq_stride));
        }
        block += Q5_0_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        store_group_halves(out + t * out_stride + row, row, last, acc_lo[t], acc_hi[t]);
    }
}

template <size_t TOKENS>
void gemm_group_q4_k_avx(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                         size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256 acc_lo[TOKENS];
    __m256 acc_hi[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc_lo[t] = _mm256_setzero_ps();
        acc_hi[t] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        prefetch_group_block(block, Q4_K_GROUP_BLOCK_BYTES);
        const uint8_t* quants = block + Q4_K_PACKED_QUANTS;
        const uint8_t* x = xq + b * Q8_K_BLOCK_BYTES;
        __m256i sumi_lo[TOKENS];
        __m256i sumi_hi[TOKENS];
        __m256i summ_lo[TOKENS];
        __m256i summ_hi[TOKENS];
        for (size_t t = 0; t < TOKENS; ++t)
        {
            sumi_lo[t] = _mm256_setzero_si256();
            sumi_hi[t] = _mm256_setzero_si256();
            summ_lo[t] = _mm256_setzero_si256();
            summ_hi[t] = _mm256_setzero_si256();
        }
        for (size_t s = 0; s < 8; ++s)
        {
            __m256i dot_lo[TOKENS];
            __m256i dot_hi[TOKENS];
            for (size_t t = 0; t < TOKENS; ++t)
            {
                dot_lo[t] = _mm256_setzero_si256();
                dot_hi[t] = _mm256_setzero_si256();
            }
            for (size_t c = 0; c < 4; ++c)
            {
                const uint8_t* chunk = quants + 64 * (4 * s + c);
                const __m256i bytes_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(chunk));
                const __m256i bytes_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(chunk + 32));
                const __m256i w_lo_a = _mm256_and_si256(bytes_lo, low_mask);
                const __m256i w_hi_a = _mm256_and_si256(bytes_hi, low_mask);
                const __m256i w_lo_b = _mm256_and_si256(_mm256_srli_epi16(bytes_lo, 4), low_mask);
                const __m256i w_hi_b = _mm256_and_si256(_mm256_srli_epi16(bytes_hi, 4), low_mask);
                for (size_t t = 0; t < TOKENS; ++t)
                {
                    const uint8_t* xt = x + t * xq_stride + 4 + 32 * s + 8 * c;
                    const __m256i xa = broadcast_4(xt);
                    const __m256i xb = broadcast_4(xt + 4);
                    dot_lo[t] = _mm256_dpbusd_avx_epi32(dot_lo[t], w_lo_a, xa);
                    dot_hi[t] = _mm256_dpbusd_avx_epi32(dot_hi[t], w_hi_a, xa);
                    dot_lo[t] = _mm256_dpbusd_avx_epi32(dot_lo[t], w_lo_b, xb);
                    dot_hi[t] = _mm256_dpbusd_avx_epi32(dot_hi[t], w_hi_b, xb);
                }
            }
            const uint8_t* scales = block + Q4_K_PACKED_SCALES + s * Q8_0_GROUP;
            const uint8_t* mins = block + Q4_K_PACKED_MINS + s * Q8_0_GROUP;
            const __m256i scale_lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(scales)));
            const __m256i scale_hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(scales + 8)));
            const __m256i min_lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(mins)));
            const __m256i min_hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(mins + 8)));
            for (size_t t = 0; t < TOKENS; ++t)
            {
                const int32_t* sums = x_sums + t * sums_stride + b * (Q8_K_BLOCK / 16);
                const __m256i sum32 = _mm256_set1_epi32(sums[2 * s] + sums[2 * s + 1]);
                sumi_lo[t] = _mm256_add_epi32(sumi_lo[t], _mm256_mullo_epi32(dot_lo[t], scale_lo));
                sumi_hi[t] = _mm256_add_epi32(sumi_hi[t], _mm256_mullo_epi32(dot_hi[t], scale_hi));
                summ_lo[t] = _mm256_add_epi32(summ_lo[t], _mm256_mullo_epi32(min_lo, sum32));
                summ_hi[t] = _mm256_add_epi32(summ_hi[t], _mm256_mullo_epi32(min_hi, sum32));
            }
        }
        const __m256 d_lo = half_scales(block);
        const __m256 d_hi = half_scales(block + 16);
        const __m256 dmin_lo = half_scales(block + 2 * Q8_0_GROUP);
        const __m256 dmin_hi = half_scales(block + 2 * Q8_0_GROUP + 16);
        for (size_t t = 0; t < TOKENS; ++t)
        {
            const __m256 vdx = _mm256_set1_ps(f32_at(x + t * xq_stride));
            acc_lo[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi_lo[t]), _mm256_mul_ps(d_lo, vdx), acc_lo[t]);
            acc_hi[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi_hi[t]), _mm256_mul_ps(d_hi, vdx), acc_hi[t]);
            acc_lo[t] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(summ_lo[t]), _mm256_mul_ps(dmin_lo, vdx), acc_lo[t]);
            acc_hi[t] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(summ_hi[t]), _mm256_mul_ps(dmin_hi, vdx), acc_hi[t]);
        }
        block += Q4_K_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        store_group_halves(out + t * out_stride + row, row, last, acc_lo[t], acc_hi[t]);
    }
}

template <size_t TOKENS>
void gemm_group_q6_k_avx(const uint8_t* block, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                         size_t sums_stride, size_t blocks, size_t row, size_t last, float* out, size_t out_stride)
{
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i mask_03 = _mm256_set1_epi8(0x03);
    const __m256i mask_0c = _mm256_set1_epi8(0x0c);
    const __m256i mask_30 = _mm256_set1_epi8(0x30);
    const __m256i mask_c0 = _mm256_set1_epi8(static_cast<char>(0xc0));
    __m256 acc_lo[TOKENS];
    __m256 acc_hi[TOKENS];
    for (size_t t = 0; t < TOKENS; ++t)
    {
        acc_lo[t] = _mm256_setzero_ps();
        acc_hi[t] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < blocks; ++b)
    {
        prefetch_group_block(block, Q6_K_GROUP_BLOCK_BYTES);
        const uint8_t* quants = block + Q6_K_PACKED_QUANTS;
        const uint8_t* high = block + Q6_K_PACKED_HIGH;
        const uint8_t* x = xq + b * Q8_K_BLOCK_BYTES;
        __m256i sumi_lo[TOKENS];
        __m256i sumi_hi[TOKENS];
        for (size_t t = 0; t < TOKENS; ++t)
        {
            sumi_lo[t] = _mm256_setzero_si256();
            sumi_hi[t] = _mm256_setzero_si256();
        }
        for (size_t s = 0; s < 16; ++s)
        {
            __m256i w[2][4];
            for (size_t half = 0; half < 2; ++half)
            {
                const size_t offset = 32 * half;
                const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(high + 64 * s + offset));
                const __m256i bytes0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * (2 * s) + offset));
                const __m256i bytes1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + 64 * (2 * s + 1) + offset));
                w[half][0] = _mm256_or_si256(_mm256_and_si256(bytes0, low_mask), _mm256_slli_epi16(_mm256_and_si256(h, mask_03), 4));
                w[half][1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bytes0, 4), low_mask),
                                             _mm256_slli_epi16(_mm256_and_si256(h, mask_0c), 2));
                w[half][2] = _mm256_or_si256(_mm256_and_si256(bytes1, low_mask), _mm256_and_si256(h, mask_30));
                w[half][3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bytes1, 4), low_mask),
                                             _mm256_srli_epi16(_mm256_and_si256(h, mask_c0), 2));
            }
            const uint8_t* scales = block + Q6_K_PACKED_SCALES + s * Q8_0_GROUP;
            const __m256i scale_lo = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(scales)));
            const __m256i scale_hi = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(scales + 8)));
            for (size_t t = 0; t < TOKENS; ++t)
            {
                const uint8_t* xt = x + t * xq_stride + 4 + 16 * s;
                __m256i dot_lo = _mm256_setzero_si256();
                __m256i dot_hi = _mm256_setzero_si256();
                for (size_t j = 0; j < 4; ++j)
                {
                    const __m256i xb = broadcast_4(xt + 4 * j);
                    dot_lo = _mm256_dpbusd_avx_epi32(dot_lo, w[0][j], xb);
                    dot_hi = _mm256_dpbusd_avx_epi32(dot_hi, w[1][j], xb);
                }
                const __m256i correction = _mm256_set1_epi32(32 * x_sums[t * sums_stride + b * (Q8_K_BLOCK / 16) + s]);
                sumi_lo[t] = _mm256_add_epi32(sumi_lo[t], _mm256_mullo_epi32(_mm256_sub_epi32(dot_lo, correction), scale_lo));
                sumi_hi[t] = _mm256_add_epi32(sumi_hi[t], _mm256_mullo_epi32(_mm256_sub_epi32(dot_hi, correction), scale_hi));
            }
        }
        const __m256 d_lo = half_scales(block);
        const __m256 d_hi = half_scales(block + 16);
        for (size_t t = 0; t < TOKENS; ++t)
        {
            const __m256 vdx = _mm256_set1_ps(f32_at(x + t * xq_stride));
            acc_lo[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi_lo[t]), _mm256_mul_ps(d_lo, vdx), acc_lo[t]);
            acc_hi[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi_hi[t]), _mm256_mul_ps(d_hi, vdx), acc_hi[t]);
        }
        block += Q6_K_GROUP_BLOCK_BYTES;
    }
    for (size_t t = 0; t < TOKENS; ++t)
    {
        store_group_halves(out + t * out_stride + row, row, last, acc_lo[t], acc_hi[t]);
    }
}

template <void (*GEMM)(const uint8_t*, const uint8_t*, size_t, const int32_t*, size_t, size_t, size_t, size_t, float*, size_t),
          void (*GEMV)(const uint8_t*, const uint8_t*, size_t, const int32_t*, size_t, size_t, size_t, size_t, float*, size_t)>
void matmul_groups_avx(size_t group_block_bytes, const uint8_t* packed, const uint8_t* xq, size_t xq_stride,
                       const int32_t* x_sums, size_t sums_stride, size_t n_tokens, size_t blocks, size_t first,
                       size_t last, float* out, size_t out_stride)
{
    for (size_t g = first / Q8_0_GROUP; g * Q8_0_GROUP < last; ++g)
    {
        const uint8_t* block = packed + g * blocks * group_block_bytes;
        const size_t row = g * Q8_0_GROUP;
        size_t t = 0;
        for (; t + GEMM_TOKENS_AVX <= n_tokens; t += GEMM_TOKENS_AVX)
        {
            GEMM(block, xq + t * xq_stride, xq_stride, x_sums + t * sums_stride, sums_stride, blocks, row, last,
                 out + t * out_stride, out_stride);
        }
        for (; t < n_tokens; ++t)
        {
            GEMV(block, xq + t * xq_stride, xq_stride, x_sums + t * sums_stride, sums_stride, blocks, row, last,
                 out + t * out_stride, out_stride);
        }
    }
}

}

void matmul_q5_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    matmul_groups_avx<gemm_group_q5_avx<GEMM_TOKENS_AVX>, gemm_group_q5_avx<1>>(
        Q5_0_GROUP_BLOCK_BYTES, packed, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in / Q5_0_BLOCK, first, last,
        out, out_stride);
    (void)n_out;
}

void matmul_q4_k_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    matmul_groups_avx<gemm_group_q4_k_avx<GEMM_TOKENS_AVX>, gemm_group_q4_k_avx<1>>(
        Q4_K_GROUP_BLOCK_BYTES, packed, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in / Q4_K_BLOCK, first, last,
        out, out_stride);
    (void)n_out;
}

void matmul_q6_k_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    matmul_groups_avx<gemm_group_q6_k_avx<GEMM_TOKENS_AVX>, gemm_group_q6_k_avx<1>>(
        Q6_K_GROUP_BLOCK_BYTES, packed, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in / Q6_K_BLOCK, first, last,
        out, out_stride);
    (void)n_out;
}

void matmul_q4_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    matmul_groups_avx<gemm_group_q4_avx<GEMM_TOKENS_AVX>, gemm_group_q4_avx<1>>(
        Q4_0_GROUP_BLOCK_BYTES, packed, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in / Q4_0_BLOCK, first, last,
        out, out_stride);
    (void)n_out;
}

void matmul_q8_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    matmul_groups_avx<gemm_group_q8_avx<GEMM_TOKENS_AVX>, gemm_group_q8_avx<1>>(
        Q8_0_GROUP_BLOCK_BYTES, packed, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in / Q8_0_BLOCK, first, last,
        out, out_stride);
    (void)n_out;
}

void matvec_q8_0_packed(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in, size_t n_out,
                        size_t first, size_t last, float* out)
{
    matmul_groups_avx<gemm_group_q8_avx<GEMM_TOKENS_AVX>, gemm_group_q8_avx<1>>(
        Q8_0_GROUP_BLOCK_BYTES, packed, xq, 0, x_sums, 0, 1, n_in / Q8_0_BLOCK, first, last, out, 0);
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

#if !defined(FAISCA_AVX512_VNNI) && !defined(FAISCA_AVX_VNNI)

void matmul_q5_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    for (size_t t = 0; t < n_tokens; ++t)
    {
        matvec_q5_0_packed_scalar(packed, xq + t * xq_stride, x_sums + t * sums_stride, n_in, n_out, first, last,
                                  out + t * out_stride);
    }
}

void matmul_q4_k_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    for (size_t t = 0; t < n_tokens; ++t)
    {
        matvec_q4_k_packed_scalar(packed, xq + t * xq_stride, x_sums + t * sums_stride, n_in, n_out, first, last,
                                  out + t * out_stride);
    }
}

void matmul_q6_k_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride)
{
    for (size_t t = 0; t < n_tokens; ++t)
    {
        matvec_q6_k_packed_scalar(packed, xq + t * xq_stride, x_sums + t * sums_stride, n_in, n_out, first, last,
                                  out + t * out_stride);
    }
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

void quantize_q8_0_scalar(const float* x, size_t n, uint8_t* out)
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

void rope_norm(float* x, size_t n_head, size_t head_dim, size_t n_rot, uint32_t pos, float freq_base,
               const float* factors)
{
    const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));
    for (size_t h = 0; h < n_head; ++h)
    {
        float* head = x + h * head_dim;
        float theta = static_cast<float>(pos);
        for (size_t i = 0; i < n_rot; i += 2)
        {
            const float angle = factors != nullptr ? theta / factors[i / 2] : theta;
            const float cos_theta = std::cos(angle);
            const float sin_theta = std::sin(angle);
            const float x0 = head[i];
            const float x1 = head[i + 1];
            head[i] = x0 * cos_theta - x1 * sin_theta;
            head[i + 1] = x0 * sin_theta + x1 * cos_theta;
            theta *= theta_scale;
        }
    }
}

void rope_neox(float* x, size_t n_head, size_t head_dim, size_t n_rot, uint32_t pos, float freq_base)
{
    const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));
    const size_t half = n_rot / 2;
    for (size_t h = 0; h < n_head; ++h)
    {
        float* head = x + h * head_dim;
        float theta = static_cast<float>(pos);
        for (size_t i = 0; i < half; ++i)
        {
            const float cos_theta = std::cos(theta);
            const float sin_theta = std::sin(theta);
            const float x0 = head[i];
            const float x1 = head[i + half];
            head[i] = x0 * cos_theta - x1 * sin_theta;
            head[i + half] = x0 * sin_theta + x1 * cos_theta;
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

void f32_to_f16_values_scalar(const float* x, size_t n, uint16_t* out)
{
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = f32_to_f16(x[i]);
    }
}

namespace
{

inline float to_f32(float value)
{
    return value;
}

inline float to_f32(uint16_t bits)
{
    return f16_to_f32(bits);
}

template <typename Value>
void attention_head_scalar_impl(const float* q, const Value* k_tiles, const Value* v, size_t v_stride, size_t head_dim,
                                size_t n_past, float scale, float* scores, float* out, size_t skip)
{
    for (size_t p = 0; p < n_past; ++p)
    {
        float s = 0.0f;
        for (size_t d = 0; d < head_dim; ++d)
        {
            s += q[d] * to_f32(k_tiles[key_index(head_dim, p, d)]);
        }
        scores[p] = p < skip ? -INFINITY : s * scale;
    }
    softmax(scores, n_past);
    for (size_t d = 0; d < head_dim; ++d)
    {
        out[d] = 0.0f;
    }
    for (size_t p = 0; p < n_past; ++p)
    {
        const float weight = scores[p];
        const Value* row = v + p * v_stride;
        for (size_t d = 0; d < head_dim; ++d)
        {
            out[d] += weight * to_f32(row[d]);
        }
    }
}

}

void attention_head_scalar(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                           size_t n_past, float scale, float* scores, float* out, size_t skip)
{
    attention_head_scalar_impl(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out, skip);
}

void attention_head_f16_scalar(const float* q, const uint16_t* k_tiles, const uint16_t* v, size_t v_stride,
                               size_t head_dim, size_t n_past, float scale, float* scores, float* out, size_t skip)
{
    attention_head_scalar_impl(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out, skip);
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

inline __m256 load_8(const float* p)
{
    return _mm256_loadu_ps(p);
}

inline __m256 load_8(const uint16_t* p)
{
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
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

inline __m512 load_16(const float* p)
{
    return _mm512_loadu_ps(p);
}

inline __m512 load_16(const uint16_t* p)
{
    return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}
#endif

}

void f32_to_f16_values(const float* x, size_t n, uint16_t* out)
{
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i),
                            _mm512_cvtps_ph(_mm512_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
#endif
    for (; i + 8 <= n; i += 8)
    {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    for (; i < n; ++i)
    {
        out[i] = f32_to_f16(x[i]);
    }
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

void swiglu(const float* gate, const float* up, size_t n, float* out)
{
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16)
    {
        const __m512 g = _mm512_loadu_ps(gate + i);
        const __m512 e = exp_avx512(_mm512_sub_ps(_mm512_setzero_ps(), g));
        const __m512 silu = _mm512_div_ps(g, _mm512_add_ps(_mm512_set1_ps(1.0f), e));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(silu, _mm512_loadu_ps(up + i)));
    }
#endif
    for (; i + 8 <= n; i += 8)
    {
        const __m256 g = _mm256_loadu_ps(gate + i);
        const __m256 e = exp_avx2(_mm256_sub_ps(_mm256_setzero_ps(), g));
        const __m256 silu = _mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(silu, _mm256_loadu_ps(up + i)));
    }
    if (i < n)
    {
        swiglu_scalar(gate + i, up + i, n - i, out + i);
    }
}

namespace
{

#if defined(__AVX512F__)
inline void store_int8_16(uint8_t* out, __m512i values)
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm512_cvtepi32_epi8(values));
}
#endif

inline void store_int8_8(uint8_t* out, __m256i values)
{
    alignas(32) int32_t v[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), values);
    for (size_t k = 0; k < 8; ++k)
    {
        out[k] = static_cast<uint8_t>(static_cast<int8_t>(v[k]));
    }
}

}

// the same arithmetic as the scalar version: x * id rounded to nearest even, so the bytes are identical
void quantize_q8_0(const float* x, size_t n, uint8_t* out)
{
    for (size_t i = 0; i < n; i += Q8_0_BLOCK)
    {
#if defined(__AVX512F__)
        const __m512 v0 = _mm512_loadu_ps(x + i);
        const __m512 v1 = _mm512_loadu_ps(x + i + 16);
        const __m512 amax_v = _mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1));
        const float amax = _mm512_reduce_max_ps(amax_v);
        const float d = amax / 127.0f;
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        const uint16_t bits = f32_to_f16(d);
        std::memcpy(out, &bits, 2);
        const __m512 vid = _mm512_set1_ps(id);
        store_int8_16(out + 2, _mm512_cvtps_epi32(_mm512_mul_ps(v0, vid)));
        store_int8_16(out + 18, _mm512_cvtps_epi32(_mm512_mul_ps(v1, vid)));
#else
        const __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 amax_v = _mm256_setzero_ps();
        for (size_t j = 0; j < Q8_0_BLOCK; j += 8)
        {
            amax_v = _mm256_max_ps(amax_v, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + i + j)));
        }
        const __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(amax_v), _mm256_extractf128_ps(amax_v, 1));
        const __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        const float amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_movehdup_ps(m2)));
        const float d = amax / 127.0f;
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        const uint16_t bits = f32_to_f16(d);
        std::memcpy(out, &bits, 2);
        const __m256 vid = _mm256_set1_ps(id);
        for (size_t j = 0; j < Q8_0_BLOCK; j += 8)
        {
            store_int8_8(out + 2 + j, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + j), vid)));
        }
#endif
        out += Q8_0_BLOCK_BYTES;
    }
}

// as quantize_q8_k_scalar: the extreme is the first value whose magnitude is the largest
void quantize_q8_k(const float* x, size_t n, uint8_t* out)
{
    for (size_t i = 0; i < n; i += Q8_K_BLOCK)
    {
        const float* block_x = x + i;
#if defined(__AVX512F__)
        __m512 amax_v = _mm512_setzero_ps();
        for (size_t j = 0; j < Q8_K_BLOCK; j += 16)
        {
            amax_v = _mm512_max_ps(amax_v, _mm512_abs_ps(_mm512_loadu_ps(block_x + j)));
        }
        const float amax = _mm512_reduce_max_ps(amax_v);
#else
        const __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 amax_v = _mm256_setzero_ps();
        for (size_t j = 0; j < Q8_K_BLOCK; j += 8)
        {
            amax_v = _mm256_max_ps(amax_v, _mm256_andnot_ps(sign, _mm256_loadu_ps(block_x + j)));
        }
        const __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(amax_v), _mm256_extractf128_ps(amax_v, 1));
        const __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        const float amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_movehdup_ps(m2)));
#endif
        if (amax == 0.0f)
        {
            std::memset(out, 0, Q8_K_BLOCK_BYTES);
            out += Q8_K_BLOCK_BYTES;
            continue;
        }
        float max = 0.0f;
        for (size_t j = 0; j < Q8_K_BLOCK; ++j)
        {
            if (std::fabs(block_x[j]) == amax)
            {
                max = block_x[j];
                break;
            }
        }
        const float iscale = -127.0f / max;
        const float d = 1.0f / iscale;
        std::memcpy(out, &d, 4);
#if defined(__AVX512F__)
        const __m512 vscale = _mm512_set1_ps(iscale);
        const __m512i limit = _mm512_set1_epi32(127);
        for (size_t j = 0; j < Q8_K_BLOCK; j += 16)
        {
            const __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(block_x + j), vscale));
            store_int8_16(out + 4 + j, _mm512_min_epi32(q, limit));
        }
#else
        const __m256 vscale = _mm256_set1_ps(iscale);
        const __m256i limit = _mm256_set1_epi32(127);
        for (size_t j = 0; j < Q8_K_BLOCK; j += 8)
        {
            const __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(block_x + j), vscale));
            store_int8_8(out + 4 + j, _mm256_min_epi32(q, limit));
        }
#endif
        out += Q8_K_BLOCK_BYTES;
    }
}

namespace
{

// the softmax denominator: 16 partial sums by position modulo 16 (lo holds 0 to 7, hi 8 to
// 15) reduced by one fixed tree, so every build gets the same bits
inline float softmax_inverse(__m256 lo, __m256 hi)
{
    const __m256 t3 = _mm256_add_ps(hi, lo);
    const __m128 t6 = _mm_add_ps(_mm256_extractf128_ps(t3, 1), _mm256_castps256_ps128(t3));
    const __m128 t8 = _mm_add_ps(t6, _mm_shuffle_ps(t6, t6, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128 t10 = _mm_add_ps(t8, _mm_shuffle_ps(t8, t8, _MM_SHUFFLE(0, 1, 0, 1)));
    return 1.0f / _mm_cvtss_f32(t10);
}

}

#if defined(__AVX512F__)

namespace
{

template <typename Value>
void attention_head_impl(const float* q, const Value* k_tiles, const Value* v, size_t v_stride, size_t head_dim,
                         size_t n_past, float scale, float* scores, float* out, size_t skip)
{
    if (head_dim % 16 != 0)
    {
        attention_head_scalar_impl(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out, skip);
        return;
    }

    // scores for 16 positions at a time: head_dim broadcast FMAs over one contiguous tile, no reductions
    const __m512 vscale = _mm512_set1_ps(scale);
    const __mmask16 skipped = static_cast<__mmask16>((1u << skip) - 1u);
    __m512 vmax = _mm512_set1_ps(-INFINITY);
    for (size_t p = 0; p < n_past; p += 16)
    {
        __mmask16 mask = static_cast<__mmask16>(p + 16 <= n_past ? 0xffffu : (1u << (n_past - p)) - 1u);
        const Value* tile = k_tiles + (p / KEY_TILE) * head_dim * KEY_TILE;
        __m512 s = _mm512_setzero_ps();
        for (size_t d = 0; d < head_dim; ++d)
        {
            s = _mm512_fmadd_ps(_mm512_set1_ps(q[d]), load_16(tile + d * KEY_TILE), s);
        }
        s = _mm512_mul_ps(s, vscale);
        if (p == 0)
        {
            mask = static_cast<__mmask16>(mask & ~skipped);
            s = _mm512_mask_mov_ps(s, skipped, _mm512_set1_ps(-INFINITY));
        }
        _mm512_storeu_ps(scores + p, s);
        vmax = _mm512_mask_max_ps(vmax, mask, vmax, s);
    }
    const float max = _mm512_reduce_max_ps(vmax);

    __m512 vsum = _mm512_setzero_ps();
    for (size_t p = 0; p < n_past; p += 16)
    {
        __mmask16 mask = static_cast<__mmask16>(p + 16 <= n_past ? 0xffffu : (1u << (n_past - p)) - 1u);
        if (p == 0)
        {
            mask = static_cast<__mmask16>(mask & ~skipped);
        }
        const __m512 e = exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(scores + p), _mm512_set1_ps(max)));
        _mm512_storeu_ps(scores + p, e);
        vsum = _mm512_mask_add_ps(vsum, mask, vsum, e);
    }
    const float inverse = softmax_inverse(_mm512_castps512_ps256(vsum),
                                          _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vsum), 1)));

    for (size_t d0 = 0; d0 < head_dim; d0 += 64)
    {
        const size_t width = head_dim - d0 < 64 ? head_dim - d0 : 64;
        __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
        for (size_t p = 0; p < n_past; ++p)
        {
            const __m512 weight = _mm512_set1_ps(scores[p]);
            const Value* row = v + p * v_stride + d0;
            for (size_t c = 0; c < width / 16; ++c)
            {
                acc[c] = _mm512_fmadd_ps(weight, load_16(row + 16 * c), acc[c]);
            }
        }
        for (size_t c = 0; c < width / 16; ++c)
        {
            _mm512_storeu_ps(out + d0 + 16 * c, _mm512_mul_ps(acc[c], _mm512_set1_ps(inverse)));
        }
    }
}

}

#else

namespace
{

template <typename Value>
void attention_head_impl(const float* q, const Value* k_tiles, const Value* v, size_t v_stride, size_t head_dim,
                         size_t n_past, float scale, float* scores, float* out, size_t skip)
{
    if (head_dim % 8 != 0)
    {
        attention_head_scalar_impl(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out, skip);
        return;
    }

    // 8 positions at a time: a tile holds 16, each d row of the tile is two ymm halves
    const size_t padded = (n_past + 7) / 8 * 8;
    const __m256 vscale = _mm256_set1_ps(scale);
    for (size_t p = 0; p < padded; p += 8)
    {
        const Value* tile = k_tiles + (p / KEY_TILE) * head_dim * KEY_TILE + (p % KEY_TILE);
        __m256 s = _mm256_setzero_ps();
        for (size_t d = 0; d < head_dim; ++d)
        {
            s = _mm256_fmadd_ps(_mm256_set1_ps(q[d]), load_8(tile + d * KEY_TILE), s);
        }
        _mm256_storeu_ps(scores + p, _mm256_mul_ps(s, vscale));
    }
    for (size_t p = 0; p < skip; ++p)
    {
        scores[p] = -INFINITY;
    }
    float max = scores[0];
    for (size_t p = 1; p < n_past; ++p)
    {
        max = scores[p] > max ? scores[p] : max;
    }
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256 vsum[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    for (size_t p = 0; p < padded; p += 8)
    {
        const __m256 e = exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(scores + p), _mm256_set1_ps(max)));
        _mm256_storeu_ps(scores + p, e);
        const __m256 keep = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(n_past - p)), lanes));
        vsum[(p / 8) & 1] = _mm256_add_ps(vsum[(p / 8) & 1], _mm256_and_ps(e, keep));
    }
    const float inverse = softmax_inverse(vsum[0], vsum[1]);

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
            const Value* row = v + p * v_stride + d0;
            for (size_t c = 0; c < width / 8; ++c)
            {
                acc[c] = _mm256_fmadd_ps(weight, load_8(row + 8 * c), acc[c]);
            }
        }
        for (size_t c = 0; c < width / 8; ++c)
        {
            _mm256_storeu_ps(out + d0 + 8 * c, _mm256_mul_ps(acc[c], _mm256_set1_ps(inverse)));
        }
    }
}

}

#endif

void attention_head(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                    size_t n_past, float scale, float* scores, float* out, size_t skip)
{
    attention_head_impl(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out, skip);
}

void attention_head_f16(const float* q, const uint16_t* k_tiles, const uint16_t* v, size_t v_stride, size_t head_dim,
                        size_t n_past, float scale, float* scores, float* out, size_t skip)
{
    attention_head_impl(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out, skip);
}

#else

void f32_to_f16_values(const float* x, size_t n, uint16_t* out)
{
    f32_to_f16_values_scalar(x, n, out);
}

void exp_values(float* x, size_t n)
{
    exp_values_scalar(x, n);
}

void swiglu(const float* gate, const float* up, size_t n, float* out)
{
    swiglu_scalar(gate, up, n, out);
}

void quantize_q8_0(const float* x, size_t n, uint8_t* out)
{
    quantize_q8_0_scalar(x, n, out);
}

void quantize_q8_k(const float* x, size_t n, uint8_t* out)
{
    quantize_q8_k_scalar(x, n, out);
}

void attention_head(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                    size_t n_past, float scale, float* scores, float* out)
{
    attention_head_scalar(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out);
}

void attention_head_f16(const float* q, const uint16_t* k_tiles, const uint16_t* v, size_t v_stride, size_t head_dim,
                        size_t n_past, float scale, float* scores, float* out)
{
    attention_head_f16_scalar(q, k_tiles, v, v_stride, head_dim, n_past, scale, scores, out);
}

#endif

void swiglu_scalar(const float* gate, const float* up, size_t n, float* out)
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
