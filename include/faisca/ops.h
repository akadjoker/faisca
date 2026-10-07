#pragma once

#include <cstddef>
#include <cstdint>

namespace faisca
{

float f16_to_f32(uint16_t bits);

// round to nearest even, as ggml and F16C do
uint16_t f32_to_f16(float value);

constexpr size_t Q8_0_BLOCK = 32;
constexpr size_t Q8_0_BLOCK_BYTES = 34; // f16 scale, 32 int8

// Q8_0 blocks of x, n a multiple of 32, as llama.cpp quantizes activations on x86:
// scale amax / 127 stored as f16, values x * 127 / amax rounded to nearest even
void quantize_q8_0(const float* x, size_t n, uint8_t* out);

void dequantize_q8_0(const uint8_t* blocks, size_t n, float* out);

// dot product of two Q8_0 rows of n values; the scalar one is the reference, the other may use SIMD
float dot_q8_0_scalar(const uint8_t* a, const uint8_t* b, size_t n);
float dot_q8_0(const uint8_t* a, const uint8_t* b, size_t n);

// out[o] = row o of the Q8_0 matrix w (n_out rows of n_in values) dot the Q8_0 vector xq
void matvec_q8_0(const uint8_t* w, const uint8_t* xq, size_t n_in, size_t n_out, float* out);

// Packed Q8_0: groups of 16 rows, block by block. A group block is 16 f16 scales
// then 8 chunks of 64 bytes; chunk j holds values 4j to 4j + 3 of the 16 rows, as
// unsigned bytes (value + 128), so that one VNNI instruction advances 16 rows.
// Rows are padded to a multiple of 16 with zeros.
constexpr size_t Q8_0_GROUP = 16;
constexpr size_t Q8_0_GROUP_BLOCK_BYTES = Q8_0_GROUP * Q8_0_BLOCK_BYTES;

// true when this build has a VNNI kernel for the packed layout
bool q8_0_packed_supported();

size_t q8_0_padded_rows(size_t n_out);
size_t q8_0_packed_bytes(size_t n_in, size_t n_out);
void pack_q8_0(const uint8_t* rows, size_t n_in, size_t n_out, uint8_t* out);
void row_q8_0_packed(const uint8_t* packed, size_t n_in, size_t row, float* out);

// sum of the quantized values of every block; the packed kernels multiply it by the
// offset their weights carry (128 for Q8_0, 8 for Q4_0)
void q8_0_block_sums(const uint8_t* xq, size_t n, int32_t* out);

// Q4_0: blocks of 32 values, f16 scale then 16 bytes, value j in the low nibble of
// byte j and value j + 16 in the high nibble, value = (nibble - 8) * scale
constexpr size_t Q4_0_BLOCK = 32;
constexpr size_t Q4_0_BLOCK_BYTES = 18;
constexpr size_t Q4_0_GROUP_BLOCK_BYTES = Q8_0_GROUP * Q4_0_BLOCK_BYTES;

void dequantize_q4_0(const uint8_t* blocks, size_t n, float* out);

// Packed Q4_0: groups of 16 rows, block by block: 16 f16 scales then 4 chunks of 64
// bytes; chunk c, bytes 4r..4r+3, holds values 8c..8c+3 of row r in the low nibbles and
// values 8c+4..8c+7 in the high nibbles
size_t q4_0_packed_bytes(size_t n_in, size_t n_out);
void pack_q4_0(const uint8_t* rows, size_t n_in, size_t n_out, uint8_t* out);
void row_q4_0_packed(const uint8_t* packed, size_t n_in, size_t row, float* out);
void matvec_q4_0_packed_scalar(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in,
                               size_t n_out, size_t first, size_t last, float* out);
void matmul_q4_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride);

// out[o] for o in [first, last), both multiples of 16 (last may be n_out)
void matvec_q8_0_packed(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in, size_t n_out,
                        size_t first, size_t last, float* out);
void matvec_q8_0_packed_scalar(const uint8_t* packed, const uint8_t* xq, const int32_t* x_sums, size_t n_in,
                               size_t n_out, size_t first, size_t last, float* out);

// several input vectors at once, each weight read once per 16 inputs: input t is
// xq + t * xq_stride with sums x_sums + t * sums_stride, output t is out + t * out_stride.
// Every output is bit identical to matvec_q8_0_packed on that input alone.
void matmul_q8_0_packed(const uint8_t* packed, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                        size_t sums_stride, size_t n_tokens, size_t n_in, size_t n_out, size_t first, size_t last,
                        float* out, size_t out_stride);

const char* simd_name();

// out[o] = sum over i of w[o * n_in + i] * x[i]
void matvec(const float* w, const float* x, size_t n_in, size_t n_out, float* out);

float dot(const float* a, const float* b, size_t n);

// out = x / sqrt(mean(x^2) + eps) * weight
void rms_norm(const float* x, const float* weight, size_t n, float eps, float* out);

// rotates the first n_rot values of every head in place, pairs (2i, 2i + 1), as ggml's NORM mode
void rope_norm(float* x, size_t n_head, size_t head_dim, size_t n_rot, uint32_t pos, float freq_base);

void softmax(float* x, size_t n);

// e^x for every value, within 2 ulp of the libm result; the scalar one is the reference
void exp_values(float* x, size_t n);
void exp_values_scalar(float* x, size_t n);

// keys of one head in tiles of 16 positions: value d of position p lives at
// k_tiles[(p / 16) * head_dim * 16 + d * 16 + p % 16], so the scores of 16 positions
// read head_dim * 64 contiguous bytes
constexpr size_t KEY_TILE = 16;

inline size_t key_index(size_t head_dim, size_t p, size_t d)
{
    return (p / KEY_TILE) * head_dim * KEY_TILE + d * KEY_TILE + p % KEY_TILE;
}

// one query against the cache of one head: v holds position p at v + p * v_stride; out
// gets head_dim values and scores is scratch for n_past values rounded up to 16
void attention_head(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                    size_t n_past, float scale, float* scores, float* out);
void attention_head_scalar(const float* q, const float* k_tiles, const float* v, size_t v_stride, size_t head_dim,
                           size_t n_past, float scale, float* scores, float* out);

// out = silu(gate) * up
void swiglu(const float* gate, const float* up, size_t n, float* out);

void add(float* x, const float* y, size_t n);

}
