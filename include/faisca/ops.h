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

// 128 * sum of the quantized values of every block, the correction the packed kernel needs
void q8_0_block_sums(const uint8_t* xq, size_t n, int32_t* out);

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

// out = silu(gate) * up
void swiglu(const float* gate, const float* up, size_t n, float* out);

void add(float* x, const float* y, size_t n);

}
