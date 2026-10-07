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
