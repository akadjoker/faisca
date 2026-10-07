#pragma once

#include <cstddef>
#include <cstdint>

namespace faisca
{

float f16_to_f32(uint16_t bits);

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
