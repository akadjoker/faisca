#pragma once

#include <cstddef>
#include <cstdint>

namespace faisca
{

extern const uint32_t spv_add[];
constexpr size_t spv_add_words = 264;
extern const uint32_t spv_attention[];
constexpr size_t spv_attention_words = 1184;
extern const uint32_t spv_embed_f32[];
constexpr size_t spv_embed_f32_words = 361;
extern const uint32_t spv_embed_q8_0[];
constexpr size_t spv_embed_q8_0_words = 527;
extern const uint32_t spv_kv_store[];
constexpr size_t spv_kv_store_words = 529;
extern const uint32_t spv_matvec_f32[];
constexpr size_t spv_matvec_f32_words = 608;
extern const uint32_t spv_matvec_q8_0[];
constexpr size_t spv_matvec_q8_0_words = 911;
extern const uint32_t spv_rms_norm[];
constexpr size_t spv_rms_norm_words = 610;
extern const uint32_t spv_rope[];
constexpr size_t spv_rope_words = 495;
extern const uint32_t spv_swiglu[];
constexpr size_t spv_swiglu_words = 336;

}
