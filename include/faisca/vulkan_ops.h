#pragma once

#include "faisca/error.h"
#include "faisca/vulkan.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// Q8_0 rows in the layout the shaders read: scales as f16 pairs packed in uints,
// quants as 8 uints per block of 32, both indexed by row * blocks + block
size_t q8_0_split_scale_words(size_t n_in, size_t n_out);
size_t q8_0_split_quant_words(size_t n_in, size_t n_out);
void split_q8_0(const uint8_t* rows, size_t n_in, size_t n_out, uint32_t* scales, uint32_t* quants);

// the compute kernels, one pipeline each, recorded between VulkanDevice::begin and submit
class VulkanKernels
{
public:
    VulkanKernels() = default;
    ~VulkanKernels();

    VulkanKernels(const VulkanKernels&) = delete;
    VulkanKernels& operator=(const VulkanKernels&) = delete;

    bool init(VulkanDevice& device, Error& error);
    void release();

    // y[t * y_stride + o] = row o of the Q8_0 matrix dot x[t * x_stride ...], n_tokens inputs
    bool matvec_q8_0(const VulkanBuffer& scales, const VulkanBuffer& quants, const VulkanBuffer& x, VulkanBuffer& y,
                     uint32_t n_in, uint32_t n_out, uint32_t n_tokens, uint32_t x_stride, uint32_t y_stride,
                     Error& error);

    // the same over a float32 matrix of n_out rows of n_in values
    bool matvec_f32(const VulkanBuffer& w, const VulkanBuffer& x, VulkanBuffer& y, uint32_t n_in, uint32_t n_out,
                    uint32_t n_tokens, uint32_t x_stride, uint32_t y_stride, Error& error);

    // x[t] = row tokens[t] of the embedding matrix, n_tokens rows of n_embd
    bool embed_q8_0(const VulkanBuffer& scales, const VulkanBuffer& quants, const VulkanBuffer& tokens, VulkanBuffer& x,
                    uint32_t n_embd, uint32_t n_tokens, Error& error);
    bool embed_f32(const VulkanBuffer& w, const VulkanBuffer& tokens, VulkanBuffer& x, uint32_t n_embd, uint32_t n_tokens,
                   Error& error);

    // y[t] = rms_norm(x[t]) * weight over n values per token
    bool rms_norm(const VulkanBuffer& x, const VulkanBuffer& weight, VulkanBuffer& y, uint32_t n, uint32_t n_tokens,
                  float eps, Error& error);

    // rotates the first n_rot values of every head of x[t] in place, token t at position pos0 + t
    bool rope(VulkanBuffer& x, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, float freq_base,
              uint32_t n_tokens, Error& error);

    // copies k and v of the batch into the caches at positions pos0 .. of one layer
    bool kv_store(const VulkanBuffer& k, const VulkanBuffer& v, VulkanBuffer& k_cache, VulkanBuffer& v_cache,
                  uint32_t n_kv_head, uint32_t head_dim, uint32_t n_ctx, uint32_t pos0, uint32_t layer,
                  uint32_t n_tokens, Error& error);

    // y[t][head] = attention of q[t][head] over the cache of one layer, scores is scratch of
    // n_tokens * n_head * n_ctx values
    bool attention(const VulkanBuffer& q, const VulkanBuffer& k_cache, const VulkanBuffer& v_cache, VulkanBuffer& scores,
                   VulkanBuffer& y, uint32_t n_head, uint32_t n_kv_head, uint32_t head_dim, uint32_t n_ctx, uint32_t pos0,
                   uint32_t layer, uint32_t n_tokens, Error& error);

    bool swiglu(const VulkanBuffer& gate, const VulkanBuffer& up, VulkanBuffer& y, uint32_t n, Error& error);
    bool add(VulkanBuffer& x, const VulkanBuffer& y, uint32_t n, Error& error);

private:
    VulkanDevice* m_device = nullptr;
    VulkanPipeline m_matvec_q8_0;
    VulkanPipeline m_matvec_f32;
    VulkanPipeline m_embed_q8_0;
    VulkanPipeline m_embed_f32;
    VulkanPipeline m_rms_norm;
    VulkanPipeline m_rope;
    VulkanPipeline m_kv_store;
    VulkanPipeline m_attention;
    VulkanPipeline m_swiglu;
    VulkanPipeline m_add;
};

}
