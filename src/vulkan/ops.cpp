#include "faisca/vulkan_ops.h"

#include "faisca/ops.h"
#include "shaders_spv.h"

#include <cmath>
#include <cstring>

namespace faisca
{

size_t q8_0_split_scale_words(size_t n_in, size_t n_out)
{
    return (n_out * (n_in / Q8_0_BLOCK) + 1) / 2;
}

size_t q8_0_split_quant_words(size_t n_in, size_t n_out)
{
    return n_out * (n_in / Q8_0_BLOCK) * 8;
}

void split_q8_0(const uint8_t* rows, size_t n_in, size_t n_out, uint32_t* scales, uint32_t* quants)
{
    const size_t blocks = n_in / Q8_0_BLOCK;
    std::memset(scales, 0, q8_0_split_scale_words(n_in, n_out) * sizeof(uint32_t));
    for (size_t i = 0; i < n_out * blocks; ++i)
    {
        const uint8_t* block = rows + i * Q8_0_BLOCK_BYTES;
        uint16_t bits = 0;
        std::memcpy(&bits, block, 2);
        scales[i / 2] |= static_cast<uint32_t>(bits) << (16 * (i % 2));
        std::memcpy(quants + i * 8, block + 2, Q8_0_BLOCK);
    }
}

size_t q4_0_split_quant_words(size_t n_in, size_t n_out)
{
    return n_out * (n_in / Q4_0_BLOCK) * 4;
}

void split_q4_0(const uint8_t* rows, size_t n_in, size_t n_out, uint32_t* scales, uint32_t* quants)
{
    const size_t blocks = n_in / Q4_0_BLOCK;
    std::memset(scales, 0, q8_0_split_scale_words(n_in, n_out) * sizeof(uint32_t));
    for (size_t i = 0; i < n_out * blocks; ++i)
    {
        const uint8_t* block = rows + i * Q4_0_BLOCK_BYTES;
        uint16_t bits = 0;
        std::memcpy(&bits, block, 2);
        scales[i / 2] |= static_cast<uint32_t>(bits) << (16 * (i % 2));
        std::memcpy(quants + i * 4, block + 2, 16);
    }
}

size_t q5_0_split_quant_words(size_t n_in, size_t n_out)
{
    return n_out * (n_in / Q5_0_BLOCK) * 5;
}

void split_q5_0(const uint8_t* rows, size_t n_in, size_t n_out, uint32_t* scales, uint32_t* quants)
{
    const size_t blocks = n_in / Q5_0_BLOCK;
    std::memset(scales, 0, q8_0_split_scale_words(n_in, n_out) * sizeof(uint32_t));
    for (size_t i = 0; i < n_out * blocks; ++i)
    {
        const uint8_t* block = rows + i * Q5_0_BLOCK_BYTES;
        uint16_t bits = 0;
        std::memcpy(&bits, block, 2);
        scales[i / 2] |= static_cast<uint32_t>(bits) << (16 * (i % 2));
        std::memcpy(quants + i * 5, block + 2, 20);
    }
}

size_t q4_k_device_words(size_t n_in, size_t n_out)
{
    return n_out * (n_in / Q4_K_BLOCK) * (Q4_K_BLOCK_BYTES / 4);
}

void pack_q4_k_device(const uint8_t* rows, size_t n_in, size_t n_out, uint32_t* out)
{
    std::memcpy(out, rows, n_out * (n_in / Q4_K_BLOCK) * Q4_K_BLOCK_BYTES);
}

size_t q6_k_device_words(size_t n_in, size_t n_out)
{
    return n_out * (n_in / Q6_K_BLOCK) * 53;
}

void pack_q6_k_device(const uint8_t* rows, size_t n_in, size_t n_out, uint32_t* out)
{
    const size_t blocks = n_out * (n_in / Q6_K_BLOCK);
    for (size_t i = 0; i < blocks; ++i)
    {
        out[i * 53 + 52] = 0;
        std::memcpy(out + i * 53, rows + i * Q6_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES);
    }
}

VulkanKernels::~VulkanKernels()
{
    release();
}

bool VulkanKernels::init(VulkanDevice& device, Error& error)
{
    release();
    m_device = &device;
    if (!device.create_pipeline(spv_matvec_q8_0, spv_matvec_q8_0_words, 4, 6 * 4, m_matvec_q8_0, error) ||
        !device.create_pipeline(spv_matvec_q4_0, spv_matvec_q4_0_words, 4, 6 * 4, m_matvec_q4_0, error) ||
        !device.create_pipeline(spv_matvec_q5_0, spv_matvec_q5_0_words, 4, 6 * 4, m_matvec_q5_0, error) ||
        !device.create_pipeline(spv_matvec_q4_k, spv_matvec_q4_k_words, 3, 6 * 4, m_matvec_q4_k, error) ||
        !device.create_pipeline(spv_matvec_q6_k, spv_matvec_q6_k_words, 3, 6 * 4, m_matvec_q6_k, error) ||
        !device.create_pipeline(spv_matvec_f32, spv_matvec_f32_words, 3, 5 * 4, m_matvec_f32, error) ||
        !device.create_pipeline(spv_embed_q8_0, spv_embed_q8_0_words, 4, 2 * 4, m_embed_q8_0, error) ||
        !device.create_pipeline(spv_embed_f32, spv_embed_f32_words, 3, 1 * 4, m_embed_f32, error) ||
        !device.create_pipeline(spv_rms_norm, spv_rms_norm_words, 3, 2 * 4, m_rms_norm, error) ||
        !device.create_pipeline(spv_rope, spv_rope_words, 2, 6 * 4, m_rope, error) ||
        !device.create_pipeline(spv_kv_store, spv_kv_store_words, 4, 5 * 4, m_kv_store, error) ||
        !device.create_pipeline(spv_attention, spv_attention_words, 5, 7 * 4, m_attention, error) ||
        !device.create_pipeline(spv_swiglu, spv_swiglu_words, 3, 1 * 4, m_swiglu, error) ||
        !device.create_pipeline(spv_add, spv_add_words, 2, 1 * 4, m_add, error))
    {
        release();
        return false;
    }
    return true;
}

void VulkanKernels::release()
{
    if (m_device != nullptr)
    {
        m_device->destroy_pipeline(m_matvec_q8_0);
        m_device->destroy_pipeline(m_matvec_q4_0);
        m_device->destroy_pipeline(m_matvec_q5_0);
        m_device->destroy_pipeline(m_matvec_q4_k);
        m_device->destroy_pipeline(m_matvec_q6_k);
        m_device->destroy_pipeline(m_matvec_f32);
        m_device->destroy_pipeline(m_embed_q8_0);
        m_device->destroy_pipeline(m_embed_f32);
        m_device->destroy_pipeline(m_rms_norm);
        m_device->destroy_pipeline(m_rope);
        m_device->destroy_pipeline(m_kv_store);
        m_device->destroy_pipeline(m_attention);
        m_device->destroy_pipeline(m_swiglu);
        m_device->destroy_pipeline(m_add);
    }
    m_device = nullptr;
}

namespace
{

inline uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    return bits;
}

inline uint32_t groups_of(uint32_t n, uint32_t size)
{
    return (n + size - 1) / size;
}

}

bool VulkanKernels::embed_q8_0(const VulkanBuffer& scales, const VulkanBuffer& quants, const VulkanBuffer& tokens,
                               VulkanBuffer& x, uint32_t n_embd, uint32_t n_tokens, Error& error)
{
    const uint32_t push[2] = {n_embd, n_embd / static_cast<uint32_t>(Q8_0_BLOCK)};
    const VulkanBuffer* buffers[4] = {&scales, &quants, &tokens, &x};
    return m_device->dispatch(m_embed_q8_0, buffers, 4, push, sizeof(push), groups_of(n_embd, 64), n_tokens, 1, error);
}

bool VulkanKernels::embed_f32(const VulkanBuffer& w, const VulkanBuffer& tokens, VulkanBuffer& x, uint32_t n_embd,
                              uint32_t n_tokens, Error& error)
{
    const uint32_t push[1] = {n_embd};
    const VulkanBuffer* buffers[3] = {&w, &tokens, &x};
    return m_device->dispatch(m_embed_f32, buffers, 3, push, sizeof(push), groups_of(n_embd, 64), n_tokens, 1, error);
}

bool VulkanKernels::rms_norm(const VulkanBuffer& x, const VulkanBuffer& weight, VulkanBuffer& y, uint32_t n,
                             uint32_t n_tokens, float eps, Error& error)
{
    const uint32_t push[2] = {n, float_bits(eps)};
    const VulkanBuffer* buffers[3] = {&x, &weight, &y};
    return m_device->dispatch(m_rms_norm, buffers, 3, push, sizeof(push), n_tokens, 1, 1, error);
}

bool VulkanKernels::rope(VulkanBuffer& x, const VulkanBuffer& factors, bool has_factors, uint32_t n_head,
                         uint32_t head_dim, uint32_t n_rot, uint32_t pos0, float freq_base, uint32_t n_tokens,
                         Error& error)
{
    const uint32_t push[6] = {n_head, head_dim, n_rot, pos0, float_bits(freq_base), has_factors ? 1u : 0u};
    const VulkanBuffer* buffers[2] = {&x, &factors};
    return m_device->dispatch(m_rope, buffers, 2, push, sizeof(push), n_tokens, n_head, 1, error);
}

bool VulkanKernels::kv_store(const VulkanBuffer& k, const VulkanBuffer& v, VulkanBuffer& k_cache, VulkanBuffer& v_cache,
                             uint32_t n_kv_head, uint32_t head_dim, uint32_t n_ctx, uint32_t pos0, uint32_t layer,
                             uint32_t n_tokens, Error& error)
{
    const uint32_t push[5] = {n_kv_head, head_dim, n_ctx, pos0, layer};
    const VulkanBuffer* buffers[4] = {&k, &v, &k_cache, &v_cache};
    return m_device->dispatch(m_kv_store, buffers, 4, push, sizeof(push), n_tokens, n_kv_head, 1, error);
}

bool VulkanKernels::attention(const VulkanBuffer& q, const VulkanBuffer& k_cache, const VulkanBuffer& v_cache,
                              VulkanBuffer& scores, VulkanBuffer& y, uint32_t n_head, uint32_t n_kv_head,
                              uint32_t head_dim, uint32_t n_ctx, uint32_t pos0, uint32_t layer, uint32_t n_tokens,
                              Error& error)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const uint32_t push[7] = {n_head, n_kv_head, head_dim, n_ctx, pos0, layer, float_bits(scale)};
    const VulkanBuffer* buffers[5] = {&q, &k_cache, &v_cache, &scores, &y};
    return m_device->dispatch(m_attention, buffers, 5, push, sizeof(push), n_tokens, n_head, 1, error);
}

bool VulkanKernels::swiglu(const VulkanBuffer& gate, const VulkanBuffer& up, VulkanBuffer& y, uint32_t n, Error& error)
{
    const uint32_t push[1] = {n};
    const VulkanBuffer* buffers[3] = {&gate, &up, &y};
    return m_device->dispatch(m_swiglu, buffers, 3, push, sizeof(push), groups_of(n, 256), 1, 1, error);
}

bool VulkanKernels::add(VulkanBuffer& x, const VulkanBuffer& y, uint32_t n, Error& error)
{
    const uint32_t push[1] = {n};
    const VulkanBuffer* buffers[2] = {&x, &y};
    return m_device->dispatch(m_add, buffers, 2, push, sizeof(push), groups_of(n, 256), 1, 1, error);
}

bool VulkanKernels::matvec_q8_0(const VulkanBuffer& scales, const VulkanBuffer& quants, const VulkanBuffer& x,
                                VulkanBuffer& y, uint32_t n_in, uint32_t n_out, uint32_t n_tokens, uint32_t x_stride,
                                uint32_t y_stride, Error& error)
{
    const uint32_t push[6] = {n_in, n_out, n_in / static_cast<uint32_t>(Q8_0_BLOCK), x_stride, y_stride, n_tokens};
    const VulkanBuffer* buffers[4] = {&scales, &quants, &x, &y};
    return m_device->dispatch(m_matvec_q8_0, buffers, 4, push, sizeof(push), n_out, n_tokens, 1, error);
}

bool VulkanKernels::matvec_q4_0(const VulkanBuffer& scales, const VulkanBuffer& quants, const VulkanBuffer& x,
                                VulkanBuffer& y, uint32_t n_in, uint32_t n_out, uint32_t n_tokens, uint32_t x_stride,
                                uint32_t y_stride, Error& error)
{
    const uint32_t push[6] = {n_in, n_out, n_in / static_cast<uint32_t>(Q4_0_BLOCK), x_stride, y_stride, n_tokens};
    const VulkanBuffer* buffers[4] = {&scales, &quants, &x, &y};
    return m_device->dispatch(m_matvec_q4_0, buffers, 4, push, sizeof(push), n_out, n_tokens, 1, error);
}

bool VulkanKernels::matvec_q5_0(const VulkanBuffer& scales, const VulkanBuffer& quants, const VulkanBuffer& x,
                                VulkanBuffer& y, uint32_t n_in, uint32_t n_out, uint32_t n_tokens, uint32_t x_stride,
                                uint32_t y_stride, Error& error)
{
    const uint32_t push[6] = {n_in, n_out, n_in / static_cast<uint32_t>(Q5_0_BLOCK), x_stride, y_stride, n_tokens};
    const VulkanBuffer* buffers[4] = {&scales, &quants, &x, &y};
    return m_device->dispatch(m_matvec_q5_0, buffers, 4, push, sizeof(push), n_out, n_tokens, 1, error);
}

bool VulkanKernels::matvec_q4_k(const VulkanBuffer& blocks, const VulkanBuffer& x, VulkanBuffer& y, uint32_t n_in,
                                uint32_t n_out, uint32_t n_tokens, uint32_t x_stride, uint32_t y_stride, Error& error)
{
    const uint32_t push[6] = {n_in, n_out, n_in / static_cast<uint32_t>(Q4_K_BLOCK), x_stride, y_stride, n_tokens};
    const VulkanBuffer* buffers[3] = {&blocks, &x, &y};
    return m_device->dispatch(m_matvec_q4_k, buffers, 3, push, sizeof(push), n_out, n_tokens, 1, error);
}

bool VulkanKernels::matvec_q6_k(const VulkanBuffer& blocks, const VulkanBuffer& x, VulkanBuffer& y, uint32_t n_in,
                                uint32_t n_out, uint32_t n_tokens, uint32_t x_stride, uint32_t y_stride, Error& error)
{
    const uint32_t push[6] = {n_in, n_out, n_in / static_cast<uint32_t>(Q6_K_BLOCK), x_stride, y_stride, n_tokens};
    const VulkanBuffer* buffers[3] = {&blocks, &x, &y};
    return m_device->dispatch(m_matvec_q6_k, buffers, 3, push, sizeof(push), n_out, n_tokens, 1, error);
}

bool VulkanKernels::matvec_f32(const VulkanBuffer& w, const VulkanBuffer& x, VulkanBuffer& y, uint32_t n_in,
                               uint32_t n_out, uint32_t n_tokens, uint32_t x_stride, uint32_t y_stride, Error& error)
{
    const uint32_t push[5] = {n_in, n_out, x_stride, y_stride, n_tokens};
    const VulkanBuffer* buffers[3] = {&w, &x, &y};
    return m_device->dispatch(m_matvec_f32, buffers, 3, push, sizeof(push), n_out, n_tokens, 1, error);
}

}
