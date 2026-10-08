#include "faisca/vulkan_ops.h"

#include "faisca/ops.h"
#include "shaders_spv.h"

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

VulkanKernels::~VulkanKernels()
{
    release();
}

bool VulkanKernels::init(VulkanDevice& device, Error& error)
{
    release();
    m_device = &device;
    if (!device.create_pipeline(spv_matvec_q8_0, spv_matvec_q8_0_words, 4, 6 * sizeof(uint32_t), m_matvec_q8_0, error) ||
        !device.create_pipeline(spv_matvec_f32, spv_matvec_f32_words, 3, 5 * sizeof(uint32_t), m_matvec_f32, error))
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
        m_device->destroy_pipeline(m_matvec_f32);
    }
    m_device = nullptr;
}

bool VulkanKernels::matvec_q8_0(const VulkanBuffer& scales, const VulkanBuffer& quants, const VulkanBuffer& x,
                                VulkanBuffer& y, uint32_t n_in, uint32_t n_out, uint32_t n_tokens, uint32_t x_stride,
                                uint32_t y_stride, Error& error)
{
    const uint32_t push[6] = {n_in, n_out, n_in / static_cast<uint32_t>(Q8_0_BLOCK), x_stride, y_stride, n_tokens};
    const VulkanBuffer* buffers[4] = {&scales, &quants, &x, &y};
    return m_device->dispatch(m_matvec_q8_0, buffers, 4, push, sizeof(push), n_out, n_tokens, 1, error);
}

bool VulkanKernels::matvec_f32(const VulkanBuffer& w, const VulkanBuffer& x, VulkanBuffer& y, uint32_t n_in,
                               uint32_t n_out, uint32_t n_tokens, uint32_t x_stride, uint32_t y_stride, Error& error)
{
    const uint32_t push[5] = {n_in, n_out, x_stride, y_stride, n_tokens};
    const VulkanBuffer* buffers[3] = {&w, &x, &y};
    return m_device->dispatch(m_matvec_f32, buffers, 3, push, sizeof(push), n_out, n_tokens, 1, error);
}

}
