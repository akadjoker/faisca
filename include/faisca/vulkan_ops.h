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

private:
    VulkanDevice* m_device = nullptr;
    VulkanPipeline m_matvec_q8_0;
    VulkanPipeline m_matvec_f32;
};

}
