#pragma once

#include "faisca/array.h"
#include "faisca/error.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// Compute over Vulkan. The loader library is opened at run time, so a build with
// this backend still runs on a machine without Vulkan: init just fails with a
// message. Only compute queues, storage buffers and push constants are used.

struct VulkanBuffer
{
    void* buffer = nullptr; // VkBuffer
    void* memory = nullptr; // VkDeviceMemory
    void* mapped = nullptr; // host address when the memory is host visible
    size_t bytes = 0;
};

struct VulkanPipeline
{
    void* layout = nullptr; // VkPipelineLayout
    void* set_layout = nullptr; // VkDescriptorSetLayout
    void* pipeline = nullptr; // VkPipeline
    uint32_t n_buffers = 0;
    uint32_t push_bytes = 0;
};

class VulkanDevice
{
public:
    static constexpr uint32_t MAX_BUFFERS = 8;
    static constexpr uint32_t MAX_PUSH_BYTES = 128;

    VulkanDevice() = default;
    ~VulkanDevice();

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    // picks the first discrete GPU, else the first device; device_index forces one
    bool init(Error& error, int device_index = -1);
    void release();

    bool ready() const
    {
        return m_device != nullptr;
    }

    const char* device_name() const
    {
        return m_device_name;
    }

    // true when every memory type is host visible (integrated GPUs, software devices)
    bool unified_memory() const
    {
        return m_unified;
    }

    // device memory, written through a staging copy when it is not host visible
    bool create_buffer(size_t bytes, VulkanBuffer& out, Error& error);
    bool create_host_buffer(size_t bytes, VulkanBuffer& out, Error& error);
    void destroy_buffer(VulkanBuffer& buffer);
    bool upload(VulkanBuffer& dst, const void* data, size_t bytes, Error& error);
    bool download(const VulkanBuffer& src, void* data, size_t bytes, Error& error);

    // a compute shader with n_buffers storage buffers at bindings 0.. and push_bytes of push constants
    bool create_pipeline(const uint32_t* spirv, size_t words, uint32_t n_buffers, uint32_t push_bytes,
                         VulkanPipeline& out, Error& error);
    void destroy_pipeline(VulkanPipeline& pipeline);

    // records dispatches between begin and submit; every dispatch sees the writes of the ones before it
    bool begin(Error& error);
    bool dispatch(const VulkanPipeline& pipeline, const VulkanBuffer* const* buffers, uint32_t n_buffers,
                  const void* push, uint32_t push_bytes, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                  Error& error);
    bool submit(Error& error);

private:
    struct Functions;

    bool load_library(Error& error);
    bool create_instance(Error& error);
    bool pick_device(int device_index, Error& error);
    bool create_device(Error& error);
    bool allocate(size_t bytes, bool host_visible, VulkanBuffer& out, Error& error);
    bool copy(const VulkanBuffer& src, VulkanBuffer& dst, size_t bytes, Error& error);
    uint32_t memory_type(uint32_t allowed, uint32_t required, uint32_t preferred) const;

    Functions* m_functions = nullptr;
    void* m_library = nullptr;
    void* m_instance = nullptr;
    void* m_physical = nullptr;
    void* m_device = nullptr;
    void* m_queue = nullptr;
    void* m_command_pool = nullptr;
    void* m_command_buffer = nullptr;
    void* m_descriptor_pool = nullptr;
    void* m_fence = nullptr;
    uint32_t m_queue_family = 0;
    uint32_t m_memory_type_count = 0;
    uint32_t m_memory_type_flags[32] = {};
    bool m_unified = false;
    bool m_recording = false;
    char m_device_name[256] = {};
};

}
