#include "faisca/vulkan.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace faisca
{

#define FAISCA_VK_INSTANCE_FUNCTIONS(X) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceFeatures2) \
    X(vkCreateDevice) \
    X(vkGetDeviceProcAddr) \
    X(vkDestroyInstance)

#define FAISCA_VK_DEVICE_FUNCTIONS(X) \
    X(vkGetDeviceQueue) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkBindBufferMemory) \
    X(vkMapMemory) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) \
    X(vkCreateDescriptorPool) \
    X(vkDestroyDescriptorPool) \
    X(vkResetDescriptorPool) \
    X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkResetCommandBuffer) \
    X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) \
    X(vkCmdDispatch) \
    X(vkCmdPipelineBarrier) \
    X(vkCmdCopyBuffer) \
    X(vkQueueSubmit) \
    X(vkCreateFence) \
    X(vkDestroyFence) \
    X(vkWaitForFences) \
    X(vkResetFences) \
    X(vkDeviceWaitIdle) \
    X(vkDestroyDevice)

struct VulkanDevice::Functions
{
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;
#define FAISCA_VK_DECLARE(name) PFN_##name name = nullptr;
    FAISCA_VK_INSTANCE_FUNCTIONS(FAISCA_VK_DECLARE)
    FAISCA_VK_DEVICE_FUNCTIONS(FAISCA_VK_DECLARE)
#undef FAISCA_VK_DECLARE
};

namespace
{

const char* result_name(VkResult result)
{
    switch (result)
    {
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "out of host memory";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "out of device memory";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "initialization failed";
        case VK_ERROR_DEVICE_LOST:
            return "device lost";
        case VK_ERROR_LAYER_NOT_PRESENT:
            return "layer not present";
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "extension not present";
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return "feature not present";
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return "incompatible driver";
        case VK_TIMEOUT:
            return "timeout";
        default:
            return "error";
    }
}

template <typename T>
T as(void* handle)
{
    return reinterpret_cast<T>(handle);
}

}

VulkanDevice::~VulkanDevice()
{
    release();
}

bool VulkanDevice::load_library(Error& error)
{
    m_functions = new Functions();
#if defined(_WIN32)
    HMODULE library = LoadLibraryA("vulkan-1.dll");
    if (library == nullptr)
    {
        error.set("vulkan-1.dll not found");
        return false;
    }
    m_library = library;
    m_functions->vkGetInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
    void* library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (library == nullptr)
    {
        error.set("libvulkan.so.1 not found");
        return false;
    }
    m_library = library;
    m_functions->vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
    if (m_functions->vkGetInstanceProcAddr == nullptr)
    {
        error.set("the Vulkan loader has no vkGetInstanceProcAddr");
        return false;
    }
    m_functions->vkCreateInstance =
        reinterpret_cast<PFN_vkCreateInstance>(m_functions->vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    m_functions->vkEnumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        m_functions->vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (m_functions->vkCreateInstance == nullptr)
    {
        error.set("the Vulkan loader has no vkCreateInstance");
        return false;
    }
    return true;
}

bool VulkanDevice::create_instance(Error& error)
{
    VkApplicationInfo application = {};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "faisca";
    application.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &application;

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult result = m_functions->vkCreateInstance(&info, nullptr, &instance);
    if (result != VK_SUCCESS)
    {
        error.set("vkCreateInstance: %s", result_name(result));
        return false;
    }
    m_instance = instance;

#define FAISCA_VK_LOAD(name) \
    m_functions->name = reinterpret_cast<PFN_##name>(m_functions->vkGetInstanceProcAddr(instance, #name)); \
    if (m_functions->name == nullptr) \
    { \
        error.set("the Vulkan loader has no " #name); \
        return false; \
    }
    FAISCA_VK_INSTANCE_FUNCTIONS(FAISCA_VK_LOAD)
#undef FAISCA_VK_LOAD
    return true;
}

bool VulkanDevice::pick_device(int device_index, Error& error)
{
    VkInstance instance = as<VkInstance>(m_instance);
    uint32_t count = 0;
    m_functions->vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0)
    {
        error.set("no Vulkan device");
        return false;
    }
    Array<VkPhysicalDevice> devices;
    if (!devices.resize_uninitialized(count))
    {
        error.set("out of memory");
        return false;
    }
    m_functions->vkEnumeratePhysicalDevices(instance, &count, devices.data());

    int chosen = -1;
    if (device_index >= 0)
    {
        if (static_cast<uint32_t>(device_index) >= count)
        {
            error.set("Vulkan device %d does not exist, %u found", device_index, count);
            return false;
        }
        chosen = device_index;
    }
    else
    {
        for (uint32_t i = 0; i < count && chosen < 0; ++i)
        {
            VkPhysicalDeviceProperties properties;
            m_functions->vkGetPhysicalDeviceProperties(devices[i], &properties);
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                chosen = static_cast<int>(i);
            }
        }
        for (uint32_t i = 0; i < count && chosen < 0; ++i)
        {
            VkPhysicalDeviceProperties properties;
            m_functions->vkGetPhysicalDeviceProperties(devices[i], &properties);
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            {
                chosen = static_cast<int>(i);
            }
        }
        chosen = chosen < 0 ? 0 : chosen;
    }
    VkPhysicalDevice physical = devices[static_cast<size_t>(chosen)];
    m_physical = physical;

    VkPhysicalDeviceProperties properties;
    m_functions->vkGetPhysicalDeviceProperties(physical, &properties);
    std::memcpy(m_device_name, properties.deviceName, sizeof(m_device_name) - 1);
    m_device_name[sizeof(m_device_name) - 1] = '\0';

    uint32_t families = 0;
    m_functions->vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
    Array<VkQueueFamilyProperties> family;
    if (!family.resize_uninitialized(families))
    {
        error.set("out of memory");
        return false;
    }
    m_functions->vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, family.data());
    int queue_family = -1;
    for (uint32_t i = 0; i < families; ++i)
    {
        const bool compute = (family[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool graphics = (family[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (compute && (queue_family < 0 || !graphics))
        {
            queue_family = static_cast<int>(i);
        }
    }
    if (queue_family < 0)
    {
        error.set("%s has no compute queue", m_device_name);
        return false;
    }
    m_queue_family = static_cast<uint32_t>(queue_family);

    VkPhysicalDeviceMemoryProperties memory;
    m_functions->vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    m_memory_type_count = memory.memoryTypeCount;
    m_unified = true;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
    {
        m_memory_type_flags[i] = memory.memoryTypes[i].propertyFlags;
        const bool device_local = (m_memory_type_flags[i] & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        const bool host_visible = (m_memory_type_flags[i] & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        if (device_local && !host_visible)
        {
            m_unified = false;
        }
    }
    return true;
}

bool VulkanDevice::create_device(Error& error)
{
    VkPhysicalDevice physical = as<VkPhysicalDevice>(m_physical);
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue = {};
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = m_queue_family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;

    VkDeviceCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;

    VkDevice device = VK_NULL_HANDLE;
    VkResult result = m_functions->vkCreateDevice(physical, &info, nullptr, &device);
    if (result != VK_SUCCESS)
    {
        error.set("vkCreateDevice: %s", result_name(result));
        return false;
    }
    m_device = device;

#define FAISCA_VK_LOAD(name) \
    m_functions->name = reinterpret_cast<PFN_##name>(m_functions->vkGetDeviceProcAddr(device, #name)); \
    if (m_functions->name == nullptr) \
    { \
        error.set("the Vulkan device has no " #name); \
        return false; \
    }
    FAISCA_VK_DEVICE_FUNCTIONS(FAISCA_VK_LOAD)
#undef FAISCA_VK_LOAD

    VkQueue vk_queue = VK_NULL_HANDLE;
    m_functions->vkGetDeviceQueue(device, m_queue_family, 0, &vk_queue);
    m_queue = vk_queue;

    VkCommandPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = m_queue_family;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    result = m_functions->vkCreateCommandPool(device, &pool, nullptr, &command_pool);
    if (result != VK_SUCCESS)
    {
        error.set("vkCreateCommandPool: %s", result_name(result));
        return false;
    }
    m_command_pool = command_pool;

    VkCommandBufferAllocateInfo allocate = {};
    allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate.commandPool = command_pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    result = m_functions->vkAllocateCommandBuffers(device, &allocate, &command_buffer);
    if (result != VK_SUCCESS)
    {
        error.set("vkAllocateCommandBuffers: %s", result_name(result));
        return false;
    }
    m_command_buffer = command_buffer;

    VkDescriptorPoolSize size = {};
    size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 4096 * MAX_BUFFERS;
    VkDescriptorPoolCreateInfo descriptor_pool = {};
    descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool.maxSets = 4096;
    descriptor_pool.poolSizeCount = 1;
    descriptor_pool.pPoolSizes = &size;
    VkDescriptorPool pool_handle = VK_NULL_HANDLE;
    result = m_functions->vkCreateDescriptorPool(device, &descriptor_pool, nullptr, &pool_handle);
    if (result != VK_SUCCESS)
    {
        error.set("vkCreateDescriptorPool: %s", result_name(result));
        return false;
    }
    m_descriptor_pool = pool_handle;

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    result = m_functions->vkCreateFence(device, &fence_info, nullptr, &fence);
    if (result != VK_SUCCESS)
    {
        error.set("vkCreateFence: %s", result_name(result));
        return false;
    }
    m_fence = fence;
    return true;
}

bool VulkanDevice::init(Error& error, int device_index)
{
    release();
    if (!load_library(error) || !create_instance(error) || !pick_device(device_index, error) || !create_device(error))
    {
        release();
        return false;
    }
    return true;
}

void VulkanDevice::release()
{
    if (m_functions != nullptr)
    {
        VkDevice device = as<VkDevice>(m_device);
        if (device != VK_NULL_HANDLE)
        {
            m_functions->vkDeviceWaitIdle(device);
            if (m_fence != nullptr)
            {
                m_functions->vkDestroyFence(device, as<VkFence>(m_fence), nullptr);
            }
            if (m_descriptor_pool != nullptr)
            {
                m_functions->vkDestroyDescriptorPool(device, as<VkDescriptorPool>(m_descriptor_pool), nullptr);
            }
            if (m_command_pool != nullptr)
            {
                m_functions->vkDestroyCommandPool(device, as<VkCommandPool>(m_command_pool), nullptr);
            }
            m_functions->vkDestroyDevice(device, nullptr);
        }
        if (m_instance != nullptr && m_functions->vkDestroyInstance != nullptr)
        {
            m_functions->vkDestroyInstance(as<VkInstance>(m_instance), nullptr);
        }
        delete m_functions;
        m_functions = nullptr;
    }
    if (m_library != nullptr)
    {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(m_library));
#else
        dlclose(m_library);
#endif
    }
    m_library = nullptr;
    m_instance = nullptr;
    m_physical = nullptr;
    m_device = nullptr;
    m_queue = nullptr;
    m_command_pool = nullptr;
    m_command_buffer = nullptr;
    m_descriptor_pool = nullptr;
    m_fence = nullptr;
    m_recording = false;
    m_device_name[0] = '\0';
}

uint32_t VulkanDevice::memory_type(uint32_t allowed, uint32_t required, uint32_t preferred) const
{
    for (uint32_t i = 0; i < m_memory_type_count; ++i)
    {
        const bool ok = (allowed & (1u << i)) != 0 && (m_memory_type_flags[i] & required) == required;
        if (ok && (m_memory_type_flags[i] & preferred) == preferred)
        {
            return i;
        }
    }
    for (uint32_t i = 0; i < m_memory_type_count; ++i)
    {
        if ((allowed & (1u << i)) != 0 && (m_memory_type_flags[i] & required) == required)
        {
            return i;
        }
    }
    return m_memory_type_count;
}

bool VulkanDevice::allocate(size_t bytes, bool host_visible, VulkanBuffer& out, Error& error)
{
    VkDevice device = as<VkDevice>(m_device);
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult result = m_functions->vkCreateBuffer(device, &info, nullptr, &buffer);
    if (result != VK_SUCCESS)
    {
        error.set("vkCreateBuffer of %zu bytes: %s", bytes, result_name(result));
        return false;
    }
    VkMemoryRequirements requirements;
    m_functions->vkGetBufferMemoryRequirements(device, buffer, &requirements);

    const uint32_t host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const uint32_t type = host_visible ? memory_type(requirements.memoryTypeBits, host_flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                       : memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, host_flags);
    if (type == m_memory_type_count)
    {
        m_functions->vkDestroyBuffer(device, buffer, nullptr);
        error.set("no suitable memory type on %s", m_device_name);
        return false;
    }
    VkMemoryAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    result = m_functions->vkAllocateMemory(device, &allocate_info, nullptr, &memory);
    if (result != VK_SUCCESS)
    {
        m_functions->vkDestroyBuffer(device, buffer, nullptr);
        error.set("vkAllocateMemory of %zu bytes: %s", bytes, result_name(result));
        return false;
    }
    result = m_functions->vkBindBufferMemory(device, buffer, memory, 0);
    if (result != VK_SUCCESS)
    {
        m_functions->vkFreeMemory(device, memory, nullptr);
        m_functions->vkDestroyBuffer(device, buffer, nullptr);
        error.set("vkBindBufferMemory: %s", result_name(result));
        return false;
    }
    out.buffer = buffer;
    out.memory = memory;
    out.bytes = bytes;
    out.mapped = nullptr;
    if ((m_memory_type_flags[type] & host_flags) == host_flags)
    {
        void* mapped = nullptr;
        result = m_functions->vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped);
        if (result != VK_SUCCESS)
        {
            destroy_buffer(out);
            error.set("vkMapMemory: %s", result_name(result));
            return false;
        }
        out.mapped = mapped;
    }
    return true;
}

bool VulkanDevice::create_buffer(size_t bytes, VulkanBuffer& out, Error& error)
{
    return allocate(bytes, false, out, error);
}

bool VulkanDevice::create_host_buffer(size_t bytes, VulkanBuffer& out, Error& error)
{
    return allocate(bytes, true, out, error);
}

void VulkanDevice::destroy_buffer(VulkanBuffer& buffer)
{
    if (m_device == nullptr || m_functions == nullptr)
    {
        return;
    }
    VkDevice device = as<VkDevice>(m_device);
    if (buffer.buffer != nullptr)
    {
        m_functions->vkDestroyBuffer(device, as<VkBuffer>(buffer.buffer), nullptr);
    }
    if (buffer.memory != nullptr)
    {
        m_functions->vkFreeMemory(device, as<VkDeviceMemory>(buffer.memory), nullptr);
    }
    buffer = VulkanBuffer();
}

bool VulkanDevice::copy(const VulkanBuffer& src, VulkanBuffer& dst, size_t bytes, Error& error)
{
    if (m_recording)
    {
        error.set("cannot copy while recording");
        return false;
    }
    VkCommandBuffer command_buffer = as<VkCommandBuffer>(m_command_buffer);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    m_functions->vkResetCommandBuffer(command_buffer, 0);
    m_functions->vkBeginCommandBuffer(command_buffer, &begin_info);
    VkBufferCopy region = {};
    region.size = bytes;
    m_functions->vkCmdCopyBuffer(command_buffer, as<VkBuffer>(src.buffer), as<VkBuffer>(dst.buffer), 1, &region);
    m_functions->vkEndCommandBuffer(command_buffer);
    m_recording = true;
    return submit(error);
}

bool VulkanDevice::upload(VulkanBuffer& dst, const void* data, size_t bytes, Error& error)
{
    if (bytes > dst.bytes)
    {
        error.set("upload of %zu bytes into a buffer of %zu", bytes, dst.bytes);
        return false;
    }
    if (dst.mapped != nullptr)
    {
        std::memcpy(dst.mapped, data, bytes);
        return true;
    }
    VulkanBuffer staging;
    if (!create_host_buffer(bytes, staging, error))
    {
        return false;
    }
    std::memcpy(staging.mapped, data, bytes);
    const bool ok = copy(staging, dst, bytes, error);
    destroy_buffer(staging);
    return ok;
}

bool VulkanDevice::download(const VulkanBuffer& src, void* data, size_t bytes, Error& error)
{
    if (bytes > src.bytes)
    {
        error.set("download of %zu bytes from a buffer of %zu", bytes, src.bytes);
        return false;
    }
    if (src.mapped != nullptr)
    {
        std::memcpy(data, src.mapped, bytes);
        return true;
    }
    VulkanBuffer staging;
    if (!create_host_buffer(bytes, staging, error))
    {
        return false;
    }
    const bool ok = copy(src, staging, bytes, error);
    if (ok)
    {
        std::memcpy(data, staging.mapped, bytes);
    }
    destroy_buffer(staging);
    return ok;
}

bool VulkanDevice::create_pipeline(const uint32_t* spirv, size_t words, uint32_t n_buffers, uint32_t push_bytes,
                                   VulkanPipeline& out, Error& error)
{
    if (n_buffers > MAX_BUFFERS || push_bytes > MAX_PUSH_BYTES)
    {
        error.set("a pipeline takes at most %u buffers and %u bytes of push constants", MAX_BUFFERS, MAX_PUSH_BYTES);
        return false;
    }
    VkDevice device = as<VkDevice>(m_device);
    VkShaderModuleCreateInfo module_info = {};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = words * sizeof(uint32_t);
    module_info.pCode = spirv;
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult result = m_functions->vkCreateShaderModule(device, &module_info, nullptr, &module);
    if (result != VK_SUCCESS)
    {
        error.set("vkCreateShaderModule: %s", result_name(result));
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[MAX_BUFFERS] = {};
    for (uint32_t i = 0; i < n_buffers; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = n_buffers;
    layout_info.pBindings = bindings;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    result = m_functions->vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout);
    if (result != VK_SUCCESS)
    {
        m_functions->vkDestroyShaderModule(device, module, nullptr);
        error.set("vkCreateDescriptorSetLayout: %s", result_name(result));
        return false;
    }

    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = push_bytes;
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    pipeline_layout_info.pushConstantRangeCount = push_bytes > 0 ? 1 : 0;
    pipeline_layout_info.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    result = m_functions->vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &layout);
    if (result != VK_SUCCESS)
    {
        m_functions->vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        m_functions->vkDestroyShaderModule(device, module, nullptr);
        error.set("vkCreatePipelineLayout: %s", result_name(result));
        return false;
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    result = m_functions->vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
    m_functions->vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS)
    {
        m_functions->vkDestroyPipelineLayout(device, layout, nullptr);
        m_functions->vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        error.set("vkCreateComputePipelines: %s", result_name(result));
        return false;
    }
    out.layout = layout;
    out.set_layout = set_layout;
    out.pipeline = pipeline;
    out.n_buffers = n_buffers;
    out.push_bytes = push_bytes;
    return true;
}

void VulkanDevice::destroy_pipeline(VulkanPipeline& pipeline)
{
    if (m_device == nullptr || m_functions == nullptr)
    {
        return;
    }
    VkDevice device = as<VkDevice>(m_device);
    if (pipeline.pipeline != nullptr)
    {
        m_functions->vkDestroyPipeline(device, as<VkPipeline>(pipeline.pipeline), nullptr);
    }
    if (pipeline.layout != nullptr)
    {
        m_functions->vkDestroyPipelineLayout(device, as<VkPipelineLayout>(pipeline.layout), nullptr);
    }
    if (pipeline.set_layout != nullptr)
    {
        m_functions->vkDestroyDescriptorSetLayout(device, as<VkDescriptorSetLayout>(pipeline.set_layout), nullptr);
    }
    pipeline = VulkanPipeline();
}

bool VulkanDevice::begin(Error& error)
{
    if (m_recording)
    {
        error.set("already recording");
        return false;
    }
    VkDevice device = as<VkDevice>(m_device);
    VkCommandBuffer command_buffer = as<VkCommandBuffer>(m_command_buffer);
    m_functions->vkResetDescriptorPool(device, as<VkDescriptorPool>(m_descriptor_pool), 0);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    m_functions->vkResetCommandBuffer(command_buffer, 0);
    const VkResult result = m_functions->vkBeginCommandBuffer(command_buffer, &begin_info);
    if (result != VK_SUCCESS)
    {
        error.set("vkBeginCommandBuffer: %s", result_name(result));
        return false;
    }
    m_recording = true;
    return true;
}

bool VulkanDevice::dispatch(const VulkanPipeline& pipeline, const VulkanBuffer* const* buffers, uint32_t n_buffers,
                            const void* push, uint32_t push_bytes, uint32_t groups_x, uint32_t groups_y,
                            uint32_t groups_z, Error& error)
{
    if (!m_recording)
    {
        error.set("dispatch outside begin and submit");
        return false;
    }
    if (n_buffers != pipeline.n_buffers || push_bytes != pipeline.push_bytes)
    {
        error.set("dispatch with %u buffers and %u push bytes, the pipeline takes %u and %u", n_buffers, push_bytes,
                  pipeline.n_buffers, pipeline.push_bytes);
        return false;
    }
    VkDevice device = as<VkDevice>(m_device);
    VkCommandBuffer command_buffer = as<VkCommandBuffer>(m_command_buffer);

    VkDescriptorSetLayout set_layout = as<VkDescriptorSetLayout>(pipeline.set_layout);
    VkDescriptorSetAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = as<VkDescriptorPool>(m_descriptor_pool);
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    const VkResult result = m_functions->vkAllocateDescriptorSets(device, &allocate_info, &set);
    if (result != VK_SUCCESS)
    {
        error.set("vkAllocateDescriptorSets: %s (too many dispatches in one submission)", result_name(result));
        return false;
    }
    VkDescriptorBufferInfo infos[MAX_BUFFERS] = {};
    VkWriteDescriptorSet writes[MAX_BUFFERS] = {};
    for (uint32_t i = 0; i < n_buffers; ++i)
    {
        infos[i].buffer = as<VkBuffer>(buffers[i]->buffer);
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    m_functions->vkUpdateDescriptorSets(device, n_buffers, writes, 0, nullptr);

    m_functions->vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, as<VkPipeline>(pipeline.pipeline));
    m_functions->vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                         as<VkPipelineLayout>(pipeline.layout), 0, 1, &set, 0, nullptr);
    if (push_bytes > 0)
    {
        m_functions->vkCmdPushConstants(command_buffer, as<VkPipelineLayout>(pipeline.layout), VK_SHADER_STAGE_COMPUTE_BIT,
                                        0, push_bytes, push);
    }
    m_functions->vkCmdDispatch(command_buffer, groups_x, groups_y, groups_z);

    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    m_functions->vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    return true;
}

bool VulkanDevice::submit(Error& error)
{
    if (!m_recording)
    {
        error.set("submit without begin");
        return false;
    }
    m_recording = false;
    VkDevice device = as<VkDevice>(m_device);
    VkCommandBuffer command_buffer = as<VkCommandBuffer>(m_command_buffer);
    VkFence fence = as<VkFence>(m_fence);
    VkResult result = m_functions->vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS)
    {
        error.set("vkEndCommandBuffer: %s", result_name(result));
        return false;
    }
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    m_functions->vkResetFences(device, 1, &fence);
    result = m_functions->vkQueueSubmit(as<VkQueue>(m_queue), 1, &submit_info, fence);
    if (result != VK_SUCCESS)
    {
        error.set("vkQueueSubmit: %s", result_name(result));
        return false;
    }
    result = m_functions->vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS)
    {
        error.set("vkWaitForFences: %s", result_name(result));
        return false;
    }
    return true;
}

}
