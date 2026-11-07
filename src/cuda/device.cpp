#include "faisca/cuda.h"

#include <new>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define FAISCA_CUDA_CALL __stdcall
#else
#include <dlfcn.h>
#define FAISCA_CUDA_CALL
#endif

namespace faisca
{

static_assert(sizeof(void*) == 8, "CUDA requires a 64-bit host");

#define FAISCA_CUDA_FUNCTIONS(X) \
    X(cuInit, (unsigned int), "cuInit") \
    X(cuDriverGetVersion, (int*), "cuDriverGetVersion") \
    X(cuDeviceGetCount, (int*), "cuDeviceGetCount") \
    X(cuDeviceGet, (int*, int), "cuDeviceGet") \
    X(cuDeviceGetName, (char*, int, int), "cuDeviceGetName") \
    X(cuDeviceGetAttribute, (int*, int, int), "cuDeviceGetAttribute") \
    X(cuCtxCreate, (void**, unsigned int, int), "cuCtxCreate_v2") \
    X(cuCtxDestroy, (void*), "cuCtxDestroy_v2") \
    X(cuCtxPushCurrent, (void*), "cuCtxPushCurrent_v2") \
    X(cuCtxPopCurrent, (void**), "cuCtxPopCurrent_v2") \
    X(cuCtxSynchronize, (), "cuCtxSynchronize") \
    X(cuMemGetInfo, (size_t*, size_t*), "cuMemGetInfo_v2") \
    X(cuMemAlloc, (uint64_t*, size_t), "cuMemAlloc_v2") \
    X(cuMemFree, (uint64_t), "cuMemFree_v2") \
    X(cuMemHostAlloc, (void**, size_t, unsigned int), "cuMemHostAlloc") \
    X(cuMemFreeHost, (void*), "cuMemFreeHost") \
    X(cuMemcpyHtoD, (uint64_t, const void*, size_t), "cuMemcpyHtoD_v2") \
    X(cuMemcpyDtoH, (void*, uint64_t, size_t), "cuMemcpyDtoH_v2") \
    X(cuMemcpyHtoDAsync, (uint64_t, const void*, size_t, void*), "cuMemcpyHtoDAsync_v2") \
    X(cuMemcpyDtoHAsync, (void*, uint64_t, size_t, void*), "cuMemcpyDtoHAsync_v2") \
    X(cuStreamCreate, (void**, unsigned int), "cuStreamCreate") \
    X(cuStreamDestroy, (void*), "cuStreamDestroy_v2") \
    X(cuStreamSynchronize, (void*), "cuStreamSynchronize") \
    X(cuStreamWaitEvent, (void*, void*, unsigned int), "cuStreamWaitEvent") \
    X(cuEventCreate, (void**, unsigned int), "cuEventCreate") \
    X(cuEventDestroy, (void*), "cuEventDestroy_v2") \
    X(cuEventRecord, (void*, void*), "cuEventRecord") \
    X(cuEventQuery, (void*), "cuEventQuery") \
    X(cuEventSynchronize, (void*), "cuEventSynchronize") \
    X(cuModuleLoadData, (void**, const void*), "cuModuleLoadData") \
    X(cuModuleUnload, (void*), "cuModuleUnload") \
    X(cuModuleGetFunction, (void**, void*, const char*), "cuModuleGetFunction") \
    X(cuLaunchKernel, (void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, \
                     unsigned int, void*, void**, void**), "cuLaunchKernel") \
    X(cuStreamBeginCapture, (void*, int), "cuStreamBeginCapture_v2") \
    X(cuStreamEndCapture, (void*, void**), "cuStreamEndCapture") \
    X(cuGraphInstantiate, (void**, void*, unsigned long long), "cuGraphInstantiateWithFlags") \
    X(cuGraphDestroy, (void*), "cuGraphDestroy") \
    X(cuGraphLaunch, (void*, void*), "cuGraphLaunch") \
    X(cuGraphExecDestroy, (void*), "cuGraphExecDestroy") \
    X(cuGetErrorName, (int, const char**), "cuGetErrorName") \
    X(cuGetErrorString, (int, const char**), "cuGetErrorString")

struct CudaDevice::Functions
{
#define FAISCA_CUDA_DECLARE(name, arguments, symbol) int (FAISCA_CUDA_CALL* name) arguments = nullptr;
    FAISCA_CUDA_FUNCTIONS(FAISCA_CUDA_DECLARE)
#undef FAISCA_CUDA_DECLARE
};

struct CudaDevice::ContextScope
{
    Functions* functions;
    bool active = false;

    ContextScope(CudaDevice& device, Error& error) : functions(device.m_functions)
    {
        if (!device.ready())
        {
            error.set("CUDA device is not initialized");
            return;
        }
        active = device.check(functions->cuCtxPushCurrent(device.m_context), "cuCtxPushCurrent", error);
    }

    ~ContextScope()
    {
        if (active)
        {
            void* popped = nullptr;
            functions->cuCtxPopCurrent(&popped);
        }
    }
};

CudaDevice::~CudaDevice()
{
    release();
}

bool CudaDevice::check(int result, const char* operation, Error& error) const
{
    if (result == 0)
    {
        return true;
    }
    const char* name = "CUDA_ERROR_UNKNOWN";
    const char* description = "unknown error";
    m_functions->cuGetErrorName(result, &name);
    m_functions->cuGetErrorString(result, &description);
    error.set("%s: %s (%d): %s", operation, name != nullptr ? name : "CUDA_ERROR_UNKNOWN", result,
              description != nullptr ? description : "unknown error");
    return false;
}

bool CudaDevice::load_library(const char* path, Error& error)
{
    m_functions = new (std::nothrow) Functions();
    if (m_functions == nullptr)
    {
        error.set("CUDA function table allocation failed");
        return false;
    }
#if defined(_WIN32)
    const char* filename = path != nullptr ? path : "nvcuda.dll";
    HMODULE library = LoadLibraryA(filename);
    if (library == nullptr)
    {
        error.set("CUDA driver %s could not be loaded (Windows error %lu)", filename,
                  static_cast<unsigned long>(GetLastError()));
        return false;
    }
    m_library = library;
#define FAISCA_CUDA_LOAD(name, arguments, symbol) \
    m_functions->name = reinterpret_cast<decltype(m_functions->name)>(reinterpret_cast<void*>(GetProcAddress(library, symbol))); \
    if (m_functions->name == nullptr) \
    { \
        error.set("CUDA driver has no %s", symbol); \
        return false; \
    }
#else
    const char* filename = path != nullptr ? path : "libcuda.so.1";
    m_library = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
    if (m_library == nullptr)
    {
        error.set("CUDA driver %s could not be loaded: %s", filename, dlerror());
        return false;
    }
#define FAISCA_CUDA_LOAD(name, arguments, symbol) \
    m_functions->name = reinterpret_cast<decltype(m_functions->name)>(dlsym(m_library, symbol)); \
    if (m_functions->name == nullptr) \
    { \
        error.set("CUDA driver has no %s", symbol); \
        return false; \
    }
#endif
    FAISCA_CUDA_FUNCTIONS(FAISCA_CUDA_LOAD)
#undef FAISCA_CUDA_LOAD
    return true;
}

bool CudaDevice::init(Error& error, int device_index, const char* driver_path)
{
    release();
    if (!load_library(driver_path, error))
    {
        release();
        return false;
    }
    int count = 0;
    int device = 0;
    if (!check(m_functions->cuInit(0), "cuInit", error) ||
        !check(m_functions->cuDriverGetVersion(&m_driver_version), "cuDriverGetVersion", error) ||
        !check(m_functions->cuDeviceGetCount(&count), "cuDeviceGetCount", error))
    {
        release();
        return false;
    }
    if (device_index < 0 || device_index >= count)
    {
        error.set("CUDA device index %d is outside the available %d devices", device_index, count);
        release();
        return false;
    }
    if (!check(m_functions->cuDeviceGet(&device, device_index), "cuDeviceGet", error) ||
        !check(m_functions->cuDeviceGetName(m_name, static_cast<int>(sizeof(m_name)), device), "cuDeviceGetName", error) ||
        !check(m_functions->cuDeviceGetAttribute(&m_major, 75, device), "cuDeviceGetAttribute major", error) ||
        !check(m_functions->cuDeviceGetAttribute(&m_minor, 76, device), "cuDeviceGetAttribute minor", error) ||
        !check(m_functions->cuCtxCreate(&m_context, 0, device), "cuCtxCreate", error))
    {
        release();
        return false;
    }
    void* popped = nullptr;
    if (!check(m_functions->cuCtxPopCurrent(&popped), "cuCtxPopCurrent", error))
    {
        release();
        return false;
    }
    return true;
}

void CudaDevice::release()
{
    if (m_context != nullptr)
    {
        const bool pushed = m_functions->cuCtxPushCurrent(m_context) == 0;
        if (pushed)
        {
            m_functions->cuCtxSynchronize();
            for (size_t index = m_resources.size(); index > 0; --index)
            {
                const Resource& resource = m_resources[index - 1];
                if (resource.id == 0)
                {
                    continue;
                }
                switch (resource.kind)
                {
                    case Kind::buffer:
                        m_functions->cuMemFree(resource.address);
                        break;
                    case Kind::host:
                        m_functions->cuMemFreeHost(resource.handle);
                        break;
                    case Kind::stream:
                        m_functions->cuStreamDestroy(resource.handle);
                        break;
                    case Kind::event:
                        m_functions->cuEventDestroy(resource.handle);
                        break;
                    case Kind::module:
                        m_functions->cuModuleUnload(resource.handle);
                        break;
                    case Kind::graph:
                        m_functions->cuGraphExecDestroy(resource.handle);
                        break;
                }
            }
            void* popped = nullptr;
            m_functions->cuCtxPopCurrent(&popped);
        }
        m_functions->cuCtxDestroy(m_context);
        if (!pushed)
        {
            for (const Resource& resource : m_resources)
            {
                if (resource.id != 0 && resource.kind == Kind::host)
                {
                    m_functions->cuMemFreeHost(resource.handle);
                }
            }
        }
    }
    m_context = nullptr;
    m_resources.clear();
    if (m_library != nullptr)
    {
#if defined(_WIN32)
        FreeLibrary(reinterpret_cast<HMODULE>(m_library));
#else
        dlclose(m_library);
#endif
    }
    m_library = nullptr;
    delete m_functions;
    m_functions = nullptr;
    m_name[0] = '\0';
    m_major = 0;
    m_minor = 0;
    m_driver_version = 0;
}

bool CudaDevice::track(Resource& resource, Error& error)
{
    if (m_next_id == 0)
    {
        error.set("CUDA resource id space exhausted");
        return false;
    }
    resource.id = m_next_id++;
    for (Resource& slot : m_resources)
    {
        if (slot.id == 0)
        {
            slot = resource;
            return true;
        }
    }
    if (!m_resources.push_back(resource))
    {
        error.set("CUDA resource tracking allocation failed");
        return false;
    }
    return true;
}

CudaDevice::Resource* CudaDevice::find(uint64_t id, Kind kind, const CudaDevice* owner, Error& error)
{
    if (ready() && owner == this && id != 0)
    {
        for (Resource& resource : m_resources)
        {
            if (resource.id == id && resource.kind == kind)
            {
                return &resource;
            }
        }
    }
    error.set("CUDA resource is stale or belongs to another device");
    return nullptr;
}

bool CudaDevice::range(const Resource& resource, size_t offset, size_t bytes, Error& error) const
{
    if (offset > resource.bytes || bytes > resource.bytes - offset)
    {
        error.set("CUDA transfer exceeds the buffer bounds");
        return false;
    }
    return true;
}

bool CudaDevice::destroy(Resource& resource, Error& error)
{
    ContextScope scope(*this, error);
    if (!scope.active || !check(m_functions->cuCtxSynchronize(), "cuCtxSynchronize before free", error))
    {
        return false;
    }
    int result = 0;
    switch (resource.kind)
    {
        case Kind::buffer:
            result = m_functions->cuMemFree(resource.address);
            break;
        case Kind::host:
            result = m_functions->cuMemFreeHost(resource.handle);
            break;
        case Kind::stream:
            result = m_functions->cuStreamDestroy(resource.handle);
            break;
        case Kind::event:
            result = m_functions->cuEventDestroy(resource.handle);
            break;
        case Kind::module:
            result = m_functions->cuModuleUnload(resource.handle);
            break;
        case Kind::graph:
            result = m_functions->cuGraphExecDestroy(resource.handle);
            break;
    }
    if (!check(result, "CUDA resource destruction", error))
    {
        return false;
    }
    resource.id = 0;
    return true;
}

bool CudaDevice::memory_info(size_t& free_bytes, size_t& total_bytes, Error& error)
{
    free_bytes = 0;
    total_bytes = 0;
    ContextScope scope(*this, error);
    return scope.active && check(m_functions->cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo", error);
}

bool CudaDevice::create_buffer(size_t bytes, CudaBuffer& out, Error& error)
{
    if (bytes == 0 || out.id != 0)
    {
        error.set("CUDA buffer size must be positive and the output must be empty");
        return false;
    }
    ContextScope scope(*this, error);
    Resource resource;
    resource.bytes = bytes;
    if (!scope.active || !check(m_functions->cuMemAlloc(&resource.address, bytes), "cuMemAlloc", error))
    {
        return false;
    }
    if (!track(resource, error))
    {
        m_functions->cuMemFree(resource.address);
        return false;
    }
    out.address = resource.address;
    out.bytes = bytes;
    out.owner = this;
    out.id = resource.id;
    return true;
}

bool CudaDevice::destroy_buffer(CudaBuffer& buffer, Error& error)
{
    if (buffer.id != 0)
    {
        Resource* resource = find(buffer.id, Kind::buffer, buffer.owner, error);
        if (resource == nullptr || !destroy(*resource, error))
        {
            return false;
        }
    }
    buffer = CudaBuffer();
    return true;
}

bool CudaDevice::create_host_buffer(size_t bytes, CudaHostBuffer& out, Error& error)
{
    if (bytes == 0 || out.id != 0)
    {
        error.set("CUDA host buffer size must be positive and the output must be empty");
        return false;
    }
    ContextScope scope(*this, error);
    Resource resource;
    resource.kind = Kind::host;
    resource.bytes = bytes;
    if (!scope.active || !check(m_functions->cuMemHostAlloc(&resource.handle, bytes, 0), "cuMemHostAlloc", error))
    {
        return false;
    }
    if (!track(resource, error))
    {
        m_functions->cuMemFreeHost(resource.handle);
        return false;
    }
    out.data = resource.handle;
    out.bytes = bytes;
    out.owner = this;
    out.id = resource.id;
    return true;
}

bool CudaDevice::destroy_host_buffer(CudaHostBuffer& buffer, Error& error)
{
    if (buffer.id != 0)
    {
        Resource* resource = find(buffer.id, Kind::host, buffer.owner, error);
        if (resource == nullptr || !destroy(*resource, error))
        {
            return false;
        }
    }
    buffer = CudaHostBuffer();
    return true;
}

bool CudaDevice::create_stream(CudaStream& out, Error& error)
{
    if (out.id != 0)
    {
        error.set("CUDA stream output must be empty");
        return false;
    }
    ContextScope scope(*this, error);
    Resource resource;
    resource.kind = Kind::stream;
    if (!scope.active || !check(m_functions->cuStreamCreate(&resource.handle, 1), "cuStreamCreate", error))
    {
        return false;
    }
    if (!track(resource, error))
    {
        m_functions->cuStreamDestroy(resource.handle);
        return false;
    }
    out.handle = resource.handle;
    out.owner = this;
    out.id = resource.id;
    return true;
}

bool CudaDevice::destroy_stream(CudaStream& stream, Error& error)
{
    if (stream.id != 0)
    {
        Resource* resource = find(stream.id, Kind::stream, stream.owner, error);
        if (resource == nullptr || !destroy(*resource, error))
        {
            return false;
        }
    }
    stream = CudaStream();
    return true;
}

bool CudaDevice::synchronize(Error& error)
{
    ContextScope scope(*this, error);
    return scope.active && check(m_functions->cuCtxSynchronize(), "cuCtxSynchronize", error);
}

bool CudaDevice::synchronize(const CudaStream& stream, Error& error)
{
    Resource* resource = find(stream.id, Kind::stream, stream.owner, error);
    ContextScope scope(*this, error);
    return resource != nullptr && scope.active &&
           check(m_functions->cuStreamSynchronize(resource->handle), "cuStreamSynchronize", error);
}

bool CudaDevice::create_event(CudaEvent& out, Error& error)
{
    if (out.id != 0)
    {
        error.set("CUDA event output must be empty");
        return false;
    }
    ContextScope scope(*this, error);
    Resource resource;
    resource.kind = Kind::event;
    if (!scope.active || !check(m_functions->cuEventCreate(&resource.handle, 2), "cuEventCreate", error))
    {
        return false;
    }
    if (!track(resource, error))
    {
        m_functions->cuEventDestroy(resource.handle);
        return false;
    }
    out.handle = resource.handle;
    out.owner = this;
    out.id = resource.id;
    return true;
}

bool CudaDevice::destroy_event(CudaEvent& event, Error& error)
{
    if (event.id != 0)
    {
        Resource* resource = find(event.id, Kind::event, event.owner, error);
        if (resource == nullptr || !destroy(*resource, error))
        {
            return false;
        }
    }
    event = CudaEvent();
    return true;
}

bool CudaDevice::begin_capture(const CudaStream& stream, Error& error)
{
    Resource* queue = find(stream.id, Kind::stream, stream.owner, error);
    ContextScope scope(*this, error);
    // CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
    return queue != nullptr && scope.active &&
           check(m_functions->cuStreamBeginCapture(queue->handle, 1), "cuStreamBeginCapture", error);
}

bool CudaDevice::end_capture(const CudaStream& stream, CudaGraph& out, Error& error)
{
    Resource* queue = find(stream.id, Kind::stream, stream.owner, error);
    ContextScope scope(*this, error);
    if (queue == nullptr || !scope.active)
    {
        return false;
    }
    void* graph = nullptr;
    if (!check(m_functions->cuStreamEndCapture(queue->handle, &graph), "cuStreamEndCapture", error))
    {
        return false;
    }
    if (out.id != 0)
    {
        m_functions->cuGraphDestroy(graph);
        error.set("CUDA graph output must be empty");
        return false;
    }
    Resource resource;
    resource.kind = Kind::graph;
    const bool instantiated = check(m_functions->cuGraphInstantiate(&resource.handle, graph, 0), "cuGraphInstantiate", error);
    m_functions->cuGraphDestroy(graph);
    if (!instantiated)
    {
        return false;
    }
    if (!track(resource, error))
    {
        m_functions->cuGraphExecDestroy(resource.handle);
        return false;
    }
    out.handle = resource.handle;
    out.owner = this;
    out.id = resource.id;
    return true;
}

bool CudaDevice::launch_graph(const CudaGraph& graph, const CudaStream& stream, Error& error)
{
    Resource* exec = find(graph.id, Kind::graph, graph.owner, error);
    Resource* queue = find(stream.id, Kind::stream, stream.owner, error);
    ContextScope scope(*this, error);
    return exec != nullptr && queue != nullptr && scope.active &&
           check(m_functions->cuGraphLaunch(exec->handle, queue->handle), "cuGraphLaunch", error);
}

bool CudaDevice::destroy_graph(CudaGraph& graph, Error& error)
{
    if (graph.id != 0)
    {
        Resource* resource = find(graph.id, Kind::graph, graph.owner, error);
        if (resource == nullptr || !destroy(*resource, error))
        {
            return false;
        }
    }
    graph = CudaGraph();
    return true;
}

bool CudaDevice::record_event(const CudaEvent& event, const CudaStream& stream, Error& error)
{
    Resource* marker = find(event.id, Kind::event, event.owner, error);
    Resource* queue = find(stream.id, Kind::stream, stream.owner, error);
    ContextScope scope(*this, error);
    return marker != nullptr && queue != nullptr && scope.active &&
           check(m_functions->cuEventRecord(marker->handle, queue->handle), "cuEventRecord", error);
}

bool CudaDevice::wait_event(const CudaStream& stream, const CudaEvent& event, Error& error)
{
    Resource* queue = find(stream.id, Kind::stream, stream.owner, error);
    Resource* marker = find(event.id, Kind::event, event.owner, error);
    ContextScope scope(*this, error);
    return queue != nullptr && marker != nullptr && scope.active &&
           check(m_functions->cuStreamWaitEvent(queue->handle, marker->handle, 0), "cuStreamWaitEvent", error);
}

bool CudaDevice::event_ready(const CudaEvent& event, bool& complete, Error& error)
{
    complete = false;
    Resource* marker = find(event.id, Kind::event, event.owner, error);
    ContextScope scope(*this, error);
    if (marker == nullptr || !scope.active)
    {
        return false;
    }
    const int result = m_functions->cuEventQuery(marker->handle);
    if (result == 600)
    {
        return true;
    }
    complete = result == 0;
    return check(result, "cuEventQuery", error);
}

bool CudaDevice::synchronize(const CudaEvent& event, Error& error)
{
    Resource* marker = find(event.id, Kind::event, event.owner, error);
    ContextScope scope(*this, error);
    return marker != nullptr && scope.active &&
           check(m_functions->cuEventSynchronize(marker->handle), "cuEventSynchronize", error);
}

bool CudaDevice::load_module(const char* ptx, CudaModule& out, Error& error)
{
    if (out.id != 0 || ptx == nullptr || ptx[0] == '\0')
    {
        error.set("CUDA module requires PTX and an empty output");
        return false;
    }
    ContextScope scope(*this, error);
    Resource resource;
    resource.kind = Kind::module;
    if (!scope.active || !check(m_functions->cuModuleLoadData(&resource.handle, ptx), "cuModuleLoadData", error))
    {
        return false;
    }
    if (!track(resource, error))
    {
        m_functions->cuModuleUnload(resource.handle);
        return false;
    }
    out.handle = resource.handle;
    out.owner = this;
    out.id = resource.id;
    return true;
}

bool CudaDevice::destroy_module(CudaModule& module, Error& error)
{
    if (module.id != 0)
    {
        Resource* resource = find(module.id, Kind::module, module.owner, error);
        if (resource == nullptr || !destroy(*resource, error))
        {
            return false;
        }
    }
    module = CudaModule();
    return true;
}

bool CudaDevice::find_kernel(const CudaModule& module, const char* name, CudaKernel& out, Error& error)
{
    out = CudaKernel();
    Resource* resource = find(module.id, Kind::module, module.owner, error);
    if (name == nullptr || name[0] == '\0')
    {
        error.set("CUDA kernel name is empty");
        return false;
    }
    ContextScope scope(*this, error);
    if (resource == nullptr || !scope.active ||
        !check(m_functions->cuModuleGetFunction(&out.handle, resource->handle, name), "cuModuleGetFunction", error))
    {
        return false;
    }
    out.owner = this;
    out.module_id = module.id;
    return true;
}

bool CudaDevice::launch(const CudaKernel& kernel, const CudaStream& stream, const CudaLaunch& configuration,
                        void** arguments, Error& error)
{
    Resource* module = find(kernel.module_id, Kind::module, kernel.owner, error);
    Resource* queue = find(stream.id, Kind::stream, stream.owner, error);
    if (configuration.grid_x == 0 || configuration.grid_y == 0 || configuration.grid_z == 0 ||
        configuration.block_x == 0 || configuration.block_y == 0 || configuration.block_z == 0)
    {
        error.set("CUDA launch dimensions must be positive");
        return false;
    }
    ContextScope scope(*this, error);
    return module != nullptr && queue != nullptr && kernel.handle != nullptr && scope.active &&
           check(m_functions->cuLaunchKernel(kernel.handle, configuration.grid_x, configuration.grid_y,
                 configuration.grid_z, configuration.block_x, configuration.block_y, configuration.block_z,
                 configuration.shared_bytes, queue->handle, arguments, nullptr), "cuLaunchKernel", error);
}

bool CudaDevice::upload(const CudaBuffer& dst, const void* data, size_t bytes, Error& error, size_t offset)
{
    Resource* resource = find(dst.id, Kind::buffer, dst.owner, error);
    if (resource == nullptr || !range(*resource, offset, bytes, error))
    {
        return false;
    }
    if (bytes != 0 && data == nullptr)
    {
        error.set("CUDA upload data is null");
        return false;
    }
    ContextScope scope(*this, error);
    return scope.active && (bytes == 0 ||
           (check(m_functions->cuMemcpyHtoD(resource->address + offset, data, bytes), "cuMemcpyHtoD", error) &&
            check(m_functions->cuCtxSynchronize(), "cuCtxSynchronize after upload", error)));
}

bool CudaDevice::download(const CudaBuffer& src, void* data, size_t bytes, Error& error, size_t offset)
{
    Resource* resource = find(src.id, Kind::buffer, src.owner, error);
    if (resource == nullptr || !range(*resource, offset, bytes, error))
    {
        return false;
    }
    if (bytes != 0 && data == nullptr)
    {
        error.set("CUDA download data is null");
        return false;
    }
    ContextScope scope(*this, error);
    return scope.active && (bytes == 0 ||
           check(m_functions->cuMemcpyDtoH(data, resource->address + offset, bytes), "cuMemcpyDtoH", error));
}

bool CudaDevice::upload_async(const CudaBuffer& dst, const CudaHostBuffer& src, size_t bytes,
                              const CudaStream& stream, Error& error, size_t dst_offset, size_t src_offset)
{
    Resource* device = find(dst.id, Kind::buffer, dst.owner, error);
    Resource* host = find(src.id, Kind::host, src.owner, error);
    Resource* queue = find(stream.id, Kind::stream, stream.owner, error);
    if (device == nullptr || host == nullptr || queue == nullptr ||
        !range(*device, dst_offset, bytes, error) || !range(*host, src_offset, bytes, error))
    {
        return false;
    }
    ContextScope scope(*this, error);
    return scope.active && (bytes == 0 || check(m_functions->cuMemcpyHtoDAsync(
        device->address + dst_offset, static_cast<const char*>(host->handle) + src_offset, bytes, queue->handle),
        "cuMemcpyHtoDAsync", error));
}

bool CudaDevice::download_async(const CudaBuffer& src, const CudaHostBuffer& dst, size_t bytes,
                                const CudaStream& stream, Error& error, size_t src_offset, size_t dst_offset)
{
    Resource* device = find(src.id, Kind::buffer, src.owner, error);
    Resource* host = find(dst.id, Kind::host, dst.owner, error);
    Resource* queue = find(stream.id, Kind::stream, stream.owner, error);
    if (device == nullptr || host == nullptr || queue == nullptr ||
        !range(*device, src_offset, bytes, error) || !range(*host, dst_offset, bytes, error))
    {
        return false;
    }
    ContextScope scope(*this, error);
    return scope.active && (bytes == 0 || check(m_functions->cuMemcpyDtoHAsync(
        static_cast<char*>(host->handle) + dst_offset, device->address + src_offset, bytes, queue->handle),
        "cuMemcpyDtoHAsync", error));
}

#undef FAISCA_CUDA_FUNCTIONS
#undef FAISCA_CUDA_CALL

}