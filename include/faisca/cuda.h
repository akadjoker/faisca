#pragma once

#include "faisca/array.h"
#include "faisca/error.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

class CudaDevice;

struct CudaBuffer
{
    uint64_t address = 0;
    size_t bytes = 0;

private:
    friend class CudaDevice;
    const CudaDevice* owner = nullptr;
    uint64_t id = 0;
};

struct CudaHostBuffer
{
    void* data = nullptr;
    size_t bytes = 0;

private:
    friend class CudaDevice;
    const CudaDevice* owner = nullptr;
    uint64_t id = 0;
};

struct CudaStream
{
    void* handle = nullptr;

private:
    friend class CudaDevice;
    const CudaDevice* owner = nullptr;
    uint64_t id = 0;
};

struct CudaEvent
{
    void* handle = nullptr;

private:
    friend class CudaDevice;
    const CudaDevice* owner = nullptr;
    uint64_t id = 0;
};

struct CudaGraph
{
    void* handle = nullptr;

private:
    friend class CudaDevice;
    const CudaDevice* owner = nullptr;
    uint64_t id = 0;
};

struct CudaModule
{
    void* handle = nullptr;

private:
    friend class CudaDevice;
    const CudaDevice* owner = nullptr;
    uint64_t id = 0;
};

struct CudaKernel
{
    void* handle = nullptr;

private:
    friend class CudaDevice;
    const CudaDevice* owner = nullptr;
    uint64_t module_id = 0;
};

struct CudaLaunch
{
    uint32_t grid_x = 1;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t block_x = 1;
    uint32_t block_y = 1;
    uint32_t block_z = 1;
    uint32_t shared_bytes = 0;
};

class CudaDevice
{
public:
    CudaDevice() = default;
    ~CudaDevice();

    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(const CudaDevice&) = delete;

    bool init(Error& error, int device_index = 0, const char* driver_path = nullptr);
    void release();

    bool ready() const
    {
        return m_context != nullptr;
    }

    const char* device_name() const
    {
        return m_name;
    }

    int compute_major() const
    {
        return m_major;
    }

    int compute_minor() const
    {
        return m_minor;
    }

    int driver_version() const
    {
        return m_driver_version;
    }

    bool memory_info(size_t& free_bytes, size_t& total_bytes, Error& error);
    bool create_buffer(size_t bytes, CudaBuffer& out, Error& error);
    bool destroy_buffer(CudaBuffer& buffer, Error& error);
    bool create_host_buffer(size_t bytes, CudaHostBuffer& out, Error& error);
    bool destroy_host_buffer(CudaHostBuffer& buffer, Error& error);
    bool create_stream(CudaStream& out, Error& error);
    bool destroy_stream(CudaStream& stream, Error& error);
    bool synchronize(Error& error);
    bool synchronize(const CudaStream& stream, Error& error);
    bool create_event(CudaEvent& out, Error& error);
    bool destroy_event(CudaEvent& event, Error& error);
    bool record_event(const CudaEvent& event, const CudaStream& stream, Error& error);
    bool wait_event(const CudaStream& stream, const CudaEvent& event, Error& error);
    bool event_ready(const CudaEvent& event, bool& complete, Error& error);
    bool synchronize(const CudaEvent& event, Error& error);
    // Captures the launches made on the stream between the two calls into an executable graph.
    bool begin_capture(const CudaStream& stream, Error& error);
    bool end_capture(const CudaStream& stream, CudaGraph& out, Error& error);
    bool launch_graph(const CudaGraph& graph, const CudaStream& stream, Error& error);
    bool destroy_graph(CudaGraph& graph, Error& error);
    bool load_module(const char* ptx, CudaModule& out, Error& error);
    bool destroy_module(CudaModule& module, Error& error);
    bool find_kernel(const CudaModule& module, const char* name, CudaKernel& out, Error& error);
    bool launch(const CudaKernel& kernel, const CudaStream& stream, const CudaLaunch& configuration,
                void** arguments, Error& error);

    bool upload(const CudaBuffer& dst, const void* data, size_t bytes, Error& error, size_t offset = 0);
    bool download(const CudaBuffer& src, void* data, size_t bytes, Error& error, size_t offset = 0);
    bool upload_async(const CudaBuffer& dst, const CudaHostBuffer& src, size_t bytes,
                      const CudaStream& stream, Error& error, size_t dst_offset = 0, size_t src_offset = 0);
    bool download_async(const CudaBuffer& src, const CudaHostBuffer& dst, size_t bytes,
                        const CudaStream& stream, Error& error, size_t src_offset = 0, size_t dst_offset = 0);

private:
    enum class Kind
    {
        buffer,
        host,
        stream,
        event,
        module,
        graph
    };

    struct Resource
    {
        Kind kind = Kind::buffer;
        uint64_t id = 0;
        uint64_t address = 0;
        void* handle = nullptr;
        size_t bytes = 0;
    };

    struct Functions;
    struct ContextScope;

    bool load_library(const char* path, Error& error);
    bool check(int result, const char* operation, Error& error) const;
    bool track(Resource& resource, Error& error);
    Resource* find(uint64_t id, Kind kind, const CudaDevice* owner, Error& error);
    bool range(const Resource& resource, size_t offset, size_t bytes, Error& error) const;
    bool destroy(Resource& resource, Error& error);

    Functions* m_functions = nullptr;
    void* m_library = nullptr;
    void* m_context = nullptr;
    Array<Resource> m_resources;
    uint64_t m_next_id = 1;
    char m_name[256] = {};
    int m_major = 0;
    int m_minor = 0;
    int m_driver_version = 0;
};

}