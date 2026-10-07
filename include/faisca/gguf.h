#pragma once

#include "faisca/error.h"
#include "faisca/mapped_file.h"

// benchmark builds swap the containers, see bench/
#if defined(FAISCA_BENCH_CONTAINERS_HEADER)
#include FAISCA_BENCH_CONTAINERS_HEADER
#else
#include "faisca/array.h"
#include "faisca/string_map.h"
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace faisca
{

enum class GgufType : uint32_t
{
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
};

enum class GgmlType : uint32_t
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
};

struct GgmlTypeInfo
{
    const char* name;
    uint32_t block_elements;
    uint32_t block_bytes;
};

const GgmlTypeInfo* ggml_type_info(GgmlType type);

const char* gguf_type_name(GgufType type);

size_t gguf_scalar_size(GgufType type);

uint64_t checksum64(const uint8_t* data, size_t size);

struct GgufArray
{
    GgufType elem_type = GgufType::UInt8;
    uint64_t count = 0;
    const uint8_t* data = nullptr; // unaligned
    uint64_t byte_size = 0;

    template <typename T>
    bool copy_to(Array<T>& out) const;

    bool strings_to(Array<std::string_view>& out) const;
};

struct GgufValue
{
    GgufType type = GgufType::UInt8;
    uint64_t u64 = 0;
    int64_t i64 = 0;
    double f64 = 0.0;
    std::string_view str;
    GgufArray array;
};

struct GgufKeyValue
{
    std::string_view key;
    GgufValue value;
};

constexpr uint32_t GGUF_MAX_DIMS = 4;

struct GgufTensor
{
    std::string_view name;
    uint32_t n_dims = 0;
    uint64_t dims[GGUF_MAX_DIMS] = {1, 1, 1, 1};
    GgmlType type = GgmlType::F32;
    uint64_t offset = 0; // from data_offset()
    uint64_t n_elements = 0;
    uint64_t n_bytes = 0;
    const uint8_t* data = nullptr;
};

class GgufFile
{
public:
    GgufFile();
    ~GgufFile();

    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    bool open(const char* path, Error& error);

    bool parse(const uint8_t* data, size_t size, Error& error);

    void close();

    bool is_open() const
    {
        return m_data != nullptr;
    }

    uint32_t version() const
    {
        return m_version;
    }

    uint32_t alignment() const
    {
        return m_alignment;
    }

    uint64_t data_offset() const
    {
        return m_data_offset;
    }

    size_t file_size() const
    {
        return m_size;
    }

    const Array<GgufKeyValue>& metadata() const
    {
        return m_metadata;
    }

    const Array<GgufTensor>& tensors() const
    {
        return m_tensors;
    }

    const GgufValue* find(std::string_view key) const;
    const GgufTensor* find_tensor(std::string_view name) const;

    bool get_string(std::string_view key, std::string_view& out) const;
    bool get_bool(std::string_view key, bool& out) const;
    bool get_float(std::string_view key, double& out) const;

    bool get_uint(std::string_view key, uint64_t& out) const;
    bool get_int(std::string_view key, int64_t& out) const;

    const GgufArray* get_array(std::string_view key) const;

private:
    bool parse_image(Error& error);

    MappedFile m_mapping;
    const uint8_t* m_data = nullptr;
    size_t m_size = 0;

    uint32_t m_version = 0;
    uint32_t m_alignment = 32;
    uint64_t m_data_offset = 0;

    Array<GgufKeyValue> m_metadata;
    Array<GgufTensor> m_tensors;
    StringMap m_metadata_index;
    StringMap m_tensor_index;
};

namespace detail
{

template <typename T>
struct GgufTypeOf;

template <>
struct GgufTypeOf<uint8_t>
{
    static constexpr GgufType value = GgufType::UInt8;
};

template <>
struct GgufTypeOf<int8_t>
{
    static constexpr GgufType value = GgufType::Int8;
};

template <>
struct GgufTypeOf<uint16_t>
{
    static constexpr GgufType value = GgufType::UInt16;
};

template <>
struct GgufTypeOf<int16_t>
{
    static constexpr GgufType value = GgufType::Int16;
};

template <>
struct GgufTypeOf<uint32_t>
{
    static constexpr GgufType value = GgufType::UInt32;
};

template <>
struct GgufTypeOf<int32_t>
{
    static constexpr GgufType value = GgufType::Int32;
};

template <>
struct GgufTypeOf<float>
{
    static constexpr GgufType value = GgufType::Float32;
};

template <>
struct GgufTypeOf<uint64_t>
{
    static constexpr GgufType value = GgufType::UInt64;
};

template <>
struct GgufTypeOf<int64_t>
{
    static constexpr GgufType value = GgufType::Int64;
};

template <>
struct GgufTypeOf<double>
{
    static constexpr GgufType value = GgufType::Float64;
};

}

template <typename T>
bool GgufArray::copy_to(Array<T>& out) const
{
    if (elem_type != detail::GgufTypeOf<T>::value)
    {
        return false;
    }
    if (!out.resize_uninitialized(static_cast<size_t>(count)))
    {
        return false;
    }
    if (count != 0)
    {
        std::memcpy(out.data(), data, static_cast<size_t>(count) * sizeof(T));
    }
    return true;
}

}
