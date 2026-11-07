#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/json.h"
#include "faisca/mapped_file.h"
#include "faisca/string_map.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace faisca
{

enum class SafetensorsDtype : uint32_t
{
    F32,
    F16,
    BF16,
    Other,
};

struct SafetensorsTensor
{
    static constexpr uint32_t MAX_DIMS = 8;

    std::string_view name;
    SafetensorsDtype dtype;
    std::string_view dtype_name;
    uint32_t n_dims;
    uint64_t shape[MAX_DIMS];
    uint64_t n_elements;
    const uint8_t* data; // into the mapped file
    uint64_t n_bytes;
};

// Hugging Face safetensors: 8 bytes of little endian header size, a JSON header
// {"name": {"dtype": ..., "shape": [...], "data_offsets": [begin, end]}, "__metadata__": {...}},
// then the data. One file, or the shards a model.safetensors.index.json lists ("weight_map").
class SafetensorsModel
{
public:
    static constexpr int MAX_SHARDS = 64;

    // a .safetensors file, or a .safetensors.index.json next to its shards
    bool open(const char* path, Error& error);

    size_t tensor_count() const
    {
        return m_tensors.size();
    }

    const SafetensorsTensor& tensor(size_t index) const
    {
        return m_tensors[index];
    }

    const SafetensorsTensor* find(std::string_view name) const;

private:
    bool open_shard(const char* path, Error& error);

    MappedFile m_files[MAX_SHARDS];
    JsonDocument m_headers[MAX_SHARDS];
    int m_shards = 0;
    Array<SafetensorsTensor> m_tensors;
    StringMap m_index;
};

}
