#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/mapped_file.h"
#include "faisca/string_map.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace faisca
{

enum class CkptDtype : uint32_t
{
    F32,
    F16,
    BF16,
};

const char* ckpt_dtype_name(CkptDtype dtype);

struct CkptTensor
{
    static constexpr uint32_t MAX_DIMS = 8;

    std::string_view name;
    CkptDtype dtype;
    uint32_t n_dims;
    uint64_t shape[MAX_DIMS];
    uint64_t n_elements;
    const uint8_t* data;
    uint64_t n_bytes;
};

// PyTorch zip checkpoints (torch.save, format version 3) holding a state dict
// of contiguous F32/F16/BF16 tensors in uncompressed entries; tensor data
// points into the file
class CkptFile
{
public:
    CkptFile() = default;

    CkptFile(const CkptFile&) = delete;
    CkptFile& operator=(const CkptFile&) = delete;

    bool open(const char* path, Error& error);

    // the buffer must outlive this object
    bool load(const uint8_t* data, size_t size, Error& error);

    size_t tensor_count() const
    {
        return m_tensors.size();
    }

    const CkptTensor& tensor(size_t index) const
    {
        return m_tensors[index];
    }

    const CkptTensor* find(std::string_view name) const;

    const uint8_t* data() const
    {
        return m_data;
    }

    size_t size() const
    {
        return m_size;
    }

private:
    MappedFile m_file;
    const uint8_t* m_data = nullptr;
    size_t m_size = 0;
    Array<CkptTensor> m_tensors;
    StringMap m_index;
};

}
