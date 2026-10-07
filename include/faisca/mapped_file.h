#pragma once

#include "faisca/error.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

class MappedFile
{
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool open(const char* path, Error& error);

    void close();

    bool is_open() const
    {
        return m_data != nullptr;
    }

    const uint8_t* data() const
    {
        return m_data;
    }

    size_t size() const
    {
        return m_size;
    }

private:
    const uint8_t* m_data = nullptr;
    size_t m_size = 0;
};

}
