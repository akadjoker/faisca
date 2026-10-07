#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace faisca
{

// bump allocator for bytes whose address must never change
class Arena
{
public:
    Arena() = default;

    ~Arena()
    {
        release();
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    char* allocate(size_t size)
    {
        if (m_head == nullptr || m_head->capacity - m_head->used < size)
        {
            const size_t capacity = size > BLOCK_BYTES ? size : BLOCK_BYTES;
            Block* block = static_cast<Block*>(std::malloc(sizeof(Block) + capacity));
            if (block == nullptr)
            {
                return nullptr;
            }
            block->next = m_head;
            block->used = 0;
            block->capacity = capacity;
            m_head = block;
        }

        char* bytes = reinterpret_cast<char*>(m_head + 1) + m_head->used;
        m_head->used += size;
        return bytes;
    }

    const char* copy(const char* data, size_t size)
    {
        char* bytes = allocate(size);
        if (bytes != nullptr && size != 0)
        {
            std::memcpy(bytes, data, size);
        }
        return bytes;
    }

    void release()
    {
        while (m_head != nullptr)
        {
            Block* next = m_head->next;
            std::free(m_head);
            m_head = next;
        }
    }

private:
    struct Block
    {
        Block* next;
        size_t used;
        size_t capacity;
    };

    static constexpr size_t BLOCK_BYTES = 64 * 1024;

    Block* m_head = nullptr;
};

}
