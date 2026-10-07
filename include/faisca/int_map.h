#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace faisca
{

class IntMap
{
public:
    enum class Insert
    {
        Added,
        Exists,
        Failed,
    };

    IntMap() = default;

    ~IntMap()
    {
        std::free(m_slots);
    }

    IntMap(const IntMap&) = delete;
    IntMap& operator=(const IntMap&) = delete;

    size_t size() const
    {
        return m_size;
    }

    bool reserve(size_t count)
    {
        size_t capacity = MIN_CAPACITY;
        while (capacity / 2 < count)
        {
            if (capacity > static_cast<size_t>(-1) / (2 * sizeof(Slot)))
            {
                return false;
            }
            capacity *= 2;
        }
        return capacity <= m_capacity || rehash(capacity);
    }

    Insert insert(uint64_t key, uint32_t value)
    {
        if ((m_size + 1) * 2 > m_capacity)
        {
            if (!reserve(m_size + 1))
            {
                return Insert::Failed;
            }
        }

        size_t index = static_cast<size_t>(mix(key)) & m_mask;
        while (m_slots[index].used != 0)
        {
            if (m_slots[index].key == key)
            {
                return Insert::Exists;
            }
            index = (index + 1) & m_mask;
        }

        m_slots[index].key = key;
        m_slots[index].value = value;
        m_slots[index].used = 1;
        m_size += 1;
        return Insert::Added;
    }

    bool find(uint64_t key, uint32_t& value) const
    {
        if (m_size == 0)
        {
            return false;
        }

        size_t index = static_cast<size_t>(mix(key)) & m_mask;
        while (m_slots[index].used != 0)
        {
            if (m_slots[index].key == key)
            {
                value = m_slots[index].value;
                return true;
            }
            index = (index + 1) & m_mask;
        }
        return false;
    }

    void release()
    {
        std::free(m_slots);
        m_slots = nullptr;
        m_capacity = 0;
        m_mask = 0;
        m_size = 0;
    }

private:
    struct Slot
    {
        uint64_t key;
        uint32_t value;
        uint32_t used;
    };

    static constexpr size_t MIN_CAPACITY = 16;

    static uint64_t mix(uint64_t key)
    {
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdull;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53ull;
        key ^= key >> 33;
        return key;
    }

    bool rehash(size_t capacity)
    {
        Slot* slots = static_cast<Slot*>(std::calloc(capacity, sizeof(Slot)));
        if (slots == nullptr)
        {
            return false;
        }

        const size_t mask = capacity - 1;
        for (size_t i = 0; i < m_capacity; ++i)
        {
            if (m_slots[i].used == 0)
            {
                continue;
            }
            size_t index = static_cast<size_t>(mix(m_slots[i].key)) & mask;
            while (slots[index].used != 0)
            {
                index = (index + 1) & mask;
            }
            slots[index] = m_slots[i];
        }

        std::free(m_slots);
        m_slots = slots;
        m_capacity = capacity;
        m_mask = mask;
        return true;
    }

    Slot* m_slots = nullptr;
    size_t m_capacity = 0;
    size_t m_mask = 0;
    size_t m_size = 0;
};

}
