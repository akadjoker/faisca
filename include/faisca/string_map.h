#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace faisca
{

namespace detail
{

inline uint64_t load_u64(const char* data)
{
    uint64_t value = 0;
    std::memcpy(&value, data, 8);
    return value;
}

inline uint32_t load_u32(const char* data)
{
    uint32_t value = 0;
    std::memcpy(&value, data, 4);
    return value;
}

inline uint64_t load_partial(const char* data, size_t size)
{
    // two overlapping reads cover 4 to 7 bytes without reading past the end
    if (size >= 4)
    {
        const uint64_t low = load_u32(data);
        const uint64_t high = load_u32(data + size - 4);
        return low | (high << (8 * (size - 4)));
    }
    if (size != 0)
    {
        const uint64_t first = static_cast<unsigned char>(data[0]);
        const uint64_t middle = static_cast<unsigned char>(data[size >> 1]);
        const uint64_t last = static_cast<unsigned char>(data[size - 1]);
        return first | (middle << (8 * (size >> 1))) | (last << (8 * (size - 1)));
    }
    return 0;
}

constexpr uint64_t HASH_MULTIPLIER = 0xc6a4a7935bd1e995ull;
constexpr uint64_t HASH_SEED = 0x8445d61a4e774912ull;

inline uint64_t hash_round(uint64_t hash, uint64_t word)
{
    word *= HASH_MULTIPLIER;
    word ^= word >> 47;
    word *= HASH_MULTIPLIER;

    hash ^= word;
    hash *= HASH_MULTIPLIER;
    return hash;
}

inline uint64_t hash_finish(uint64_t hash)
{
    hash ^= hash >> 47;
    hash *= HASH_MULTIPLIER;
    hash ^= hash >> 47;
    return hash;
}

}

inline uint64_t hash_bytes(const char* data, size_t size)
{
    uint64_t hash = detail::HASH_SEED ^ (static_cast<uint64_t>(size) * detail::HASH_MULTIPLIER);

    size_t remaining = size;
    while (remaining >= 8)
    {
        hash = detail::hash_round(hash, detail::load_u64(data));
        data += 8;
        remaining -= 8;
    }
    if (remaining != 0)
    {
        hash = detail::hash_round(hash, detail::load_partial(data, remaining));
    }
    return detail::hash_finish(hash);
}

class StringMap
{
public:
    enum class Insert
    {
        Added,
        Exists,
        Failed,
    };

    StringMap() = default;

    ~StringMap()
    {
        std::free(m_block);
    }

    StringMap(const StringMap&) = delete;
    StringMap& operator=(const StringMap&) = delete;

    size_t size() const
    {
        return m_size;
    }

    bool reserve(size_t count)
    {
        size_t capacity = MIN_CAPACITY;
        while (capacity - capacity / 3 < count)
        {
            if (capacity > static_cast<size_t>(-1) / (2 * (sizeof(Slot) + 1)))
            {
                return false;
            }
            capacity *= 2;
        }
        return capacity <= m_capacity || rehash(capacity);
    }

    Insert insert(std::string_view key, uint32_t value)
    {
        if (key.size() >= EMPTY)
        {
            return Insert::Failed;
        }
        if (m_size + 1 > m_capacity - m_capacity / 3)
        {
            if (!reserve(m_size + 1))
            {
                return Insert::Failed;
            }
        }

        Slot wanted;
        const uint64_t hash = make_slot(key, wanted);
        wanted.value = value;
        const uint8_t tag = tag_of(hash);

        size_t index = static_cast<size_t>(hash) & m_mask;
        for (;;)
        {
            const uint8_t found = m_tags[index];
            if (found == 0)
            {
                break;
            }
            if (found == tag && same_key(m_slots[index], wanted, key))
            {
                return Insert::Exists;
            }
            index = (index + 1) & m_mask;
        }

        m_slots[index] = wanted;
        m_tags[index] = tag;
        m_size += 1;
        return Insert::Added;
    }

    bool find(std::string_view key, uint32_t& value) const
    {
        if (m_size == 0 || key.size() >= EMPTY)
        {
            return false;
        }

        Slot wanted;
        const uint64_t hash = make_slot(key, wanted);
        const uint8_t tag = tag_of(hash);

        size_t index = static_cast<size_t>(hash) & m_mask;
        for (;;)
        {
            const uint8_t found = m_tags[index];
            if (found == 0)
            {
                return false;
            }
            if (found == tag)
            {
                const Slot& slot = m_slots[index];
                if (same_key(slot, wanted, key))
                {
                    value = slot.value;
                    return true;
                }
            }
            index = (index + 1) & m_mask;
        }
    }

    void release()
    {
        std::free(m_block);
        m_block = nullptr;
        m_slots = nullptr;
        m_tags = nullptr;
        m_capacity = 0;
        m_mask = 0;
        m_size = 0;
    }

private:
    static constexpr size_t INLINE_BYTES = 16;
    static constexpr uint32_t EMPTY = 0xffffffffu;
    static constexpr size_t MIN_CAPACITY = 16;

    struct Slot
    {
        uint64_t a; // key bytes 0-7, or the address of a key longer than 16 bytes
        uint64_t b; // key bytes 8-15, or the hash of a key longer than 16 bytes
        uint32_t size;
        uint32_t value;
    };

    // never 0: a 0 tag marks a free slot
    static uint8_t tag_of(uint64_t hash)
    {
        return static_cast<uint8_t>((hash >> 57) | 0x80u);
    }

    static uint64_t make_slot(std::string_view key, Slot& slot)
    {
        const char* data = key.data();
        const size_t size = key.size();
        slot.size = static_cast<uint32_t>(size);

        if (size > INLINE_BYTES)
        {
            const uint64_t hash = hash_bytes(data, size);
            slot.a = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data));
            slot.b = hash;
            return hash;
        }

        uint64_t hash = detail::HASH_SEED ^ (static_cast<uint64_t>(size) * detail::HASH_MULTIPLIER);
        if (size > 8)
        {
            slot.a = detail::load_u64(data);
            slot.b = size == 16 ? detail::load_u64(data + 8) : detail::load_partial(data + 8, size - 8);
            hash = detail::hash_round(hash, slot.a);
            hash = detail::hash_round(hash, slot.b);
        }
        else
        {
            slot.a = size == 8 ? detail::load_u64(data) : detail::load_partial(data, size);
            slot.b = 0;
            hash = detail::hash_round(hash, slot.a);
        }
        return detail::hash_finish(hash);
    }

    static bool same_key(const Slot& slot, const Slot& wanted, std::string_view key)
    {
        if (slot.size != wanted.size || slot.b != wanted.b)
        {
            return false;
        }
        if (wanted.size <= INLINE_BYTES)
        {
            return slot.a == wanted.a;
        }
        const char* stored = reinterpret_cast<const char*>(static_cast<uintptr_t>(slot.a));
        return std::memcmp(stored, key.data(), key.size()) == 0;
    }

    static uint64_t hash_of(const Slot& slot)
    {
        if (slot.size > INLINE_BYTES)
        {
            return slot.b;
        }
        uint64_t hash = detail::HASH_SEED ^ (static_cast<uint64_t>(slot.size) * detail::HASH_MULTIPLIER);
        hash = detail::hash_round(hash, slot.a);
        if (slot.size > 8)
        {
            hash = detail::hash_round(hash, slot.b);
        }
        return detail::hash_finish(hash);
    }

    bool rehash(size_t capacity)
    {
        void* block = std::malloc(capacity * (sizeof(Slot) + 1));
        if (block == nullptr)
        {
            return false;
        }
        Slot* slots = static_cast<Slot*>(block);
        uint8_t* tags = reinterpret_cast<uint8_t*>(slots + capacity);
        std::memset(tags, 0, capacity);

        const size_t mask = capacity - 1;
        for (size_t i = 0; i < m_capacity; ++i)
        {
            if (m_tags[i] == 0)
            {
                continue;
            }
            size_t index = static_cast<size_t>(hash_of(m_slots[i])) & mask;
            while (tags[index] != 0)
            {
                index = (index + 1) & mask;
            }
            slots[index] = m_slots[i];
            tags[index] = m_tags[i];
        }

        std::free(m_block);
        m_block = block;
        m_slots = slots;
        m_tags = tags;
        m_capacity = capacity;
        m_mask = mask;
        return true;
    }

    void* m_block = nullptr;
    Slot* m_slots = nullptr;
    uint8_t* m_tags = nullptr;
    size_t m_capacity = 0;
    size_t m_mask = 0;
    size_t m_size = 0;
};

}
