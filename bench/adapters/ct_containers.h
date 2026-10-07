#pragma once

#include <ct/hashmap.hpp>
#include <ct/span.hpp>
#include <ct/vector.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace faisca
{

template <typename T>
class Array
{
public:
    T* data()
    {
        return m_items.begin();
    }

    const T* data() const
    {
        return m_items.begin();
    }

    size_t size() const
    {
        return m_items.size();
    }

    size_t capacity() const
    {
        return m_items.capacity();
    }

    bool empty() const
    {
        return m_items.empty();
    }

    T& operator[](size_t index)
    {
        return m_items[index];
    }

    const T& operator[](size_t index) const
    {
        return m_items[index];
    }

    T* begin()
    {
        return m_items.begin();
    }

    T* end()
    {
        return m_items.end();
    }

    const T* begin() const
    {
        return m_items.begin();
    }

    const T* end() const
    {
        return m_items.end();
    }

    bool reserve(size_t capacity)
    {
        m_items.reserve(capacity);
        return true;
    }

    bool push_back(const T& value)
    {
        m_items.push_back(value);
        return true;
    }

    void push_back_unchecked(const T& value)
    {
        m_items.push_back(value);
    }

    bool resize_uninitialized(size_t size)
    {
        m_items.resize(size);
        return true;
    }

    void clear()
    {
        m_items.clear();
    }

    void release()
    {
        m_items = ct::Vector<T>();
    }

private:
    ct::Vector<T> m_items;
};

class StringMap
{
public:
    enum class Insert
    {
        Added,
        Exists,
        Failed,
    };

    size_t size() const
    {
        return m_map.size();
    }

    bool reserve(size_t count)
    {
        m_map.reserve(count);
        return true;
    }

    Insert insert(std::string_view key, uint32_t value)
    {
        const ct::StringView ct_key(key.data(), key.size());
        if (m_map.find(ct_key) != nullptr)
        {
            return Insert::Exists;
        }
        m_map.put(ct_key, value);
        return Insert::Added;
    }

    bool find(std::string_view key, uint32_t& value) const
    {
        const uint32_t* found = m_map.find(ct::StringView(key.data(), key.size()));
        if (found == nullptr)
        {
            return false;
        }
        value = *found;
        return true;
    }

    void release()
    {
        m_map = ct::HashMap<ct::StringView, uint32_t>();
    }

private:
    ct::HashMap<ct::StringView, uint32_t> m_map;
};

}
