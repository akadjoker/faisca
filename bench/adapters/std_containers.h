#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace faisca
{

template <typename T>
class Array
{
public:
    T* data()
    {
        return m_items.data();
    }

    const T* data() const
    {
        return m_items.data();
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
        return m_items.data();
    }

    T* end()
    {
        return m_items.data() + m_items.size();
    }

    const T* begin() const
    {
        return m_items.data();
    }

    const T* end() const
    {
        return m_items.data() + m_items.size();
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
        std::vector<T>().swap(m_items);
    }

private:
    std::vector<T> m_items;
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
        return m_map.emplace(key, value).second ? Insert::Added : Insert::Exists;
    }

    bool find(std::string_view key, uint32_t& value) const
    {
        const auto it = m_map.find(key);
        if (it == m_map.end())
        {
            return false;
        }
        value = it->second;
        return true;
    }

    void release()
    {
        std::unordered_map<std::string_view, uint32_t>().swap(m_map);
    }

private:
    std::unordered_map<std::string_view, uint32_t> m_map;
};

}
