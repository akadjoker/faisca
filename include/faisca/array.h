#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace faisca
{

template <typename T>
class Array
{
    static_assert(std::is_trivially_copyable<T>::value, "faisca::Array only holds trivially copyable types");
    static_assert(alignof(T) <= alignof(std::max_align_t), "faisca::Array does not support over-aligned types");

public:
    Array() = default;

    ~Array()
    {
        std::free(m_data);
    }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    T* data()
    {
        return m_data;
    }

    const T* data() const
    {
        return m_data;
    }

    size_t size() const
    {
        return m_size;
    }

    size_t capacity() const
    {
        return m_capacity;
    }

    bool empty() const
    {
        return m_size == 0;
    }

    T& operator[](size_t index)
    {
        return m_data[index];
    }

    const T& operator[](size_t index) const
    {
        return m_data[index];
    }

    T* begin()
    {
        return m_data;
    }

    T* end()
    {
        return m_data + m_size;
    }

    const T* begin() const
    {
        return m_data;
    }

    const T* end() const
    {
        return m_data + m_size;
    }

    bool reserve(size_t capacity)
    {
        if (capacity <= m_capacity)
        {
            return true;
        }
        return reallocate(capacity);
    }

    bool push_back(const T& value)
    {
        if (m_size == m_capacity)
        {
            // value may point into this array
            const T copy = value;
            if (!grow())
            {
                return false;
            }
            m_data[m_size++] = copy;
            return true;
        }
        m_data[m_size++] = value;
        return true;
    }

    void push_back_unchecked(const T& value)
    {
        m_data[m_size++] = value;
    }

    // values must not point into this array
    bool append(const T* values, size_t count)
    {
        if (count > m_capacity - m_size && !grow_to(m_size + count))
        {
            return false;
        }
        if (count != 0)
        {
            std::memcpy(m_data + m_size, values, count * sizeof(T));
            m_size += count;
        }
        return true;
    }

    bool resize_uninitialized(size_t size)
    {
        if (!reserve(size))
        {
            return false;
        }
        m_size = size;
        return true;
    }

    void clear()
    {
        m_size = 0;
    }

    void swap(Array& other)
    {
        T* data = m_data;
        const size_t size = m_size;
        const size_t capacity = m_capacity;
        m_data = other.m_data;
        m_size = other.m_size;
        m_capacity = other.m_capacity;
        other.m_data = data;
        other.m_size = size;
        other.m_capacity = capacity;
    }

    void release()
    {
        std::free(m_data);
        m_data = nullptr;
        m_size = 0;
        m_capacity = 0;
    }

private:
    bool grow()
    {
        const size_t max_capacity = static_cast<size_t>(-1) / sizeof(T);
        if (m_capacity > max_capacity / 2)
        {
            return false;
        }
        return reallocate(m_capacity == 0 ? 8 : m_capacity * 2);
    }

    bool grow_to(size_t needed)
    {
        const size_t max_capacity = static_cast<size_t>(-1) / sizeof(T);
        if (needed > max_capacity)
        {
            return false;
        }
        size_t capacity = m_capacity == 0 ? 8 : m_capacity;
        while (capacity < needed)
        {
            capacity = capacity > max_capacity / 2 ? needed : capacity * 2;
        }
        return reallocate(capacity);
    }

    bool reallocate(size_t capacity)
    {
        if (capacity > static_cast<size_t>(-1) / sizeof(T))
        {
            return false;
        }
        void* block = std::realloc(m_data, capacity * sizeof(T));
        if (block == nullptr)
        {
            return false;
        }
        m_data = static_cast<T*>(block);
        m_capacity = capacity;
        return true;
    }

    T* m_data = nullptr;
    size_t m_size = 0;
    size_t m_capacity = 0;
};

}
