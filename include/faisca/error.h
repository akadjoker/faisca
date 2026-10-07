#pragma once

#include <cstddef>

#if defined(__MINGW32__)
#define FAISCA_PRINTF_FORMAT(format_index, first_argument) \
    __attribute__((format(gnu_printf, format_index, first_argument)))
#elif defined(__GNUC__) || defined(__clang__)
#define FAISCA_PRINTF_FORMAT(format_index, first_argument) \
    __attribute__((format(printf, format_index, first_argument)))
#else
#define FAISCA_PRINTF_FORMAT(format_index, first_argument)
#endif

namespace faisca
{

struct Error
{
    static constexpr size_t CAPACITY = 512;

    char message[CAPACITY];

    Error()
    {
        message[0] = '\0';
    }

    bool empty() const
    {
        return message[0] == '\0';
    }

    const char* c_str() const
    {
        return message;
    }

    void clear()
    {
        message[0] = '\0';
    }

    // never pass this object's own message as an argument
    void set(const char* format, ...) FAISCA_PRINTF_FORMAT(2, 3);
};

}
