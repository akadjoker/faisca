#include "faisca/error.h"

#include <cstdarg>
#include <cstdio>

namespace faisca
{

void Error::set(const char* format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    const int written = std::vsnprintf(message, CAPACITY, format, arguments);
    va_end(arguments);

    if (written < 0)
    {
        message[0] = '\0';
    }
    else if (static_cast<size_t>(written) >= CAPACITY)
    {
        message[CAPACITY - 4] = '.';
        message[CAPACITY - 3] = '.';
        message[CAPACITY - 2] = '.';
        message[CAPACITY - 1] = '\0';
    }
}

}
