#pragma once

#include <cstddef>
#include <cstdint>

namespace faisca
{

constexpr uint8_t UNICODE_LETTER = 1;
constexpr uint8_t UNICODE_NUMBER = 2;
constexpr uint8_t UNICODE_WHITESPACE = 4;

constexpr uint32_t UNICODE_REPLACEMENT = 0xfffd;
constexpr uint32_t UNICODE_LIMIT = 0x110000;

extern const uint16_t UNICODE_BLOCK_INDEX[4352];
extern const uint8_t UNICODE_BLOCKS[][256];

inline uint8_t unicode_class(uint32_t code_point)
{
    if (code_point >= UNICODE_LIMIT)
    {
        return 0;
    }
    return UNICODE_BLOCKS[UNICODE_BLOCK_INDEX[code_point >> 8]][code_point & 0xff];
}

// structural check only, like llama.cpp: overlong forms and surrogates decode,
// a malformed byte yields U+FFFD and advances one byte
inline uint32_t utf8_decode(const char* data, size_t size, size_t& offset)
{
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    const uint8_t first = bytes[offset];

    if ((first & 0x80) == 0)
    {
        offset += 1;
        return first;
    }
    if ((first & 0x40) != 0)
    {
        if ((first & 0x20) == 0)
        {
            if (offset + 1 < size && (bytes[offset + 1] & 0xc0) == 0x80)
            {
                const uint32_t code_point = (static_cast<uint32_t>(first & 0x1f) << 6) | (bytes[offset + 1] & 0x3f);
                offset += 2;
                return code_point;
            }
        }
        else if ((first & 0x10) == 0)
        {
            if (offset + 2 < size && (bytes[offset + 1] & 0xc0) == 0x80 && (bytes[offset + 2] & 0xc0) == 0x80)
            {
                const uint32_t code_point = (static_cast<uint32_t>(first & 0x0f) << 12) |
                                            (static_cast<uint32_t>(bytes[offset + 1] & 0x3f) << 6) |
                                            (bytes[offset + 2] & 0x3f);
                offset += 3;
                return code_point;
            }
        }
        else if ((first & 0x08) == 0)
        {
            if (offset + 3 < size && (bytes[offset + 1] & 0xc0) == 0x80 && (bytes[offset + 2] & 0xc0) == 0x80 &&
                (bytes[offset + 3] & 0xc0) == 0x80)
            {
                const uint32_t code_point = (static_cast<uint32_t>(first & 0x07) << 18) |
                                            (static_cast<uint32_t>(bytes[offset + 1] & 0x3f) << 12) |
                                            (static_cast<uint32_t>(bytes[offset + 2] & 0x3f) << 6) |
                                            (bytes[offset + 3] & 0x3f);
                offset += 4;
                return code_point;
            }
        }
    }

    offset += 1;
    return UNICODE_REPLACEMENT;
}

// returns the number of bytes written (1 to 4)
inline size_t utf8_encode(uint32_t code_point, char* out)
{
    // llama.cpp throws here, a malformed 4 byte sequence can decode to this
    if (code_point >= UNICODE_LIMIT)
    {
        code_point = UNICODE_REPLACEMENT;
    }

    if (code_point <= 0x7f)
    {
        out[0] = static_cast<char>(code_point);
        return 1;
    }
    if (code_point <= 0x7ff)
    {
        out[0] = static_cast<char>(0xc0 | (code_point >> 6));
        out[1] = static_cast<char>(0x80 | (code_point & 0x3f));
        return 2;
    }
    if (code_point <= 0xffff)
    {
        out[0] = static_cast<char>(0xe0 | (code_point >> 12));
        out[1] = static_cast<char>(0x80 | ((code_point >> 6) & 0x3f));
        out[2] = static_cast<char>(0x80 | (code_point & 0x3f));
        return 3;
    }
    out[0] = static_cast<char>(0xf0 | (code_point >> 18));
    out[1] = static_cast<char>(0x80 | ((code_point >> 12) & 0x3f));
    out[2] = static_cast<char>(0x80 | ((code_point >> 6) & 0x3f));
    out[3] = static_cast<char>(0x80 | (code_point & 0x3f));
    return 4;
}

}
