#include "faisca/json.h"

#include <cmath>

namespace faisca
{

namespace
{

constexpr int MAX_DEPTH = 256;

bool append_utf8(char* out, size_t& length, uint32_t code)
{
    if (code < 0x80)
    {
        out[length++] = static_cast<char>(code);
    }
    else if (code < 0x800)
    {
        out[length++] = static_cast<char>(0xc0 | (code >> 6));
        out[length++] = static_cast<char>(0x80 | (code & 0x3f));
    }
    else if (code < 0x10000)
    {
        out[length++] = static_cast<char>(0xe0 | (code >> 12));
        out[length++] = static_cast<char>(0x80 | ((code >> 6) & 0x3f));
        out[length++] = static_cast<char>(0x80 | (code & 0x3f));
    }
    else if (code < 0x110000)
    {
        out[length++] = static_cast<char>(0xf0 | (code >> 18));
        out[length++] = static_cast<char>(0x80 | ((code >> 12) & 0x3f));
        out[length++] = static_cast<char>(0x80 | ((code >> 6) & 0x3f));
        out[length++] = static_cast<char>(0x80 | (code & 0x3f));
    }
    else
    {
        return false;
    }
    return true;
}

int hex_digit(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f')
    {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F')
    {
        return c - 'A' + 10;
    }
    return -1;
}

}

bool JsonDocument::fail(const char* what)
{
    if (m_error->empty())
    {
        m_error->set("JSON: %s at byte %zu", what, m_pos);
    }
    return false;
}

uint32_t JsonDocument::add(const JsonValue& value)
{
    if (!m_values.push_back(value))
    {
        m_out_of_memory = true;
        return JSON_NONE;
    }
    return static_cast<uint32_t>(m_values.size() - 1);
}

void JsonDocument::skip_space()
{
    while (m_pos < m_size &&
           (m_data[m_pos] == ' ' || m_data[m_pos] == '\t' || m_data[m_pos] == '\n' || m_data[m_pos] == '\r'))
    {
        ++m_pos;
    }
}

bool JsonDocument::parse(const char* data, size_t size, Error& error)
{
    m_values.clear();
    m_strings.release();
    m_data = data;
    m_size = size;
    m_pos = 0;
    m_error = &error;
    m_out_of_memory = false;
    m_root = JSON_NONE;
    skip_space();
    uint32_t root = JSON_NONE;
    if (!parse_value(root, 0))
    {
        return false;
    }
    skip_space();
    if (m_pos != m_size)
    {
        return fail("trailing characters");
    }
    m_root = root;
    return true;
}

bool JsonDocument::parse_string(std::string_view& out)
{
    // m_data[m_pos] is the opening quote
    const size_t start = ++m_pos;
    bool escaped = false;
    while (m_pos < m_size && m_data[m_pos] != '"')
    {
        const unsigned char c = static_cast<unsigned char>(m_data[m_pos]);
        if (c < 0x20)
        {
            return fail("control character in a string");
        }
        if (c == '\\')
        {
            escaped = true;
            m_pos += 2;
            continue;
        }
        ++m_pos;
    }
    if (m_pos >= m_size)
    {
        return fail("unterminated string");
    }
    const size_t end = m_pos++;
    if (!escaped)
    {
        out = std::string_view(m_data + start, end - start);
        return true;
    }
    // decoded text is never longer than the escaped one
    char* buffer = m_strings.allocate(end - start + 1);
    if (buffer == nullptr)
    {
        m_out_of_memory = true;
        return fail("out of memory");
    }
    size_t length = 0;
    for (size_t i = start; i < end; ++i)
    {
        if (m_data[i] != '\\')
        {
            buffer[length++] = m_data[i];
            continue;
        }
        if (++i >= end)
        {
            return fail("bad escape");
        }
        switch (m_data[i])
        {
        case '"':
            buffer[length++] = '"';
            break;
        case '\\':
            buffer[length++] = '\\';
            break;
        case '/':
            buffer[length++] = '/';
            break;
        case 'b':
            buffer[length++] = '\b';
            break;
        case 'f':
            buffer[length++] = '\f';
            break;
        case 'n':
            buffer[length++] = '\n';
            break;
        case 'r':
            buffer[length++] = '\r';
            break;
        case 't':
            buffer[length++] = '\t';
            break;
        case 'u':
        {
            const auto read4 = [&](size_t at, uint32_t& code) {
                if (at + 4 > end)
                {
                    return false;
                }
                code = 0;
                for (size_t k = 0; k < 4; ++k)
                {
                    const int d = hex_digit(m_data[at + k]);
                    if (d < 0)
                    {
                        return false;
                    }
                    code = code * 16 + static_cast<uint32_t>(d);
                }
                return true;
            };
            uint32_t code = 0;
            if (!read4(i + 1, code))
            {
                return fail("bad \\u escape");
            }
            i += 4;
            if (code >= 0xd800 && code < 0xdc00)
            {
                uint32_t low = 0;
                if (i + 2 < end && m_data[i + 1] == '\\' && m_data[i + 2] == 'u' && read4(i + 3, low) &&
                    low >= 0xdc00 && low < 0xe000)
                {
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                    i += 6;
                }
                else
                {
                    return fail("lone surrogate");
                }
            }
            else if (code >= 0xdc00 && code < 0xe000)
            {
                return fail("lone surrogate");
            }
            if (!append_utf8(buffer, length, code))
            {
                return fail("bad code point");
            }
            break;
        }
        default:
            return fail("bad escape");
        }
    }
    out = std::string_view(buffer, length);
    return true;
}

bool JsonDocument::parse_number(JsonValue& out)
{
    const size_t start = m_pos;
    bool negative = false;
    if (m_data[m_pos] == '-')
    {
        negative = true;
        ++m_pos;
    }
    if (m_pos >= m_size || m_data[m_pos] < '0' || m_data[m_pos] > '9')
    {
        return fail("bad number");
    }
    if (m_data[m_pos] == '0' && m_pos + 1 < m_size && m_data[m_pos + 1] >= '0' && m_data[m_pos + 1] <= '9')
    {
        return fail("leading zero");
    }
    uint64_t mantissa = 0;
    bool exact = true;
    int digits_dropped = 0;
    while (m_pos < m_size && m_data[m_pos] >= '0' && m_data[m_pos] <= '9')
    {
        const uint64_t d = static_cast<uint64_t>(m_data[m_pos] - '0');
        if (mantissa <= (UINT64_MAX - d) / 10)
        {
            mantissa = mantissa * 10 + d;
        }
        else
        {
            exact = false;
            ++digits_dropped;
        }
        ++m_pos;
    }
    int exponent = digits_dropped;
    bool fraction = false;
    if (m_pos < m_size && m_data[m_pos] == '.')
    {
        fraction = true;
        ++m_pos;
        if (m_pos >= m_size || m_data[m_pos] < '0' || m_data[m_pos] > '9')
        {
            return fail("bad fraction");
        }
        while (m_pos < m_size && m_data[m_pos] >= '0' && m_data[m_pos] <= '9')
        {
            const uint64_t d = static_cast<uint64_t>(m_data[m_pos] - '0');
            if (mantissa <= (UINT64_MAX - d) / 10)
            {
                mantissa = mantissa * 10 + d;
                --exponent;
            }
            ++m_pos;
        }
    }
    if (m_pos < m_size && (m_data[m_pos] == 'e' || m_data[m_pos] == 'E'))
    {
        fraction = true;
        ++m_pos;
        bool negative_exponent = false;
        if (m_pos < m_size && (m_data[m_pos] == '+' || m_data[m_pos] == '-'))
        {
            negative_exponent = m_data[m_pos] == '-';
            ++m_pos;
        }
        if (m_pos >= m_size || m_data[m_pos] < '0' || m_data[m_pos] > '9')
        {
            return fail("bad exponent");
        }
        int e = 0;
        while (m_pos < m_size && m_data[m_pos] >= '0' && m_data[m_pos] <= '9')
        {
            if (e < 100000)
            {
                e = e * 10 + (m_data[m_pos] - '0');
            }
            ++m_pos;
        }
        exponent += negative_exponent ? -e : e;
    }
    out.kind = JsonKind::Number;
    out.number = static_cast<double>(mantissa) * std::pow(10.0, exponent);
    if (negative)
    {
        out.number = -out.number;
    }
    out.integral = !fraction && exact && mantissa <= (negative ? 0x8000000000000000ull : 0x7fffffffffffffffull);
    if (out.integral)
    {
        out.integer = negative ? static_cast<int64_t>(0 - mantissa) : static_cast<int64_t>(mantissa);
    }
    out.text = std::string_view(m_data + start, m_pos - start);
    return true;
}

bool JsonDocument::parse_value(uint32_t& out, int depth)
{
    if (depth > MAX_DEPTH)
    {
        return fail("nested too deep");
    }
    skip_space();
    if (m_pos >= m_size)
    {
        return fail("unexpected end");
    }
    JsonValue value;
    value.first = JSON_NONE;
    value.next = JSON_NONE;
    const char c = m_data[m_pos];
    const auto literal = [&](const char* word, size_t length, JsonKind kind) {
        if (m_size - m_pos < length || std::string_view(m_data + m_pos, length) != std::string_view(word, length))
        {
            return fail("bad literal");
        }
        m_pos += length;
        value.kind = kind;
        out = add(value);
        return out != JSON_NONE || fail("out of memory");
    };
    if (c == 'n')
    {
        return literal("null", 4, JsonKind::Null);
    }
    if (c == 't')
    {
        return literal("true", 4, JsonKind::True);
    }
    if (c == 'f')
    {
        return literal("false", 5, JsonKind::False);
    }
    if (c == '"')
    {
        value.kind = JsonKind::String;
        if (!parse_string(value.text))
        {
            return false;
        }
        out = add(value);
        return out != JSON_NONE || fail("out of memory");
    }
    if (c == '-' || (c >= '0' && c <= '9'))
    {
        if (!parse_number(value))
        {
            return false;
        }
        out = add(value);
        return out != JSON_NONE || fail("out of memory");
    }
    if (c != '[' && c != '{')
    {
        return fail("unexpected character");
    }
    const bool object = c == '{';
    value.kind = object ? JsonKind::Object : JsonKind::Array;
    const uint32_t self = add(value);
    if (self == JSON_NONE)
    {
        return fail("out of memory");
    }
    ++m_pos;
    skip_space();
    const char close = object ? '}' : ']';
    uint32_t last = JSON_NONE;
    uint32_t count = 0;
    const auto link = [&](uint32_t child) {
        if (last == JSON_NONE)
        {
            m_values[self].first = child;
        }
        else
        {
            m_values[last].next = child;
        }
        last = child;
        ++count;
    };
    if (m_pos < m_size && m_data[m_pos] == close)
    {
        ++m_pos;
        out = self;
        return true;
    }
    for (;;)
    {
        if (object)
        {
            skip_space();
            if (m_pos >= m_size || m_data[m_pos] != '"')
            {
                return fail("object key expected");
            }
            JsonValue key;
            key.kind = JsonKind::String;
            key.first = JSON_NONE;
            key.next = JSON_NONE;
            if (!parse_string(key.text))
            {
                return false;
            }
            const uint32_t key_index = add(key);
            if (key_index == JSON_NONE)
            {
                return fail("out of memory");
            }
            link(key_index);
            skip_space();
            if (m_pos >= m_size || m_data[m_pos] != ':')
            {
                return fail("':' expected");
            }
            ++m_pos;
        }
        uint32_t child = JSON_NONE;
        if (!parse_value(child, depth + 1))
        {
            return false;
        }
        link(child);
        skip_space();
        if (m_pos < m_size && m_data[m_pos] == ',')
        {
            ++m_pos;
            continue;
        }
        if (m_pos < m_size && m_data[m_pos] == close)
        {
            ++m_pos;
            break;
        }
        return fail(object ? "',' or '}' expected" : "',' or ']' expected");
    }
    m_values[self].count = count;
    out = self;
    return true;
}

uint32_t JsonDocument::find(uint32_t object, std::string_view key) const
{
    if (object == JSON_NONE || m_values[object].kind != JsonKind::Object)
    {
        return JSON_NONE;
    }
    for (uint32_t k = m_values[object].first; k != JSON_NONE;)
    {
        const uint32_t v = m_values[k].next;
        if (m_values[k].text == key)
        {
            return v;
        }
        k = m_values[v].next;
    }
    return JSON_NONE;
}

}
