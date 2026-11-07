#pragma once

#include "faisca/arena.h"
#include "faisca/array.h"
#include "faisca/error.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace faisca
{

enum class JsonKind : uint8_t
{
    Null,
    False,
    True,
    Number,
    String,
    Array,
    Object,
};

// one node; object members are a string node (the key) followed by its value
struct JsonValue
{
    JsonKind kind = JsonKind::Null;
    bool integral = false; // number without fraction or exponent that fits an int64
    uint32_t first = 0;    // first child (arrays, objects), JSON_NONE when empty
    uint32_t next = 0;     // next sibling, JSON_NONE at the end
    uint32_t count = 0;    // children: elements, or 2 * members
    int64_t integer = 0;
    double number = 0.0;
    std::string_view text; // strings, decoded to UTF-8
};

constexpr uint32_t JSON_NONE = 0xffffffffu;

// RFC 8259 JSON, the whole document in memory; strings without escapes point into the input,
// which must outlive the document
class JsonDocument
{
public:
    bool parse(const char* data, size_t size, Error& error);

    uint32_t root() const
    {
        return m_root;
    }

    const JsonValue& value(uint32_t index) const
    {
        return m_values[index];
    }

    // the value of member `key` of an object, JSON_NONE when absent or not an object
    uint32_t find(uint32_t object, std::string_view key) const;

private:
    bool parse_value(uint32_t& out, int depth);
    bool parse_string(std::string_view& out);
    bool parse_number(JsonValue& out);
    void skip_space();
    bool fail(const char* what);
    uint32_t add(const JsonValue& value);

    Array<JsonValue> m_values;
    Arena m_strings;
    const char* m_data = nullptr;
    size_t m_size = 0;
    size_t m_pos = 0;
    uint32_t m_root = JSON_NONE;
    Error* m_error = nullptr;
    bool m_out_of_memory = false;
};

}
