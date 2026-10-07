#include "faisca/gguf.h"

#include <cinttypes>
#include <cstdio>
#include <limits>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "faisca only supports little-endian hosts"
#endif
#endif

namespace faisca
{

namespace
{

constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32;
constexpr uint32_t GGUF_MAX_TYPE_ID = 12;
constexpr int GGUF_MAX_ARRAY_DEPTH = 8;

constexpr uint64_t MIN_STRING_BYTES = 8;
constexpr uint64_t MIN_ARRAY_BYTES = 12;
constexpr uint64_t MIN_KEY_VALUE_BYTES = 13;
constexpr uint64_t MIN_TENSOR_INFO_BYTES = 24;

// counts come from the file: never reserve more than this up front
constexpr uint64_t MAX_RESERVE = 4096;

constexpr uint64_t MAX_ENTRIES = 0xfffffffeull;

class Reader
{
public:
    Reader(const uint8_t* data, size_t size)
        : m_data(data), m_size(size), m_pos(0)
    {
    }

    size_t pos() const
    {
        return m_pos;
    }

    uint64_t remaining() const
    {
        return m_size - m_pos;
    }

    const uint8_t* cursor() const
    {
        return m_data + m_pos;
    }

    template <typename T>
    bool read(T& out)
    {
        if (remaining() < sizeof(T))
        {
            return false;
        }
        std::memcpy(&out, m_data + m_pos, sizeof(T));
        m_pos += sizeof(T);
        return true;
    }

    bool skip(uint64_t count)
    {
        if (remaining() < count)
        {
            return false;
        }
        m_pos += static_cast<size_t>(count);
        return true;
    }

    // loading spends its time here (a vocabulary is ~100k strings), so the
    // cursor stays in a local: one load and one add per string
    bool skip_strings(uint64_t count)
    {
        const uint8_t* cursor = m_data + m_pos;
        const uint8_t* const end = m_data + m_size;

        for (uint64_t i = 0; i < count; ++i)
        {
            if (static_cast<size_t>(end - cursor) < sizeof(uint64_t))
            {
                return false;
            }
            uint64_t length = 0;
            std::memcpy(&length, cursor, sizeof(uint64_t));
            cursor += sizeof(uint64_t);

            if (static_cast<uint64_t>(end - cursor) < length)
            {
                return false;
            }
            cursor += static_cast<size_t>(length);
        }

        m_pos = static_cast<size_t>(cursor - m_data);
        return true;
    }

    bool read_string(std::string_view& out)
    {
        uint64_t length = 0;
        if (!read(length))
        {
            return false;
        }
        if (remaining() < length)
        {
            return false;
        }
        out = std::string_view(reinterpret_cast<const char*>(m_data + m_pos), static_cast<size_t>(length));
        m_pos += static_cast<size_t>(length);
        return true;
    }

private:
    const uint8_t* m_data;
    size_t m_size;
    size_t m_pos;
};

struct Quoted
{
    char text[104];
};

Quoted quoted(std::string_view text)
{
    constexpr size_t max_length = 96;

    Quoted out;
    size_t at = 0;
    out.text[at++] = '\'';
    for (size_t i = 0; i < text.size() && i < max_length; ++i)
    {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        out.text[at++] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '?';
    }
    if (text.size() > max_length)
    {
        out.text[at++] = '.';
        out.text[at++] = '.';
        out.text[at++] = '.';
    }
    out.text[at++] = '\'';
    out.text[at] = '\0';
    return out;
}

bool skip_elements(Reader& reader, GgufType elem_type, uint64_t count, int depth, const char*& reason)
{
    const size_t scalar = gguf_scalar_size(elem_type);
    if (scalar != 0)
    {
        if (count > reader.remaining() / scalar)
        {
            reason = "array is larger than the file";
            return false;
        }
        return reader.skip(count * scalar);
    }

    if (elem_type == GgufType::String)
    {
        if (count > reader.remaining() / MIN_STRING_BYTES)
        {
            reason = "string array is larger than the file";
            return false;
        }
        if (!reader.skip_strings(count))
        {
            reason = "string array is truncated";
            return false;
        }
        return true;
    }

    if (depth >= GGUF_MAX_ARRAY_DEPTH)
    {
        reason = "arrays are nested too deeply";
        return false;
    }
    if (count > reader.remaining() / MIN_ARRAY_BYTES)
    {
        reason = "nested array is larger than the file";
        return false;
    }
    for (uint64_t i = 0; i < count; ++i)
    {
        uint32_t inner_type = 0;
        uint64_t inner_count = 0;
        if (!reader.read(inner_type) || !reader.read(inner_count))
        {
            reason = "nested array is truncated";
            return false;
        }
        if (inner_type > GGUF_MAX_TYPE_ID)
        {
            reason = "nested array has an unknown element type";
            return false;
        }
        if (!skip_elements(reader, static_cast<GgufType>(inner_type), inner_count, depth + 1, reason))
        {
            return false;
        }
    }
    return true;
}

template <typename T>
bool read_scalar(Reader& reader, T& out, Error& error)
{
    if (!reader.read(out))
    {
        error.set("value is truncated");
        return false;
    }
    return true;
}

bool parse_value(Reader& reader, GgufType type, GgufValue& out, Error& error)
{
    out.type = type;

    switch (type)
    {
        case GgufType::UInt8:
        {
            uint8_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.u64 = v;
            return true;
        }
        case GgufType::Int8:
        {
            int8_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.i64 = v;
            return true;
        }
        case GgufType::UInt16:
        {
            uint16_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.u64 = v;
            return true;
        }
        case GgufType::Int16:
        {
            int16_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.i64 = v;
            return true;
        }
        case GgufType::UInt32:
        {
            uint32_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.u64 = v;
            return true;
        }
        case GgufType::Int32:
        {
            int32_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.i64 = v;
            return true;
        }
        case GgufType::Float32:
        {
            float v = 0.0f;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.f64 = static_cast<double>(v);
            return true;
        }
        case GgufType::Bool:
        {
            uint8_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            if (v > 1)
            {
                error.set("bool value is %u, expected 0 or 1", static_cast<unsigned>(v));
                return false;
            }
            out.u64 = v;
            return true;
        }
        case GgufType::String:
        {
            if (!reader.read_string(out.str))
            {
                error.set("string is truncated");
                return false;
            }
            return true;
        }
        case GgufType::Array:
        {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!reader.read(elem_type) || !reader.read(count))
            {
                error.set("array header is truncated");
                return false;
            }
            if (elem_type > GGUF_MAX_TYPE_ID)
            {
                error.set("array has unknown element type %" PRIu32, elem_type);
                return false;
            }

            out.array.elem_type = static_cast<GgufType>(elem_type);
            out.array.count = count;
            out.array.data = reader.cursor();

            const size_t start = reader.pos();
            const char* reason = "array is truncated";
            if (!skip_elements(reader, out.array.elem_type, count, 1, reason))
            {
                error.set("%s", reason);
                return false;
            }
            out.array.byte_size = reader.pos() - start;
            return true;
        }
        case GgufType::UInt64:
        {
            uint64_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.u64 = v;
            return true;
        }
        case GgufType::Int64:
        {
            int64_t v = 0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.i64 = v;
            return true;
        }
        case GgufType::Float64:
        {
            double v = 0.0;
            if (!read_scalar(reader, v, error))
            {
                return false;
            }
            out.f64 = v;
            return true;
        }
    }

    error.set("unknown value type %" PRIu32, static_cast<uint32_t>(type));
    return false;
}

bool is_unsigned(GgufType type)
{
    return type == GgufType::UInt8 || type == GgufType::UInt16 || type == GgufType::UInt32 ||
           type == GgufType::UInt64;
}

bool is_signed(GgufType type)
{
    return type == GgufType::Int8 || type == GgufType::Int16 || type == GgufType::Int32 ||
           type == GgufType::Int64;
}

}

const GgmlTypeInfo* ggml_type_info(GgmlType type)
{
    static const GgmlTypeInfo f32 = {"F32", 1, 4};
    static const GgmlTypeInfo f16 = {"F16", 1, 2};
    static const GgmlTypeInfo q4_0 = {"Q4_0", 32, 18};
    static const GgmlTypeInfo q4_1 = {"Q4_1", 32, 20};
    static const GgmlTypeInfo q5_0 = {"Q5_0", 32, 22};
    static const GgmlTypeInfo q5_1 = {"Q5_1", 32, 24};
    static const GgmlTypeInfo q8_0 = {"Q8_0", 32, 34};
    static const GgmlTypeInfo q8_1 = {"Q8_1", 32, 40};
    static const GgmlTypeInfo q2_k = {"Q2_K", 256, 84};
    static const GgmlTypeInfo q3_k = {"Q3_K", 256, 110};
    static const GgmlTypeInfo q4_k = {"Q4_K", 256, 144};
    static const GgmlTypeInfo q5_k = {"Q5_K", 256, 176};
    static const GgmlTypeInfo q6_k = {"Q6_K", 256, 210};
    static const GgmlTypeInfo q8_k = {"Q8_K", 256, 292};
    static const GgmlTypeInfo iq2_xxs = {"IQ2_XXS", 256, 66};
    static const GgmlTypeInfo iq2_xs = {"IQ2_XS", 256, 74};
    static const GgmlTypeInfo iq3_xxs = {"IQ3_XXS", 256, 98};
    static const GgmlTypeInfo iq1_s = {"IQ1_S", 256, 50};
    static const GgmlTypeInfo iq4_nl = {"IQ4_NL", 32, 18};
    static const GgmlTypeInfo iq3_s = {"IQ3_S", 256, 110};
    static const GgmlTypeInfo iq2_s = {"IQ2_S", 256, 82};
    static const GgmlTypeInfo iq4_xs = {"IQ4_XS", 256, 136};
    static const GgmlTypeInfo i8 = {"I8", 1, 1};
    static const GgmlTypeInfo i16 = {"I16", 1, 2};
    static const GgmlTypeInfo i32 = {"I32", 1, 4};
    static const GgmlTypeInfo i64 = {"I64", 1, 8};
    static const GgmlTypeInfo f64 = {"F64", 1, 8};
    static const GgmlTypeInfo iq1_m = {"IQ1_M", 256, 56};
    static const GgmlTypeInfo bf16 = {"BF16", 1, 2};
    static const GgmlTypeInfo tq1_0 = {"TQ1_0", 256, 54};
    static const GgmlTypeInfo tq2_0 = {"TQ2_0", 256, 66};
    static const GgmlTypeInfo mxfp4 = {"MXFP4", 32, 17};
    static const GgmlTypeInfo nvfp4 = {"NVFP4", 64, 36};
    static const GgmlTypeInfo q1_0 = {"Q1_0", 128, 18};

    switch (type)
    {
        case GgmlType::F32:
            return &f32;
        case GgmlType::F16:
            return &f16;
        case GgmlType::Q4_0:
            return &q4_0;
        case GgmlType::Q4_1:
            return &q4_1;
        case GgmlType::Q5_0:
            return &q5_0;
        case GgmlType::Q5_1:
            return &q5_1;
        case GgmlType::Q8_0:
            return &q8_0;
        case GgmlType::Q8_1:
            return &q8_1;
        case GgmlType::Q2_K:
            return &q2_k;
        case GgmlType::Q3_K:
            return &q3_k;
        case GgmlType::Q4_K:
            return &q4_k;
        case GgmlType::Q5_K:
            return &q5_k;
        case GgmlType::Q6_K:
            return &q6_k;
        case GgmlType::Q8_K:
            return &q8_k;
        case GgmlType::IQ2_XXS:
            return &iq2_xxs;
        case GgmlType::IQ2_XS:
            return &iq2_xs;
        case GgmlType::IQ3_XXS:
            return &iq3_xxs;
        case GgmlType::IQ1_S:
            return &iq1_s;
        case GgmlType::IQ4_NL:
            return &iq4_nl;
        case GgmlType::IQ3_S:
            return &iq3_s;
        case GgmlType::IQ2_S:
            return &iq2_s;
        case GgmlType::IQ4_XS:
            return &iq4_xs;
        case GgmlType::I8:
            return &i8;
        case GgmlType::I16:
            return &i16;
        case GgmlType::I32:
            return &i32;
        case GgmlType::I64:
            return &i64;
        case GgmlType::F64:
            return &f64;
        case GgmlType::IQ1_M:
            return &iq1_m;
        case GgmlType::BF16:
            return &bf16;
        case GgmlType::TQ1_0:
            return &tq1_0;
        case GgmlType::TQ2_0:
            return &tq2_0;
        case GgmlType::MXFP4:
            return &mxfp4;
        case GgmlType::NVFP4:
            return &nvfp4;
        case GgmlType::Q1_0:
            return &q1_0;
    }
    return nullptr;
}

const char* gguf_type_name(GgufType type)
{
    switch (type)
    {
        case GgufType::UInt8:
            return "u8";
        case GgufType::Int8:
            return "i8";
        case GgufType::UInt16:
            return "u16";
        case GgufType::Int16:
            return "i16";
        case GgufType::UInt32:
            return "u32";
        case GgufType::Int32:
            return "i32";
        case GgufType::Float32:
            return "f32";
        case GgufType::Bool:
            return "bool";
        case GgufType::String:
            return "str";
        case GgufType::Array:
            return "arr";
        case GgufType::UInt64:
            return "u64";
        case GgufType::Int64:
            return "i64";
        case GgufType::Float64:
            return "f64";
    }
    return "?";
}

size_t gguf_scalar_size(GgufType type)
{
    switch (type)
    {
        case GgufType::UInt8:
        case GgufType::Int8:
        case GgufType::Bool:
            return 1;
        case GgufType::UInt16:
        case GgufType::Int16:
            return 2;
        case GgufType::UInt32:
        case GgufType::Int32:
        case GgufType::Float32:
            return 4;
        case GgufType::UInt64:
        case GgufType::Int64:
        case GgufType::Float64:
            return 8;
        case GgufType::String:
        case GgufType::Array:
            return 0;
    }
    return 0;
}

uint64_t checksum64(const uint8_t* data, size_t size)
{
    uint64_t sum = 0;
    uint64_t index = 0;
    size_t pos = 0;

    for (; pos + 8 <= size; pos += 8, ++index)
    {
        uint64_t word = 0;
        std::memcpy(&word, data + pos, 8);
        sum += word * (2 * index + 1);
    }

    if (pos < size)
    {
        uint64_t word = 0;
        std::memcpy(&word, data + pos, size - pos);
        sum += word * (2 * index + 1);
    }

    return sum + static_cast<uint64_t>(size);
}

bool GgufArray::strings_to(Array<std::string_view>& out) const
{
    if (elem_type != GgufType::String)
    {
        return false;
    }

    Reader reader(data, static_cast<size_t>(byte_size));
    out.clear();
    if (!out.reserve(static_cast<size_t>(count)))
    {
        return false;
    }
    for (uint64_t i = 0; i < count; ++i)
    {
        std::string_view item;
        if (!reader.read_string(item))
        {
            out.clear();
            return false;
        }
        out.push_back_unchecked(item);
    }
    return true;
}

GgufFile::GgufFile() = default;

GgufFile::~GgufFile() = default;

void GgufFile::close()
{
    m_metadata.release();
    m_tensors.release();
    m_metadata_index.release();
    m_tensor_index.release();
    m_mapping.close();

    m_data = nullptr;
    m_size = 0;
    m_version = 0;
    m_alignment = GGUF_DEFAULT_ALIGNMENT;
    m_data_offset = 0;
}

bool GgufFile::open(const char* path, Error& error)
{
    close();

    if (!m_mapping.open(path, error))
    {
        return false;
    }
    m_data = m_mapping.data();
    m_size = m_mapping.size();

    Error reason;
    if (!parse_image(reason))
    {
        error.set("'%s': %s", path, reason.c_str());
        close();
        return false;
    }
    return true;
}

bool GgufFile::parse(const uint8_t* data, size_t size, Error& error)
{
    close();

    if (data == nullptr && size != 0)
    {
        error.set("null buffer");
        return false;
    }
    m_data = data;
    m_size = size;

    if (!parse_image(error))
    {
        close();
        return false;
    }
    return true;
}

bool GgufFile::parse_image(Error& error)
{
    Reader reader(m_data, m_size);

    uint32_t magic = 0;
    if (!reader.read(magic) || magic != GGUF_MAGIC)
    {
        error.set("not a GGUF file (bad magic)");
        return false;
    }

    if (!reader.read(m_version))
    {
        error.set("header is truncated");
        return false;
    }
    if (m_version != 0 && (m_version & 0xffffu) == 0)
    {
        error.set("file looks big-endian, only little-endian GGUF is supported");
        return false;
    }
    if (m_version != 2 && m_version != 3)
    {
        error.set("unsupported GGUF version %" PRIu32 " (supported: 2 and 3)", m_version);
        return false;
    }

    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    if (!reader.read(tensor_count) || !reader.read(metadata_count))
    {
        error.set("header is truncated");
        return false;
    }
    if (metadata_count > reader.remaining() / MIN_KEY_VALUE_BYTES || metadata_count > MAX_ENTRIES)
    {
        error.set("metadata count %" PRIu64 " is larger than the file", metadata_count);
        return false;
    }
    if (tensor_count > reader.remaining() / MIN_TENSOR_INFO_BYTES || tensor_count > MAX_ENTRIES)
    {
        error.set("tensor count %" PRIu64 " is larger than the file", tensor_count);
        return false;
    }

    const size_t metadata_reserve = static_cast<size_t>(metadata_count < MAX_RESERVE ? metadata_count : MAX_RESERVE);
    if (!m_metadata.reserve(metadata_reserve) || !m_metadata_index.reserve(metadata_reserve))
    {
        error.set("out of memory while reading the metadata");
        return false;
    }

    for (uint64_t i = 0; i < metadata_count; ++i)
    {
        GgufKeyValue kv;
        if (!reader.read_string(kv.key))
        {
            error.set("metadata key %" PRIu64 " is truncated", i);
            return false;
        }
        if (kv.key.empty() || kv.key.size() > 65535)
        {
            error.set("metadata key %" PRIu64 " has invalid length %" PRIu64, i, static_cast<uint64_t>(kv.key.size()));
            return false;
        }

        uint32_t type = 0;
        if (!reader.read(type))
        {
            error.set("metadata key %s: type is truncated", quoted(kv.key).text);
            return false;
        }
        if (type > GGUF_MAX_TYPE_ID)
        {
            error.set("metadata key %s: unknown value type %" PRIu32, quoted(kv.key).text, type);
            return false;
        }

        Error reason;
        if (!parse_value(reader, static_cast<GgufType>(type), kv.value, reason))
        {
            error.set("metadata key %s: %s", quoted(kv.key).text, reason.c_str());
            return false;
        }

        const StringMap::Insert inserted = m_metadata_index.insert(kv.key, static_cast<uint32_t>(m_metadata.size()));
        if (inserted == StringMap::Insert::Exists)
        {
            error.set("duplicate metadata key %s", quoted(kv.key).text);
            return false;
        }
        if (inserted == StringMap::Insert::Failed || !m_metadata.push_back(kv))
        {
            error.set("out of memory while reading the metadata");
            return false;
        }
    }

    m_alignment = GGUF_DEFAULT_ALIGNMENT;
    if (find("general.alignment") != nullptr)
    {
        uint64_t alignment = 0;
        if (!get_uint("general.alignment", alignment))
        {
            error.set("general.alignment is not an unsigned integer");
            return false;
        }
        if (alignment == 0 || alignment % 8 != 0 || alignment > std::numeric_limits<uint32_t>::max())
        {
            error.set("general.alignment is %" PRIu64 ", expected a non-zero multiple of 8", alignment);
            return false;
        }
        m_alignment = static_cast<uint32_t>(alignment);
    }

    const size_t tensor_reserve = static_cast<size_t>(tensor_count < MAX_RESERVE ? tensor_count : MAX_RESERVE);
    if (!m_tensors.reserve(tensor_reserve) || !m_tensor_index.reserve(tensor_reserve))
    {
        error.set("out of memory while reading the tensor list");
        return false;
    }

    for (uint64_t i = 0; i < tensor_count; ++i)
    {
        GgufTensor tensor;
        if (!reader.read_string(tensor.name))
        {
            error.set("tensor %" PRIu64 ": name is truncated", i);
            return false;
        }

        if (!reader.read(tensor.n_dims))
        {
            error.set("tensor %s: info is truncated", quoted(tensor.name).text);
            return false;
        }
        if (tensor.n_dims > GGUF_MAX_DIMS)
        {
            error.set("tensor %s: %" PRIu32 " dimensions, at most %" PRIu32 " are supported", quoted(tensor.name).text,
                      tensor.n_dims, GGUF_MAX_DIMS);
            return false;
        }

        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < tensor.n_dims; ++d)
        {
            uint64_t dim = 0;
            if (!reader.read(dim))
            {
                error.set("tensor %s: info is truncated", quoted(tensor.name).text);
                return false;
            }
            if (dim != 0 && n_elements > std::numeric_limits<uint64_t>::max() / dim)
            {
                error.set("tensor %s: element count overflows", quoted(tensor.name).text);
                return false;
            }
            tensor.dims[d] = dim;
            n_elements *= dim;
        }
        tensor.n_elements = n_elements;

        uint32_t type = 0;
        if (!reader.read(type) || !reader.read(tensor.offset))
        {
            error.set("tensor %s: info is truncated", quoted(tensor.name).text);
            return false;
        }
        tensor.type = static_cast<GgmlType>(type);

        const GgmlTypeInfo* info = ggml_type_info(tensor.type);
        if (info == nullptr)
        {
            error.set("tensor %s: unknown tensor type %" PRIu32, quoted(tensor.name).text, type);
            return false;
        }
        if (n_elements % info->block_elements != 0)
        {
            error.set("tensor %s: %" PRIu64 " elements is not a multiple of the %s block size %" PRIu32,
                      quoted(tensor.name).text, n_elements, info->name, info->block_elements);
            return false;
        }

        const uint64_t blocks = n_elements / info->block_elements;
        if (blocks > std::numeric_limits<uint64_t>::max() / info->block_bytes)
        {
            error.set("tensor %s: byte size overflows", quoted(tensor.name).text);
            return false;
        }
        tensor.n_bytes = blocks * info->block_bytes;

        const StringMap::Insert inserted = m_tensor_index.insert(tensor.name, static_cast<uint32_t>(m_tensors.size()));
        if (inserted == StringMap::Insert::Exists)
        {
            error.set("duplicate tensor name %s", quoted(tensor.name).text);
            return false;
        }
        if (inserted == StringMap::Insert::Failed || !m_tensors.push_back(tensor))
        {
            error.set("out of memory while reading the tensor list");
            return false;
        }
    }

    const uint64_t pos = reader.pos();
    m_data_offset = pos + (m_alignment - (pos % m_alignment)) % m_alignment;

    for (GgufTensor& tensor : m_tensors)
    {
        if (tensor.offset % m_alignment != 0)
        {
            error.set("tensor %s: offset %" PRIu64 " is not aligned to %" PRIu32, quoted(tensor.name).text,
                      tensor.offset, m_alignment);
            return false;
        }

        // ordered so that no sum can overflow
        const uint64_t size = m_size;
        if (m_data_offset > size || tensor.offset > size - m_data_offset ||
            tensor.n_bytes > size - m_data_offset - tensor.offset)
        {
            error.set("tensor %s: data lies outside the file", quoted(tensor.name).text);
            return false;
        }

        tensor.data = m_data + m_data_offset + tensor.offset;
    }

    return true;
}

const GgufValue* GgufFile::find(std::string_view key) const
{
    uint32_t position = 0;
    if (!m_metadata_index.find(key, position))
    {
        return nullptr;
    }
    return &m_metadata[position].value;
}

const GgufTensor* GgufFile::find_tensor(std::string_view name) const
{
    uint32_t position = 0;
    if (!m_tensor_index.find(name, position))
    {
        return nullptr;
    }
    return &m_tensors[position];
}

bool GgufFile::get_string(std::string_view key, std::string_view& out) const
{
    const GgufValue* value = find(key);
    if (value == nullptr || value->type != GgufType::String)
    {
        return false;
    }
    out = value->str;
    return true;
}

bool GgufFile::get_bool(std::string_view key, bool& out) const
{
    const GgufValue* value = find(key);
    if (value == nullptr || value->type != GgufType::Bool)
    {
        return false;
    }
    out = value->u64 != 0;
    return true;
}

bool GgufFile::get_float(std::string_view key, double& out) const
{
    const GgufValue* value = find(key);
    if (value == nullptr || (value->type != GgufType::Float32 && value->type != GgufType::Float64))
    {
        return false;
    }
    out = value->f64;
    return true;
}

bool GgufFile::get_uint(std::string_view key, uint64_t& out) const
{
    const GgufValue* value = find(key);
    if (value == nullptr)
    {
        return false;
    }
    if (is_unsigned(value->type))
    {
        out = value->u64;
        return true;
    }
    if (is_signed(value->type) && value->i64 >= 0)
    {
        out = static_cast<uint64_t>(value->i64);
        return true;
    }
    return false;
}

bool GgufFile::get_int(std::string_view key, int64_t& out) const
{
    const GgufValue* value = find(key);
    if (value == nullptr)
    {
        return false;
    }
    if (is_signed(value->type))
    {
        out = value->i64;
        return true;
    }
    if (is_unsigned(value->type) && value->u64 <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    {
        out = static_cast<int64_t>(value->u64);
        return true;
    }
    return false;
}

const GgufArray* GgufFile::get_array(std::string_view key) const
{
    const GgufValue* value = find(key);
    if (value == nullptr || value->type != GgufType::Array)
    {
        return nullptr;
    }
    return &value->array;
}

}
