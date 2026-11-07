#include "faisca/safetensors.h"

#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

bool ends_with(std::string_view text, std::string_view suffix)
{
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

}

bool SafetensorsModel::open_shard(const char* path, Error& error)
{
    if (m_shards == MAX_SHARDS)
    {
        error.set("safetensors: more than %d shards", MAX_SHARDS);
        return false;
    }
    MappedFile& file = m_files[m_shards];
    JsonDocument& header = m_headers[m_shards];
    if (!file.open(path, error))
    {
        return false;
    }
    ++m_shards;
    if (file.size() < 8)
    {
        error.set("%s: too short for a safetensors file", path);
        return false;
    }
    uint64_t header_size = 0;
    std::memcpy(&header_size, file.data(), 8);
    if (header_size > file.size() - 8)
    {
        error.set("%s: header of %llu bytes past the end", path, static_cast<unsigned long long>(header_size));
        return false;
    }
    if (!header.parse(reinterpret_cast<const char*>(file.data() + 8), static_cast<size_t>(header_size), error))
    {
        return false;
    }
    const uint32_t root = header.root();
    if (header.value(root).kind != JsonKind::Object)
    {
        error.set("%s: the header is not a JSON object", path);
        return false;
    }
    const uint8_t* data = file.data() + 8 + header_size;
    const uint64_t data_size = file.size() - 8 - header_size;
    for (uint32_t k = header.value(root).first; k != JSON_NONE;)
    {
        const uint32_t v = header.value(k).next;
        const std::string_view name = header.value(k).text;
        k = header.value(v).next;
        if (name == "__metadata__")
        {
            continue;
        }
        const uint32_t dtype = header.find(v, "dtype");
        const uint32_t shape = header.find(v, "shape");
        const uint32_t offsets = header.find(v, "data_offsets");
        if (dtype == JSON_NONE || header.value(dtype).kind != JsonKind::String || shape == JSON_NONE ||
            header.value(shape).kind != JsonKind::Array || offsets == JSON_NONE ||
            header.value(offsets).kind != JsonKind::Array || header.value(offsets).count != 2)
        {
            error.set("%s: tensor '%.*s' has no dtype, shape or data_offsets", path, static_cast<int>(name.size()),
                      name.data());
            return false;
        }
        SafetensorsTensor t = {};
        t.name = name;
        t.dtype_name = header.value(dtype).text;
        uint64_t element = 0;
        if (t.dtype_name == "F32")
        {
            t.dtype = SafetensorsDtype::F32;
            element = 4;
        }
        else if (t.dtype_name == "F16")
        {
            t.dtype = SafetensorsDtype::F16;
            element = 2;
        }
        else if (t.dtype_name == "BF16")
        {
            t.dtype = SafetensorsDtype::BF16;
            element = 2;
        }
        else
        {
            t.dtype = SafetensorsDtype::Other;
            element = t.dtype_name == "F64" || t.dtype_name == "I64" || t.dtype_name == "U64" ? 8
                      : t.dtype_name == "I32" || t.dtype_name == "U32"                           ? 4
                      : t.dtype_name == "I16" || t.dtype_name == "U16"                           ? 2
                                                                                                 : 1;
        }
        if (header.value(shape).count > SafetensorsTensor::MAX_DIMS)
        {
            error.set("%s: '%.*s' has too many dimensions", path, static_cast<int>(name.size()), name.data());
            return false;
        }
        t.n_elements = 1;
        for (uint32_t d = header.value(shape).first; d != JSON_NONE; d = header.value(d).next)
        {
            const JsonValue& dim = header.value(d);
            if (dim.kind != JsonKind::Number || !dim.integral || dim.integer < 0)
            {
                error.set("%s: '%.*s' has a bad shape", path, static_cast<int>(name.size()), name.data());
                return false;
            }
            t.shape[t.n_dims++] = static_cast<uint64_t>(dim.integer);
            if (dim.integer != 0 && t.n_elements > (uint64_t(1) << 48) / static_cast<uint64_t>(dim.integer))
            {
                error.set("%s: '%.*s' is too large", path, static_cast<int>(name.size()), name.data());
                return false;
            }
            t.n_elements *= static_cast<uint64_t>(dim.integer);
        }
        const JsonValue& begin = header.value(header.value(offsets).first);
        const JsonValue& end = header.value(begin.next);
        if (!begin.integral || !end.integral || begin.integer < 0 || end.integer < begin.integer ||
            static_cast<uint64_t>(end.integer) > data_size ||
            static_cast<uint64_t>(end.integer - begin.integer) != t.n_elements * element)
        {
            error.set("%s: '%.*s' data offsets do not match its shape", path, static_cast<int>(name.size()),
                      name.data());
            return false;
        }
        t.data = data + begin.integer;
        t.n_bytes = static_cast<uint64_t>(end.integer - begin.integer);
        if (!m_tensors.push_back(t))
        {
            error.set("out of memory");
            return false;
        }
        if (m_index.insert(name, static_cast<uint32_t>(m_tensors.size() - 1)) != StringMap::Insert::Added)
        {
            error.set("%s: tensor '%.*s' twice", path, static_cast<int>(name.size()), name.data());
            return false;
        }
    }
    return true;
}

bool SafetensorsModel::open(const char* path, Error& error)
{
    const std::string_view p(path);
    if (!ends_with(p, ".json"))
    {
        return open_shard(path, error);
    }
    // an index: open every shard its weight_map names, once, from the index's directory
    MappedFile index_file;
    JsonDocument index;
    if (!index_file.open(path, error) ||
        !index.parse(reinterpret_cast<const char*>(index_file.data()), index_file.size(), error))
    {
        return false;
    }
    const uint32_t map = index.find(index.root(), "weight_map");
    if (map == JSON_NONE || index.value(map).kind != JsonKind::Object)
    {
        error.set("%s: no weight_map", path);
        return false;
    }
    const size_t slash = p.rfind('/');
    const std::string_view directory = slash == std::string_view::npos ? std::string_view() : p.substr(0, slash + 1);
    std::string_view opened[MAX_SHARDS];
    int count = 0;
    for (uint32_t k = index.value(map).first; k != JSON_NONE;)
    {
        const uint32_t v = index.value(k).next;
        k = index.value(v).next;
        const std::string_view shard = index.value(v).text;
        bool seen = false;
        for (int i = 0; i < count; ++i)
        {
            seen = seen || opened[i] == shard;
        }
        if (seen)
        {
            continue;
        }
        if (count == MAX_SHARDS || shard.find('/') != std::string_view::npos)
        {
            error.set("%s: too many shards or a shard outside the directory", path);
            return false;
        }
        opened[count++] = shard;
        char shard_path[4096];
        std::snprintf(shard_path, sizeof(shard_path), "%.*s%.*s", static_cast<int>(directory.size()),
                      directory.data(), static_cast<int>(shard.size()), shard.data());
        if (!open_shard(shard_path, error))
        {
            return false;
        }
    }
    return true;
}

const SafetensorsTensor* SafetensorsModel::find(std::string_view name) const
{
    uint32_t index = 0;
    if (!m_index.find(name, index))
    {
        return nullptr;
    }
    return &m_tensors[index];
}

}
