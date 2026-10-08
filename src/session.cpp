#include "faisca/session.h"

#include "faisca/mapped_file.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace faisca
{

// Snapshot layout, every field little endian, every section 8 byte aligned:
//   Header: magic "FSCA", version, n_layer, n_head_kv, head_dim, kv_f16, n_vocab,
//           n_tokens, name_bytes, n_sections (u32 each)
//   model name, padded
//   offset table: n_sections entries of {kind u32, layer u32, offset u64, size u64},
//           kind 1 tokens, 2 keys of a layer, 3 values of a layer, offsets from the start
//   sections

namespace
{

constexpr uint32_t MAGIC = 0x41435346u; // "FSCA"
constexpr uint32_t HEADER_WORDS = 10;
constexpr uint32_t KIND_TOKENS = 1;
constexpr uint32_t KIND_KEYS = 2;
constexpr uint32_t KIND_VALUES = 3;

struct Entry
{
    uint32_t kind;
    uint32_t layer;
    uint64_t offset;
    uint64_t size;
};

inline size_t align8(size_t value)
{
    return (value + 7) & ~static_cast<size_t>(7);
}

inline void put_u32(uint8_t*& at, uint32_t value)
{
    std::memcpy(at, &value, 4);
    at += 4;
}

inline uint32_t get_u32(const uint8_t* at)
{
    uint32_t value = 0;
    std::memcpy(&value, at, 4);
    return value;
}

}

bool Session::init(Runner& runner, const char* model_name, Error& error)
{
    if (runner.model() == nullptr)
    {
        error.set("the runner is not initialized");
        return false;
    }
    m_runner = &runner;
    m_tokens.clear();
    std::memset(m_model_name, 0, sizeof(m_model_name));
    if (model_name != nullptr)
    {
        std::strncpy(m_model_name, model_name, sizeof(m_model_name) - 1);
    }
    runner.reset();
    return true;
}

bool Session::truncate(size_t count, Error& error)
{
    if (count > m_tokens.size())
    {
        error.set("cannot keep %zu tokens of %zu", count, m_tokens.size());
        return false;
    }
    m_tokens.resize_uninitialized(count);
    m_runner->truncate(static_cast<uint32_t>(count));
    return true;
}

bool Session::feed(const int32_t* tokens, size_t count, Array<float>& logits, Error& error, size_t* reused)
{
    if (m_runner == nullptr)
    {
        error.set("the session is not initialized");
        return false;
    }
    if (count == 0)
    {
        error.set("no tokens to feed");
        return false;
    }
    size_t common = 0;
    const size_t limit = count - 1 < m_tokens.size() ? count - 1 : m_tokens.size();
    while (common < limit && m_tokens[common] == tokens[common])
    {
        ++common;
    }
    if (reused != nullptr)
    {
        *reused = common;
    }
    if (!truncate(common, error) || !m_runner->forward(tokens + common, count - common, logits, error))
    {
        return false;
    }
    if (!m_tokens.append(tokens + common, count - common))
    {
        error.set("out of memory");
        return false;
    }
    return true;
}

bool Session::push(int32_t token, Array<float>& logits, Error& error)
{
    if (m_runner == nullptr)
    {
        error.set("the session is not initialized");
        return false;
    }
    if (!m_runner->forward(&token, 1, logits, error))
    {
        return false;
    }
    if (!m_tokens.push_back(token))
    {
        error.set("out of memory");
        return false;
    }
    return true;
}

size_t Session::snapshot_bytes() const
{
    const LlamaHparams& h = m_runner->model()->hparams();
    const uint32_t pos = m_runner->position();
    const size_t name_bytes = std::strlen(m_model_name);
    const size_t sections = 1 + 2 * static_cast<size_t>(h.n_layer);
    size_t total = align8(HEADER_WORDS * 4 + name_bytes) + sections * sizeof(Entry);
    total += align8(m_tokens.size() * sizeof(int32_t));
    total += static_cast<size_t>(h.n_layer) * (align8(m_runner->state_k_bytes(pos)) + align8(m_runner->state_v_bytes(pos)));
    return total;
}

bool Session::snapshot(Array<uint8_t>& out, Error& error) const
{
    if (m_runner == nullptr)
    {
        error.set("the session is not initialized");
        return false;
    }
    if (m_tokens.size() != m_runner->position())
    {
        error.set("the session holds %zu tokens but the cache %" PRIu32 " positions", m_tokens.size(), m_runner->position());
        return false;
    }
    const LlamaHparams& h = m_runner->model()->hparams();
    const uint32_t pos = m_runner->position();
    if (!out.resize_uninitialized(snapshot_bytes()))
    {
        error.set("out of memory");
        return false;
    }
    std::memset(out.data(), 0, out.size());
    uint8_t* at = out.data();
    const size_t name_bytes = std::strlen(m_model_name);
    const uint32_t sections = 1 + 2 * h.n_layer;
    put_u32(at, MAGIC);
    put_u32(at, FILE_VERSION);
    put_u32(at, h.n_layer);
    put_u32(at, h.n_head_kv);
    put_u32(at, h.head_dim);
    put_u32(at, m_runner->kv_f16() ? 1u : 0u);
    put_u32(at, h.n_vocab);
    put_u32(at, pos);
    put_u32(at, static_cast<uint32_t>(name_bytes));
    put_u32(at, sections);
    std::memcpy(at, m_model_name, name_bytes);
    at = out.data() + align8(HEADER_WORDS * 4 + name_bytes);
    Entry* table = reinterpret_cast<Entry*>(at);
    at += sections * sizeof(Entry);

    Entry* entry = table;
    entry->kind = KIND_TOKENS;
    entry->layer = 0;
    entry->offset = static_cast<uint64_t>(at - out.data());
    entry->size = m_tokens.size() * sizeof(int32_t);
    std::memcpy(at, m_tokens.data(), m_tokens.size() * sizeof(int32_t));
    at += align8(m_tokens.size() * sizeof(int32_t));
    ++entry;

    // the runner writes the layers back to back; the table points into that run
    Array<uint8_t> state;
    if (!state.resize_uninitialized(m_runner->state_bytes(pos)) || !m_runner->export_state(state.data(), state.size()))
    {
        error.set("cannot export the cache");
        return false;
    }
    const uint8_t* source = state.data();
    const size_t k_bytes = m_runner->state_k_bytes(pos);
    const size_t v_bytes = m_runner->state_v_bytes(pos);
    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        entry->kind = KIND_KEYS;
        entry->layer = l;
        entry->offset = static_cast<uint64_t>(at - out.data());
        entry->size = k_bytes;
        std::memcpy(at, source, k_bytes);
        source += k_bytes;
        at += align8(k_bytes);
        ++entry;
        entry->kind = KIND_VALUES;
        entry->layer = l;
        entry->offset = static_cast<uint64_t>(at - out.data());
        entry->size = v_bytes;
        std::memcpy(at, source, v_bytes);
        source += v_bytes;
        at += align8(v_bytes);
        ++entry;
    }
    return true;
}

bool Session::restore(const uint8_t* data, size_t bytes, Error& error)
{
    if (m_runner == nullptr)
    {
        error.set("the session is not initialized");
        return false;
    }
    if (bytes < HEADER_WORDS * 4 || get_u32(data) != MAGIC)
    {
        error.set("not a faisca session");
        return false;
    }
    if (get_u32(data + 4) != FILE_VERSION)
    {
        error.set("session version %" PRIu32 ", this build reads %" PRIu32, get_u32(data + 4), FILE_VERSION);
        return false;
    }
    const LlamaHparams& h = m_runner->model()->hparams();
    const uint32_t n_layer = get_u32(data + 8);
    const uint32_t n_head_kv = get_u32(data + 12);
    const uint32_t head_dim = get_u32(data + 16);
    const uint32_t kv_f16 = get_u32(data + 20);
    const uint32_t n_vocab = get_u32(data + 24);
    const uint32_t pos = get_u32(data + 28);
    const uint32_t name_bytes = get_u32(data + 32);
    const uint32_t sections = get_u32(data + 36);
    if (n_layer != h.n_layer || n_head_kv != h.n_head_kv || head_dim != h.head_dim || n_vocab != h.n_vocab)
    {
        error.set("the session was made with another model: %" PRIu32 " layers, %" PRIu32 " kv heads of %" PRIu32
                  ", vocabulary %" PRIu32, n_layer, n_head_kv, head_dim, n_vocab);
        return false;
    }
    if ((kv_f16 != 0) != m_runner->kv_f16())
    {
        error.set("the session holds the cache in %s, the runner uses %s", kv_f16 != 0 ? "f16" : "float32",
                  m_runner->kv_f16() ? "f16" : "float32");
        return false;
    }
    if (name_bytes > bytes - HEADER_WORDS * 4 || sections != 1 + 2 * n_layer)
    {
        error.set("the session file is damaged");
        return false;
    }
    if (m_model_name[0] != '\0' && (name_bytes != std::strlen(m_model_name) ||
                                     std::memcmp(data + HEADER_WORDS * 4, m_model_name, name_bytes) != 0))
    {
        error.set("the session was made with the model '%.*s', this is '%s'", static_cast<int>(name_bytes),
                  reinterpret_cast<const char*>(data + HEADER_WORDS * 4), m_model_name);
        return false;
    }
    const size_t table_at = align8(HEADER_WORDS * 4 + name_bytes);
    if (table_at + sections * sizeof(Entry) > bytes)
    {
        error.set("the session file is damaged");
        return false;
    }
    Array<Entry> table;
    if (!table.resize_uninitialized(sections))
    {
        error.set("out of memory");
        return false;
    }
    std::memcpy(table.data(), data + table_at, sections * sizeof(Entry));
    for (size_t i = 0; i < sections; ++i)
    {
        if (table[i].offset > bytes || table[i].size > bytes - table[i].offset)
        {
            error.set("the session file is damaged");
            return false;
        }
    }
    if (table[0].kind != KIND_TOKENS || table[0].size != static_cast<uint64_t>(pos) * sizeof(int32_t))
    {
        error.set("the session file is damaged");
        return false;
    }
    const size_t k_bytes = m_runner->state_k_bytes(pos);
    const size_t v_bytes = m_runner->state_v_bytes(pos);
    Array<uint8_t> state;
    if (!state.resize_uninitialized(m_runner->state_bytes(pos)))
    {
        error.set("out of memory");
        return false;
    }
    uint8_t* at = state.data();
    for (uint32_t l = 0; l < n_layer; ++l)
    {
        const Entry& keys = table[1 + 2 * l];
        const Entry& values = table[2 + 2 * l];
        if (keys.kind != KIND_KEYS || keys.layer != l || keys.size != k_bytes || values.kind != KIND_VALUES ||
            values.layer != l || values.size != v_bytes)
        {
            error.set("the session file is damaged");
            return false;
        }
        std::memcpy(at, data + keys.offset, k_bytes);
        at += k_bytes;
        std::memcpy(at, data + values.offset, v_bytes);
        at += v_bytes;
    }
    if (!m_runner->import_state(state.data(), state.size(), pos, error))
    {
        return false;
    }
    m_tokens.clear();
    if (!m_tokens.append(reinterpret_cast<const int32_t*>(data + table[0].offset), pos))
    {
        error.set("out of memory");
        return false;
    }
    return true;
}

bool Session::save(const char* path, Error& error) const
{
    Array<uint8_t> bytes;
    if (!snapshot(bytes, error))
    {
        return false;
    }
    FILE* file = std::fopen(path, "wb");
    if (file == nullptr)
    {
        error.set("cannot write '%s'", path);
        return false;
    }
    const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    const bool closed = std::fclose(file) == 0;
    if (!ok || !closed)
    {
        error.set("cannot write '%s'", path);
        return false;
    }
    return true;
}

bool Session::load(const char* path, Error& error)
{
    MappedFile file;
    if (!file.open(path, error))
    {
        return false;
    }
    return restore(file.data(), file.size(), error);
}

}
