#include "faisca/torch_ckpt.h"

#include <cstdio>
#include <cstring>

namespace faisca
{

const char* ckpt_dtype_name(CkptDtype dtype)
{
    switch (dtype)
    {
    case CkptDtype::F32:
        return "F32";
    case CkptDtype::F16:
        return "F16";
    case CkptDtype::BF16:
        return "BF16";
    }
    return "?";
}

namespace
{

uint16_t read_u16(const uint8_t* p)
{
    uint16_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

uint32_t read_u32(const uint8_t* p)
{
    uint32_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

uint64_t read_u64(const uint8_t* p)
{
    uint64_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

struct ZipEntry
{
    std::string_view name;
    uint64_t offset;
    uint64_t size;
    uint16_t method;
};

class Zip
{
public:
    bool parse(const uint8_t* data, size_t size, Error& error);

    const ZipEntry* find(std::string_view name) const
    {
        uint32_t index;
        if (!m_index.find(name, index))
        {
            return nullptr;
        }
        return &m_entries[index];
    }

    size_t count() const
    {
        return m_entries.size();
    }

    const ZipEntry& entry(size_t index) const
    {
        return m_entries[index];
    }

private:
    Array<ZipEntry> m_entries;
    StringMap m_index;
};

bool Zip::parse(const uint8_t* data, size_t size, Error& error)
{
    const size_t EOCD_SIZE = 22;
    if (size < EOCD_SIZE)
    {
        error.set("not a zip file: %zu bytes", size);
        return false;
    }
    size_t eocd = size;
    const size_t lowest = size > EOCD_SIZE + 65535 ? size - EOCD_SIZE - 65535 : 0;
    for (size_t pos = size - EOCD_SIZE + 1; pos-- > lowest;)
    {
        if (read_u32(data + pos) == 0x06054b50u)
        {
            eocd = pos;
            break;
        }
    }
    if (eocd == size)
    {
        error.set("zip end of central directory not found");
        return false;
    }

    uint64_t n_entries = read_u16(data + eocd + 10);
    uint64_t cd_size = read_u32(data + eocd + 12);
    uint64_t cd_offset = read_u32(data + eocd + 16);
    if (n_entries == 0xffffu || cd_size == 0xffffffffu || cd_offset == 0xffffffffu)
    {
        if (eocd < 20 || read_u32(data + eocd - 20) != 0x07064b50u)
        {
            error.set("zip64 locator missing");
            return false;
        }
        const uint64_t z64 = read_u64(data + eocd - 20 + 8);
        if (z64 > size || size - z64 < 56 || read_u32(data + z64) != 0x06064b50u)
        {
            error.set("zip64 end of central directory invalid");
            return false;
        }
        n_entries = read_u64(data + z64 + 32);
        cd_size = read_u64(data + z64 + 40);
        cd_offset = read_u64(data + z64 + 48);
    }
    if (cd_offset > size || cd_size > size - cd_offset)
    {
        error.set("zip central directory out of range");
        return false;
    }
    if (n_entries > cd_size / 46 || n_entries > 0xffffffffu)
    {
        error.set("zip entry count %llu invalid", static_cast<unsigned long long>(n_entries));
        return false;
    }
    if (!m_entries.reserve(static_cast<size_t>(n_entries)) || !m_index.reserve(static_cast<size_t>(n_entries)))
    {
        error.set("out of memory for %llu zip entries", static_cast<unsigned long long>(n_entries));
        return false;
    }

    const uint8_t* p = data + cd_offset;
    const uint8_t* end = p + cd_size;
    for (uint64_t i = 0; i < n_entries; ++i)
    {
        if (end - p < 46 || read_u32(p) != 0x02014b50u)
        {
            error.set("zip central header %llu invalid", static_cast<unsigned long long>(i));
            return false;
        }
        const uint16_t method = read_u16(p + 10);
        uint64_t comp_size = read_u32(p + 20);
        uint64_t uncomp_size = read_u32(p + 24);
        const uint16_t name_size = read_u16(p + 28);
        const uint16_t extra_size = read_u16(p + 30);
        const uint16_t comment_size = read_u16(p + 32);
        uint64_t local = read_u32(p + 42);
        if (static_cast<size_t>(end - p) < 46u + name_size + extra_size + comment_size)
        {
            error.set("zip central header %llu truncated", static_cast<unsigned long long>(i));
            return false;
        }
        const std::string_view name(reinterpret_cast<const char*>(p + 46), name_size);

        const uint8_t* extra = p + 46 + name_size;
        const uint8_t* extra_end = extra + extra_size;
        while (extra_end - extra >= 4)
        {
            const uint16_t id = read_u16(extra);
            const uint16_t length = read_u16(extra + 2);
            const uint8_t* field = extra + 4;
            if (extra_end - field < length)
            {
                break;
            }
            if (id == 0x0001)
            {
                const uint8_t* q = field;
                const uint8_t* q_end = field + length;
                if (uncomp_size == 0xffffffffu && q_end - q >= 8)
                {
                    uncomp_size = read_u64(q);
                    q += 8;
                }
                if (comp_size == 0xffffffffu && q_end - q >= 8)
                {
                    comp_size = read_u64(q);
                    q += 8;
                }
                if (local == 0xffffffffu && q_end - q >= 8)
                {
                    local = read_u64(q);
                }
            }
            extra = field + length;
        }

        if (local > size || size - local < 30 || read_u32(data + local) != 0x04034b50u)
        {
            error.set("zip local header of '%.*s' invalid", static_cast<int>(name.size()), name.data());
            return false;
        }
        const uint64_t data_offset = local + 30 + read_u16(data + local + 26) + read_u16(data + local + 28);
        if (data_offset > size || comp_size > size - data_offset)
        {
            error.set("zip data of '%.*s' out of range", static_cast<int>(name.size()), name.data());
            return false;
        }
        if (method == 0 && comp_size != uncomp_size)
        {
            error.set("zip entry '%.*s' stored with mismatched sizes", static_cast<int>(name.size()), name.data());
            return false;
        }

        const uint32_t index = static_cast<uint32_t>(m_entries.size());
        if (m_index.insert(name, index) != StringMap::Insert::Added)
        {
            error.set("zip entry '%.*s' duplicated or out of memory", static_cast<int>(name.size()), name.data());
            return false;
        }
        m_entries.push_back_unchecked(ZipEntry{name, data_offset, comp_size, method});
        p += 46 + name_size + extra_size + comment_size;
    }
    return true;
}

enum class Kind : uint8_t
{
    None,
    Bool,
    Int,
    String,
    Global,
    Tuple,
    List,
    Dict,
    Storage,
    Tensor,
};

constexpr uint32_t NIL = 0xffffffffu;

struct Value
{
    Kind kind;
    int64_t number;
    std::string_view text;
    std::string_view text2;
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    uint32_t refs[3];
};

struct Item
{
    uint32_t value;
    uint32_t next;
};

class Unpickler
{
public:
    bool run(const uint8_t* data, size_t size, uint32_t& result, Error& error);

    const Value& value(uint32_t id) const
    {
        return m_values[id];
    }

    uint32_t child(uint32_t id, uint32_t index) const
    {
        uint32_t item = m_values[id].head;
        for (uint32_t i = 0; i < index && item != NIL; ++i)
        {
            item = m_items[item].next;
        }
        return item == NIL ? NIL : m_items[item].value;
    }

    uint32_t first_item(uint32_t id) const
    {
        return m_values[id].head;
    }

    const Item& item(uint32_t index) const
    {
        return m_items[index];
    }

private:
    bool make(Kind kind, uint32_t& id);
    bool make_text(Kind kind, std::string_view text, uint32_t& id);
    bool make_int(int64_t number, uint32_t& id);
    bool add_child(uint32_t parent, uint32_t child);
    bool push(uint32_t id);
    bool pop(uint32_t& id);
    bool pop_mark(size_t& start);
    bool tuple_from(size_t start, uint32_t& id);
    bool reduce(uint32_t callable, uint32_t args, uint32_t& id);
    bool persistent(uint32_t pid, uint32_t& id);
    bool fail(const char* what);

    Array<Value> m_values;
    Array<Item> m_items;
    Array<uint32_t> m_stack;
    Array<size_t> m_marks;
    Array<uint32_t> m_memo;
    Error* m_error = nullptr;
    size_t m_pos = 0;
};

bool Unpickler::fail(const char* what)
{
    m_error->set("pickle: %s at byte %zu", what, m_pos);
    return false;
}

bool Unpickler::make(Kind kind, uint32_t& id)
{
    Value value = {};
    value.kind = kind;
    value.head = NIL;
    value.tail = NIL;
    value.refs[0] = value.refs[1] = value.refs[2] = NIL;
    if (m_values.size() >= NIL - 1 || !m_values.push_back(value))
    {
        return fail("out of memory");
    }
    id = static_cast<uint32_t>(m_values.size() - 1);
    return true;
}

bool Unpickler::make_text(Kind kind, std::string_view text, uint32_t& id)
{
    if (!make(kind, id))
    {
        return false;
    }
    m_values[id].text = text;
    return true;
}

bool Unpickler::make_int(int64_t number, uint32_t& id)
{
    if (!make(Kind::Int, id))
    {
        return false;
    }
    m_values[id].number = number;
    return true;
}

bool Unpickler::add_child(uint32_t parent, uint32_t child)
{
    if (m_items.size() >= NIL - 1 || !m_items.push_back(Item{child, NIL}))
    {
        return fail("out of memory");
    }
    const uint32_t index = static_cast<uint32_t>(m_items.size() - 1);
    Value& p = m_values[parent];
    if (p.tail == NIL)
    {
        p.head = index;
    }
    else
    {
        m_items[p.tail].next = index;
    }
    p.tail = index;
    p.count += 1;
    return true;
}

bool Unpickler::push(uint32_t id)
{
    if (!m_stack.push_back(id))
    {
        return fail("out of memory");
    }
    return true;
}

bool Unpickler::pop(uint32_t& id)
{
    const size_t floor = m_marks.empty() ? 0 : m_marks[m_marks.size() - 1];
    if (m_stack.size() <= floor)
    {
        return fail("stack underflow");
    }
    id = m_stack[m_stack.size() - 1];
    m_stack.resize_uninitialized(m_stack.size() - 1);
    return true;
}

bool Unpickler::pop_mark(size_t& start)
{
    if (m_marks.empty())
    {
        return fail("no mark");
    }
    start = m_marks[m_marks.size() - 1];
    m_marks.resize_uninitialized(m_marks.size() - 1);
    return true;
}

bool Unpickler::tuple_from(size_t start, uint32_t& id)
{
    if (!make(Kind::Tuple, id))
    {
        return false;
    }
    for (size_t i = start; i < m_stack.size(); ++i)
    {
        if (!add_child(id, m_stack[i]))
        {
            return false;
        }
    }
    m_stack.resize_uninitialized(start);
    return true;
}

bool Unpickler::reduce(uint32_t callable, uint32_t args, uint32_t& id)
{
    const Value& c = m_values[callable];
    if (c.kind != Kind::Global || m_values[args].kind != Kind::Tuple)
    {
        return fail("REDUCE needs a global and a tuple");
    }
    if (c.text == "collections" && c.text2 == "OrderedDict")
    {
        return make(Kind::Dict, id);
    }
    if (c.text == "torch._utils" && c.text2 == "_rebuild_parameter")
    {
        id = child(args, 0);
        return id != NIL || fail("_rebuild_parameter without data");
    }
    if (c.text == "torch._utils" && c.text2 == "_rebuild_tensor_v2")
    {
        if (m_values[args].count < 4)
        {
            return fail("_rebuild_tensor_v2 with too few arguments");
        }
        const uint32_t storage = child(args, 0);
        const uint32_t offset = child(args, 1);
        const uint32_t shape = child(args, 2);
        const uint32_t stride = child(args, 3);
        if (m_values[storage].kind != Kind::Storage || m_values[offset].kind != Kind::Int ||
            m_values[shape].kind != Kind::Tuple || m_values[stride].kind != Kind::Tuple)
        {
            return fail("_rebuild_tensor_v2 argument types");
        }
        if (!make(Kind::Tensor, id))
        {
            return false;
        }
        Value& t = m_values[id];
        t.number = m_values[offset].number;
        t.refs[0] = storage;
        t.refs[1] = shape;
        t.refs[2] = stride;
        return true;
    }
    m_error->set("pickle: unsupported callable %.*s.%.*s at byte %zu", static_cast<int>(c.text.size()),
                 c.text.data(), static_cast<int>(c.text2.size()), c.text2.data(), m_pos);
    return false;
}

bool Unpickler::persistent(uint32_t pid, uint32_t& id)
{
    const Value& p = m_values[pid];
    if (p.kind != Kind::Tuple || p.count < 5)
    {
        return fail("persistent id is not a storage tuple");
    }
    const uint32_t tag = child(pid, 0);
    const uint32_t type = child(pid, 1);
    const uint32_t key = child(pid, 2);
    const uint32_t numel = child(pid, 4);
    if (m_values[tag].kind != Kind::String || m_values[tag].text != "storage" ||
        m_values[type].kind != Kind::Global || m_values[key].kind != Kind::String ||
        m_values[numel].kind != Kind::Int)
    {
        return fail("persistent id fields");
    }
    if (!make(Kind::Storage, id))
    {
        return false;
    }
    Value& s = m_values[id];
    s.text = m_values[key].text;
    s.text2 = m_values[type].text2;
    s.number = m_values[numel].number;
    return true;
}

bool Unpickler::run(const uint8_t* data, size_t size, uint32_t& result, Error& error)
{
    m_error = &error;
    m_pos = 0;
    auto need = [&](size_t n) { return size - m_pos >= n; };

    while (m_pos < size)
    {
        const uint8_t op = data[m_pos++];
        uint32_t id = NIL;
        uint32_t a = NIL;
        uint32_t b = NIL;
        size_t start = 0;
        switch (op)
        {
        case 0x80: // PROTO
            if (!need(1))
            {
                return fail("truncated");
            }
            if (data[m_pos] < 2 || data[m_pos] > 5)
            {
                return fail("unsupported protocol");
            }
            m_pos += 1;
            break;
        case 0x95: // FRAME
            if (!need(8))
            {
                return fail("truncated");
            }
            m_pos += 8;
            break;
        case '.': // STOP
            if (m_stack.size() != 1 || !m_marks.empty())
            {
                return fail("STOP with an unbalanced stack");
            }
            result = m_stack[0];
            return true;
        case '(': // MARK
            if (!m_marks.push_back(m_stack.size()))
            {
                return fail("out of memory");
            }
            break;
        case '}': // EMPTY_DICT
            if (!make(Kind::Dict, id) || !push(id))
            {
                return false;
            }
            break;
        case ']': // EMPTY_LIST
            if (!make(Kind::List, id) || !push(id))
            {
                return false;
            }
            break;
        case ')': // EMPTY_TUPLE
            if (!make(Kind::Tuple, id) || !push(id))
            {
                return false;
            }
            break;
        case 'N': // NONE
            if (!make(Kind::None, id) || !push(id))
            {
                return false;
            }
            break;
        case 0x88: // NEWTRUE
        case 0x89: // NEWFALSE
            if (!make_int(op == 0x88 ? 1 : 0, id) || !push(id))
            {
                return false;
            }
            m_values[id].kind = Kind::Bool;
            break;
        case 'K': // BININT1
            if (!need(1))
            {
                return fail("truncated");
            }
            if (!make_int(data[m_pos], id) || !push(id))
            {
                return false;
            }
            m_pos += 1;
            break;
        case 'M': // BININT2
            if (!need(2))
            {
                return fail("truncated");
            }
            if (!make_int(read_u16(data + m_pos), id) || !push(id))
            {
                return false;
            }
            m_pos += 2;
            break;
        case 'J': // BININT
            if (!need(4))
            {
                return fail("truncated");
            }
            if (!make_int(static_cast<int32_t>(read_u32(data + m_pos)), id) || !push(id))
            {
                return false;
            }
            m_pos += 4;
            break;
        case 0x8a: // LONG1
        {
            if (!need(1))
            {
                return fail("truncated");
            }
            const size_t n = data[m_pos];
            if (n > 8 || !need(1 + n))
            {
                return fail("LONG1 wider than 64 bits");
            }
            uint64_t bits = 0;
            for (size_t i = 0; i < n; ++i)
            {
                bits |= static_cast<uint64_t>(data[m_pos + 1 + i]) << (8 * i);
            }
            if (n != 0 && n < 8 && (data[m_pos + n] & 0x80u) != 0)
            {
                bits |= ~uint64_t(0) << (8 * n);
            }
            if (!make_int(static_cast<int64_t>(bits), id) || !push(id))
            {
                return false;
            }
            m_pos += 1 + n;
            break;
        }
        case 'X': // BINUNICODE
        case 0x8c: // SHORT_BINUNICODE
        {
            const size_t width = op == 'X' ? 4 : 1;
            if (!need(width))
            {
                return fail("truncated");
            }
            const size_t n = op == 'X' ? read_u32(data + m_pos) : data[m_pos];
            m_pos += width;
            if (!need(n))
            {
                return fail("truncated string");
            }
            if (!make_text(Kind::String, std::string_view(reinterpret_cast<const char*>(data + m_pos), n), id) ||
                !push(id))
            {
                return false;
            }
            m_pos += n;
            break;
        }
        case 'c': // GLOBAL
        {
            const char* text = reinterpret_cast<const char*>(data + m_pos);
            const void* nl1 = std::memchr(text, '\n', size - m_pos);
            if (nl1 == nullptr)
            {
                return fail("truncated GLOBAL");
            }
            const size_t module_size = static_cast<size_t>(static_cast<const char*>(nl1) - text);
            const size_t rest = size - m_pos - module_size - 1;
            const void* nl2 = std::memchr(text + module_size + 1, '\n', rest);
            if (nl2 == nullptr)
            {
                return fail("truncated GLOBAL");
            }
            const size_t name_size = static_cast<size_t>(static_cast<const char*>(nl2) - (text + module_size + 1));
            if (!make_text(Kind::Global, std::string_view(text, module_size), id) || !push(id))
            {
                return false;
            }
            m_values[id].text2 = std::string_view(text + module_size + 1, name_size);
            m_pos += module_size + 1 + name_size + 1;
            break;
        }
        case 0x93: // STACK_GLOBAL
            if (!pop(b) || !pop(a))
            {
                return false;
            }
            if (m_values[a].kind != Kind::String || m_values[b].kind != Kind::String)
            {
                return fail("STACK_GLOBAL needs two strings");
            }
            if (!make_text(Kind::Global, m_values[a].text, id) || !push(id))
            {
                return false;
            }
            m_values[id].text2 = m_values[b].text;
            break;
        case 't': // TUPLE
            if (!pop_mark(start) || !tuple_from(start, id) || !push(id))
            {
                return false;
            }
            break;
        case 0x85: // TUPLE1
        case 0x86: // TUPLE2
        case 0x87: // TUPLE3
        {
            const size_t n = static_cast<size_t>(op - 0x84);
            const size_t floor = m_marks.empty() ? 0 : m_marks[m_marks.size() - 1];
            if (m_stack.size() < floor + n)
            {
                return fail("stack underflow");
            }
            if (!tuple_from(m_stack.size() - n, id) || !push(id))
            {
                return false;
            }
            break;
        }
        case 'q': // BINPUT
        case 'r': // LONG_BINPUT
        case 0x94: // MEMOIZE
        {
            size_t key = m_memo.size();
            if (op != 0x94)
            {
                const size_t width = op == 'q' ? 1 : 4;
                if (!need(width))
                {
                    return fail("truncated");
                }
                key = op == 'q' ? data[m_pos] : read_u32(data + m_pos);
                m_pos += width;
            }
            if (m_stack.empty())
            {
                return fail("memo of an empty stack");
            }
            if (key > (1u << 24))
            {
                return fail("memo key too large");
            }
            while (m_memo.size() <= key)
            {
                if (!m_memo.push_back(NIL))
                {
                    return fail("out of memory");
                }
            }
            m_memo[key] = m_stack[m_stack.size() - 1];
            break;
        }
        case 'h': // BINGET
        case 'j': // LONG_BINGET
        {
            const size_t width = op == 'h' ? 1 : 4;
            if (!need(width))
            {
                return fail("truncated");
            }
            const size_t key = op == 'h' ? data[m_pos] : read_u32(data + m_pos);
            m_pos += width;
            if (key >= m_memo.size() || m_memo[key] == NIL)
            {
                return fail("memo key not set");
            }
            if (!push(m_memo[key]))
            {
                return false;
            }
            break;
        }
        case 'Q': // BINPERSID
            if (!pop(a) || !persistent(a, id) || !push(id))
            {
                return false;
            }
            break;
        case 'R': // REDUCE
            if (!pop(b) || !pop(a) || !reduce(a, b, id) || !push(id))
            {
                return false;
            }
            break;
        case 'b': // BUILD
            if (!pop(a))
            {
                return false;
            }
            break;
        case 'a': // APPEND
            if (!pop(b) || !pop(a) || m_values[a].kind != Kind::List || !add_child(a, b) || !push(a))
            {
                return m_error->empty() ? fail("APPEND to a non list") : false;
            }
            break;
        case 'e': // APPENDS
        case 'u': // SETITEMS
        {
            if (!pop_mark(start))
            {
                return false;
            }
            const size_t floor = m_marks.empty() ? 0 : m_marks[m_marks.size() - 1];
            if (start == floor || start > m_stack.size())
            {
                return fail("APPENDS/SETITEMS without a target");
            }
            const uint32_t target = m_stack[start - 1];
            const Kind want = op == 'e' ? Kind::List : Kind::Dict;
            if (m_values[target].kind != want || (op == 'u' && (m_stack.size() - start) % 2 != 0))
            {
                return fail("APPENDS/SETITEMS target");
            }
            for (size_t i = start; i < m_stack.size(); ++i)
            {
                if (!add_child(target, m_stack[i]))
                {
                    return false;
                }
            }
            m_stack.resize_uninitialized(start);
            break;
        }
        case 's': // SETITEM
            if (!pop(b) || !pop(a) || m_stack.size() == 0)
            {
                return m_error->empty() ? fail("SETITEM without a target") : false;
            }
            id = m_stack[m_stack.size() - 1];
            if (m_values[id].kind != Kind::Dict || !add_child(id, a) || !add_child(id, b))
            {
                return m_error->empty() ? fail("SETITEM target") : false;
            }
            break;
        default:
        {
            char what[48];
            std::snprintf(what, sizeof(what), "unsupported opcode 0x%02x", op);
            m_pos -= 1;
            return fail(what);
        }
        }
    }
    return fail("missing STOP");
}

bool storage_dtype(std::string_view type, CkptDtype& dtype, uint64_t& element_size)
{
    if (type == "FloatStorage")
    {
        dtype = CkptDtype::F32;
        element_size = 4;
        return true;
    }
    if (type == "HalfStorage")
    {
        dtype = CkptDtype::F16;
        element_size = 2;
        return true;
    }
    if (type == "BFloat16Storage")
    {
        dtype = CkptDtype::BF16;
        element_size = 2;
        return true;
    }
    return false;
}

}

bool CkptFile::open(const char* path, Error& error)
{
    if (!m_file.open(path, error))
    {
        return false;
    }
    return load(m_file.data(), m_file.size(), error);
}

const CkptTensor* CkptFile::find(std::string_view name) const
{
    uint32_t index;
    if (!m_index.find(name, index))
    {
        return nullptr;
    }
    return &m_tensors[index];
}

bool CkptFile::load(const uint8_t* data, size_t size, Error& error)
{
    m_data = data;
    m_size = size;
    m_tensors.clear();
    m_index.clear();

    Zip zip;
    if (!zip.parse(data, size, error))
    {
        return false;
    }

    std::string_view prefix;
    const ZipEntry* pickle = nullptr;
    const std::string_view suffix = "/data.pkl";
    for (size_t i = 0; i < zip.count(); ++i)
    {
        const std::string_view name = zip.entry(i).name;
        if (name.size() > suffix.size() && name.substr(name.size() - suffix.size()) == suffix &&
            name.find('/') == name.size() - suffix.size())
        {
            prefix = name.substr(0, name.size() - suffix.size() + 1);
            pickle = &zip.entry(i);
            break;
        }
    }
    if (pickle == nullptr)
    {
        error.set("no <archive>/data.pkl entry: not a torch zip checkpoint");
        return false;
    }
    if (pickle->method != 0)
    {
        error.set("data.pkl is compressed (method %u)", static_cast<unsigned>(pickle->method));
        return false;
    }

    char path[512];
    std::snprintf(path, sizeof(path), "%.*sbyteorder", static_cast<int>(prefix.size()), prefix.data());
    const ZipEntry* order = zip.find(path);
    if (order != nullptr &&
        std::string_view(reinterpret_cast<const char*>(data + order->offset), static_cast<size_t>(order->size)) !=
            "little")
    {
        error.set("big endian checkpoints are not supported");
        return false;
    }

    Unpickler unpickler;
    uint32_t root;
    if (!unpickler.run(data + pickle->offset, static_cast<size_t>(pickle->size), root, error))
    {
        return false;
    }
    // one tensor of the checkpoint, data pointing into the file
    const auto add = [&](std::string_view name, uint32_t value_id) -> bool {
        const int name_size = static_cast<int>(name.size());
        const Value& t = unpickler.value(value_id);
        if (t.kind != Kind::Tensor)
        {
            error.set("'%.*s' is not a tensor", name_size, name.data());
            return false;
        }
        const Value& storage = unpickler.value(t.refs[0]);
        const Value& shape = unpickler.value(t.refs[1]);
        const Value& stride = unpickler.value(t.refs[2]);

        CkptTensor tensor = {};
        tensor.name = name;
        uint64_t element_size = 0;
        if (!storage_dtype(storage.text2, tensor.dtype, element_size))
        {
            error.set("'%.*s': unsupported storage %.*s", name_size, name.data(),
                      static_cast<int>(storage.text2.size()), storage.text2.data());
            return false;
        }
        if (shape.count > CkptTensor::MAX_DIMS || stride.count != shape.count)
        {
            error.set("'%.*s': %u dims with %u strides", name_size, name.data(), shape.count, stride.count);
            return false;
        }
        tensor.n_dims = shape.count;
        tensor.n_elements = 1;
        for (uint32_t d = 0; d < shape.count; ++d)
        {
            const Value& dim = unpickler.value(unpickler.child(t.refs[1], d));
            if (dim.kind != Kind::Int || dim.number < 0)
            {
                error.set("'%.*s': bad dimension %u", name_size, name.data(), d);
                return false;
            }
            tensor.shape[d] = static_cast<uint64_t>(dim.number);
            if (tensor.shape[d] != 0 && tensor.n_elements > (uint64_t(1) << 48) / tensor.shape[d])
            {
                error.set("'%.*s': too many elements", name_size, name.data());
                return false;
            }
            tensor.n_elements *= tensor.shape[d];
        }
        uint64_t expected = 1;
        for (uint32_t d = shape.count; d-- > 0;)
        {
            const Value& s = unpickler.value(unpickler.child(t.refs[2], d));
            if (s.kind != Kind::Int)
            {
                error.set("'%.*s': bad stride %u", name_size, name.data(), d);
                return false;
            }
            if (tensor.shape[d] != 1 && static_cast<uint64_t>(s.number) != expected)
            {
                error.set("'%.*s': not contiguous", name_size, name.data());
                return false;
            }
            expected *= tensor.shape[d];
        }

        std::snprintf(path, sizeof(path), "%.*sdata/%.*s", static_cast<int>(prefix.size()), prefix.data(),
                      static_cast<int>(storage.text.size()), storage.text.data());
        const ZipEntry* entry = zip.find(path);
        if (entry == nullptr || entry->method != 0)
        {
            error.set("'%.*s': storage %s missing or compressed", name_size, name.data(), path);
            return false;
        }
        if (storage.number < 0 || t.number < 0 || static_cast<uint64_t>(storage.number) > entry->size / element_size ||
            tensor.n_elements > static_cast<uint64_t>(storage.number) ||
            static_cast<uint64_t>(t.number) > static_cast<uint64_t>(storage.number) - tensor.n_elements)
        {
            error.set("'%.*s': outside its storage", name_size, name.data());
            return false;
        }
        tensor.data = data + entry->offset + static_cast<uint64_t>(t.number) * element_size;
        tensor.n_bytes = tensor.n_elements * element_size;

        if (m_index.insert(tensor.name, static_cast<uint32_t>(m_tensors.size())) != StringMap::Insert::Added)
        {
            error.set("'%.*s' duplicated", name_size, name.data());
            return false;
        }
        m_tensors.push_back_unchecked(tensor);
        return true;
    };

    // torch.save(tensor): one tensor at the root, found under the empty name
    if (unpickler.value(root).kind == Kind::Tensor)
    {
        if (!m_tensors.reserve(1) || !m_index.reserve(1))
        {
            error.set("out of memory for 1 tensor");
            return false;
        }
        return add(std::string_view(), root);
    }
    if (unpickler.value(root).kind != Kind::Dict)
    {
        error.set("checkpoint root is neither a dict nor a tensor");
        return false;
    }

    for (uint32_t item = unpickler.first_item(root); item != NIL;)
    {
        const uint32_t key = unpickler.item(item).value;
        const uint32_t next = unpickler.item(item).next;
        const uint32_t value = unpickler.item(next).value;
        if (unpickler.value(key).kind == Kind::String && unpickler.value(key).text == "state_dict" &&
            unpickler.value(value).kind == Kind::Dict)
        {
            root = value;
            break;
        }
        item = unpickler.item(next).next;
    }

    const uint32_t n = unpickler.value(root).count / 2;
    if (!m_tensors.reserve(n) || !m_index.reserve(n))
    {
        error.set("out of memory for %u tensors", n);
        return false;
    }
    for (uint32_t item = unpickler.first_item(root); item != NIL;)
    {
        const uint32_t key_id = unpickler.item(item).value;
        const uint32_t next = unpickler.item(item).next;
        const uint32_t value_id = unpickler.item(next).value;
        item = unpickler.item(next).next;
        const Value& key = unpickler.value(key_id);
        if (key.kind != Kind::String)
        {
            error.set("state dict key is not a string");
            return false;
        }
        if (!add(key.text, value_id))
        {
            return false;
        }
    }
    return true;
}

}
