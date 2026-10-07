#include "faisca/tokenizer.h"

#include "faisca/unicode.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>

namespace faisca
{

namespace
{

constexpr uint32_t DEAD = 0xffffffffu;
constexpr uint8_t OUT_OF_RANGE = 0x80;
constexpr size_t NOT_FOUND = static_cast<size_t>(-1);
constexpr uint32_t BYTE_CODE_POINTS = 324;

// GPT-2 writes each byte as one printable code point
struct ByteTables
{
    uint16_t code_point[256];
    int16_t byte[BYTE_CODE_POINTS];
};

constexpr bool is_printable_byte(uint32_t byte)
{
    return (byte >= 0x21 && byte <= 0x7e) || (byte >= 0xa1 && byte <= 0xac) || (byte >= 0xae && byte <= 0xff);
}

constexpr ByteTables make_byte_tables()
{
    ByteTables tables{};
    for (uint32_t i = 0; i < BYTE_CODE_POINTS; ++i)
    {
        tables.byte[i] = -1;
    }

    uint32_t next = 256;
    for (uint32_t byte = 0; byte < 256; ++byte)
    {
        const uint32_t code_point = is_printable_byte(byte) ? byte : next++;
        tables.code_point[byte] = static_cast<uint16_t>(code_point);
        tables.byte[code_point] = static_cast<int16_t>(byte);
    }
    return tables;
}

constexpr ByteTables BYTE_TABLES = make_byte_tables();

size_t find_bytes(const char* text, size_t size, size_t from, std::string_view pattern)
{
    if (pattern.empty() || size < pattern.size() || from > size - pattern.size())
    {
        return NOT_FOUND;
    }

    const size_t last = size - pattern.size();
    size_t pos = from;
    while (pos <= last)
    {
        const void* hit = std::memchr(text + pos, pattern[0], last - pos + 1);
        if (hit == nullptr)
        {
            return NOT_FOUND;
        }
        pos = static_cast<size_t>(static_cast<const char*>(hit) - text);
        if (std::memcmp(text + pos, pattern.data(), pattern.size()) == 0)
        {
            return pos;
        }
        ++pos;
    }
    return NOT_FOUND;
}

uint64_t pair_key(uint32_t left, uint32_t right)
{
    return (static_cast<uint64_t>(left) << 32) | right;
}

bool bigram_before(uint32_t rank_a, int32_t left_a, uint32_t rank_b, int32_t left_b)
{
    return rank_a < rank_b || (rank_a == rank_b && left_a < left_b);
}

TokenType token_type_from(int32_t value)
{
    if (value >= static_cast<int32_t>(TokenType::Normal) && value <= static_cast<int32_t>(TokenType::Byte))
    {
        return static_cast<TokenType>(value);
    }
    return TokenType::Undefined;
}

bool is_special_type(TokenType type)
{
    return type == TokenType::Control || type == TokenType::Unknown;
}

}

void Tokenizer::release()
{
    m_arena.release();
    m_tokens.release();
    m_symbols.release();
    m_symbol_index.release();
    m_merges.release();
    m_merged.release();
    m_special.release();

    m_pre = PreTokenizer::Gpt2;
    m_clean_spaces = false;
    m_add_space_prefix = false;
    m_add_bos = false;
    m_add_eos = false;
    m_bos = TOKEN_NONE;
    m_eos = TOKEN_NONE;

    m_fragments.release();
    m_fragments_next.release();
    m_code_points.release();
    m_words.release();
    m_words_next.release();
    m_nodes.release();
    m_heap.release();
    m_scratch.release();
}

bool Tokenizer::load(const GgufFile& file, Error& error)
{
    release();
    if (load_tokens(file, error) && load_merges(file, error) && load_special_tokens(file, error))
    {
        return true;
    }
    release();
    return false;
}

bool Tokenizer::intern(std::string_view text, bool copy, uint32_t& symbol)
{
    if (m_symbol_index.find(text, symbol))
    {
        return true;
    }

    if (copy)
    {
        const char* stored = m_arena.copy(text.data(), text.size());
        if (stored == nullptr)
        {
            return false;
        }
        text = std::string_view(stored, text.size());
    }

    if (m_symbols.size() >= DEAD)
    {
        return false;
    }
    symbol = static_cast<uint32_t>(m_symbols.size());

    Symbol entry;
    entry.text = text;
    entry.token = TOKEN_NONE;
    return m_symbols.push_back(entry) && m_symbol_index.insert(text, symbol) == StringMap::Insert::Added;
}

int32_t Tokenizer::find_token(std::string_view text) const
{
    uint32_t symbol = 0;
    if (!m_symbol_index.find(text, symbol))
    {
        return TOKEN_NONE;
    }
    return m_symbols[symbol].token;
}

bool Tokenizer::load_tokens(const GgufFile& file, Error& error)
{
    std::string_view model;
    if (!file.get_string("tokenizer.ggml.model", model))
    {
        error.set("tokenizer.ggml.model is missing");
        return false;
    }
    if (model != "gpt2")
    {
        error.set("tokenizer model '%.*s' is not supported, only 'gpt2' (byte level BPE)", static_cast<int>(model.size()),
                  model.data());
        return false;
    }

    std::string_view pre;
    if (!file.get_string("tokenizer.ggml.pre", pre))
    {
        error.set("tokenizer.ggml.pre is missing");
        return false;
    }
    if (pre == "gpt-2" || pre == "phi-2")
    {
        m_pre = PreTokenizer::Gpt2;
        m_clean_spaces = true;
    }
    else if (pre == "smollm")
    {
        m_pre = PreTokenizer::SmolLm;
        m_clean_spaces = false;
    }
    else
    {
        error.set("pre-tokenizer '%.*s' is not supported yet (supported: gpt-2, phi-2, smollm)",
                  static_cast<int>(pre.size()), pre.data());
        return false;
    }

    const GgufArray* token_array = file.get_array("tokenizer.ggml.tokens");
    if (token_array == nullptr || token_array->elem_type != GgufType::String)
    {
        error.set("tokenizer.ggml.tokens is missing or is not an array of strings");
        return false;
    }

    Array<std::string_view> texts;
    if (!token_array->strings_to(texts))
    {
        error.set("out of memory while reading the vocabulary");
        return false;
    }
    const size_t count = texts.size();
    if (count == 0 || count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    {
        error.set("tokenizer.ggml.tokens has %zu entries", count);
        return false;
    }

    Array<int32_t> types;
    const GgufArray* type_array = file.get_array("tokenizer.ggml.token_type");
    if (type_array != nullptr)
    {
        if (type_array->elem_type != GgufType::Int32 || type_array->count < count)
        {
            error.set("tokenizer.ggml.token_type is not an int32 array with one entry per token");
            return false;
        }
        if (!type_array->copy_to(types))
        {
            error.set("out of memory while reading the vocabulary");
            return false;
        }
    }

    if (!m_tokens.reserve(count) || !m_symbols.reserve(count + 256) || !m_symbol_index.reserve(count + 256))
    {
        error.set("out of memory while reading the vocabulary");
        return false;
    }

    for (size_t i = 0; i < count; ++i)
    {
        std::string_view text = texts[i];
        if (text.empty())
        {
            char name[32];
            const int length = std::snprintf(name, sizeof(name), "[EMPTY_%zu]", i);
            const char* stored = m_arena.copy(name, static_cast<size_t>(length));
            if (stored == nullptr)
            {
                error.set("out of memory while reading the vocabulary");
                return false;
            }
            text = std::string_view(stored, static_cast<size_t>(length));
        }

        Token token;
        token.text = text;
        token.type = types.empty() ? TokenType::Normal : token_type_from(types[i]);
        m_tokens.push_back_unchecked(token);

        uint32_t symbol = 0;
        if (!intern(text, false, symbol))
        {
            error.set("out of memory while reading the vocabulary");
            return false;
        }
        // when two tokens share a text the last one wins, as in llama.cpp
        m_symbols[symbol].token = static_cast<int32_t>(i);
    }

    for (uint32_t byte = 0; byte < 256; ++byte)
    {
        char encoded[4];
        const size_t length = utf8_encode(BYTE_TABLES.code_point[byte], encoded);
        if (!intern(std::string_view(encoded, length), true, m_byte_symbol[byte]))
        {
            error.set("out of memory while reading the vocabulary");
            return false;
        }
    }
    for (uint32_t byte = 0; byte < 256; ++byte)
    {
        const char raw = static_cast<char>(byte);
        m_byte_token[byte] = find_token(std::string_view(&raw, 1));
    }

    return true;
}

bool Tokenizer::load_merges(const GgufFile& file, Error& error)
{
    const GgufArray* merge_array = file.get_array("tokenizer.ggml.merges");
    if (merge_array == nullptr || merge_array->elem_type != GgufType::String)
    {
        error.set("tokenizer.ggml.merges is missing or is not an array of strings");
        return false;
    }

    Array<std::string_view> merges;
    if (!merge_array->strings_to(merges) || !m_merges.reserve(merges.size()) || !m_merged.reserve(merges.size()))
    {
        error.set("out of memory while reading the merges");
        return false;
    }
    if (merges.size() >= DEAD)
    {
        error.set("tokenizer.ggml.merges has %zu entries", merges.size());
        return false;
    }

    for (size_t rank = 0; rank < merges.size(); ++rank)
    {
        const std::string_view merge = merges[rank];
        uint32_t merged = DEAD;

        // "left right", split at the first space after the first byte
        const void* space = merge.size() > 1 ? std::memchr(merge.data() + 1, ' ', merge.size() - 1) : nullptr;
        if (space != nullptr)
        {
            const size_t split = static_cast<size_t>(static_cast<const char*>(space) - merge.data());
            const std::string_view left = merge.substr(0, split);
            const std::string_view right = merge.substr(split + 1);

            if (!right.empty())
            {
                uint32_t left_symbol = 0;
                uint32_t right_symbol = 0;
                if (!m_scratch.resize_uninitialized(left.size() + right.size()))
                {
                    error.set("out of memory while reading the merges");
                    return false;
                }
                std::memcpy(m_scratch.data(), left.data(), left.size());
                std::memcpy(m_scratch.data() + left.size(), right.data(), right.size());

                if (!intern(left, false, left_symbol) || !intern(right, false, right_symbol) ||
                    !intern(std::string_view(m_scratch.data(), m_scratch.size()), true, merged) ||
                    m_merges.insert(pair_key(left_symbol, right_symbol), static_cast<uint32_t>(rank)) ==
                        IntMap::Insert::Failed)
                {
                    error.set("out of memory while reading the merges");
                    return false;
                }
            }
        }

        m_merged.push_back_unchecked(merged);
    }

    return true;
}

bool Tokenizer::load_special_tokens(const GgufFile& file, Error& error)
{
    const uint64_t count = m_tokens.size();

    // llama.cpp defaults for this tokenizer model
    m_bos = count > 11 ? 11 : TOKEN_NONE;
    m_eos = count > 11 ? 11 : TOKEN_NONE;

    uint64_t id = 0;
    if (file.get_uint("tokenizer.ggml.bos_token_id", id) && id < count)
    {
        m_bos = static_cast<int32_t>(id);
    }
    if (file.get_uint("tokenizer.ggml.eos_token_id", id) && id < count)
    {
        m_eos = static_cast<int32_t>(id);
    }

    file.get_bool("tokenizer.ggml.add_bos_token", m_add_bos);
    file.get_bool("tokenizer.ggml.add_eos_token", m_add_eos);
    file.get_bool("tokenizer.ggml.add_space_prefix", m_add_space_prefix);

    for (size_t i = 0; i < m_tokens.size(); ++i)
    {
        const TokenType type = m_tokens[i].type;
        if (type != TokenType::Control && type != TokenType::UserDefined && type != TokenType::Unknown)
        {
            continue;
        }
        if (!m_special.push_back(static_cast<int32_t>(i)))
        {
            error.set("out of memory while reading the vocabulary");
            return false;
        }
    }

    // longest text first, ties by id
    for (size_t i = 1; i < m_special.size(); ++i)
    {
        const int32_t token = m_special[i];
        const size_t length = token_text(token).size();
        size_t at = i;
        while (at > 0 && token_text(m_special[at - 1]).size() < length)
        {
            m_special[at] = m_special[at - 1];
            --at;
        }
        m_special[at] = token;
    }

    return true;
}

bool Tokenizer::partition(std::string_view text, bool parse_special)
{
    m_fragments.clear();

    Fragment whole;
    whole.token = TOKEN_NONE;
    whole.offset = 0;
    whole.length = text.size();
    if (!m_fragments.push_back(whole))
    {
        return false;
    }
    if (m_special.empty())
    {
        return true;
    }

    bool present[256] = {};
    for (const char c : text)
    {
        present[static_cast<uint8_t>(c)] = true;
    }

    for (const int32_t special : m_special)
    {
        const Token& token = m_tokens[static_cast<size_t>(special)];
        if (!parse_special && is_special_type(token.type))
        {
            continue;
        }
        if (!present[static_cast<uint8_t>(token.text[0])])
        {
            continue;
        }

        m_fragments_next.clear();
        for (const Fragment& fragment : m_fragments)
        {
            if (fragment.token != TOKEN_NONE)
            {
                if (!m_fragments_next.push_back(fragment))
                {
                    return false;
                }
                continue;
            }

            size_t offset = fragment.offset;
            size_t length = fragment.length;
            for (;;)
            {
                const size_t match = find_bytes(text.data(), offset + length, offset, token.text);

                Fragment raw;
                raw.token = TOKEN_NONE;
                raw.offset = offset;
                raw.length = match == NOT_FOUND ? length : match - offset;
                if (raw.length > 0 && !m_fragments_next.push_back(raw))
                {
                    return false;
                }
                if (match == NOT_FOUND)
                {
                    break;
                }

                Fragment found;
                found.token = special;
                found.offset = match;
                found.length = token.text.size();
                if (!m_fragments_next.push_back(found))
                {
                    return false;
                }

                const size_t end = match + token.text.size();
                length = offset + length - end;
                offset = end;
                if (length == 0)
                {
                    break;
                }
            }
        }
        m_fragments.swap(m_fragments_next);
    }

    return true;
}

bool Tokenizer::encode(std::string_view text, bool add_special, bool parse_special, Array<int32_t>& out)
{
    m_fragments.clear();
    if (!text.empty() && !partition(text, parse_special))
    {
        return false;
    }

    if (add_special && m_add_bos && m_bos != TOKEN_NONE && !out.push_back(m_bos))
    {
        return false;
    }

    for (size_t i = 0; i < m_fragments.size(); ++i)
    {
        const Fragment fragment = m_fragments[i];
        if (fragment.token != TOKEN_NONE)
        {
            if (!out.push_back(fragment.token))
            {
                return false;
            }
        }
        else if (!encode_text(text.substr(fragment.offset, fragment.length), out))
        {
            return false;
        }
    }

    if (add_special && m_add_eos && m_eos != TOKEN_NONE && !out.push_back(m_eos))
    {
        return false;
    }
    return true;
}

bool Tokenizer::encode_text(std::string_view text, Array<int32_t>& out)
{
    m_code_points.clear();
    if (!m_code_points.reserve(text.size()))
    {
        return false;
    }
    for (size_t offset = 0; offset < text.size();)
    {
        m_code_points.push_back_unchecked(utf8_decode(text.data(), text.size(), offset));
    }

    m_words.clear();
    if (!m_words.push_back(static_cast<uint32_t>(m_code_points.size())))
    {
        return false;
    }

    if (m_pre == PreTokenizer::SmolLm)
    {
        if (!split_numbers(m_words, m_words_next))
        {
            return false;
        }
        m_words.swap(m_words_next);
    }
    if (!split_gpt2(m_words, m_words_next))
    {
        return false;
    }
    m_words.swap(m_words_next);

    size_t start = 0;
    for (size_t i = 0; i < m_words.size(); ++i)
    {
        const size_t length = m_words[i];
        if (!encode_word(m_code_points.data() + start, length, out))
        {
            return false;
        }
        start += length;
    }
    return true;
}

// regex \p{N}: every number becomes a piece of its own
bool Tokenizer::split_numbers(const Array<uint32_t>& in, Array<uint32_t>& out) const
{
    out.clear();

    size_t start = 0;
    for (const uint32_t length : in)
    {
        const size_t end = start + length;
        size_t gap = start;
        for (size_t pos = start; pos < end; ++pos)
        {
            const uint32_t code_point = m_code_points[pos];
            const uint8_t type = unicode_class(code_point);
            const bool number = (type & UNICODE_NUMBER) != 0 && !(code_point >= 128 && (type & UNICODE_WHITESPACE) != 0);
            if (!number)
            {
                continue;
            }
            if (pos > gap && !out.push_back(static_cast<uint32_t>(pos - gap)))
            {
                return false;
            }
            if (!out.push_back(1))
            {
                return false;
            }
            gap = pos + 1;
        }
        if (end > gap && !out.push_back(static_cast<uint32_t>(end - gap)))
        {
            return false;
        }
        start = end;
    }
    return true;
}

// regex 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
// follows llama.cpp's unicode_regex_split_custom_gpt2 step by step
bool Tokenizer::split_gpt2(const Array<uint32_t>& in, Array<uint32_t>& out) const
{
    out.clear();

    const uint32_t* code_points = m_code_points.data();
    const uint8_t word_classes = UNICODE_LETTER | UNICODE_NUMBER | UNICODE_WHITESPACE | OUT_OF_RANGE;

    size_t start = 0;
    for (const uint32_t length : in)
    {
        const size_t begin = start;
        const size_t end = start + length;
        start = end;

        const auto class_at = [code_points, end](size_t pos) -> uint8_t
        {
            return pos < end ? unicode_class(code_points[pos]) : OUT_OF_RANGE;
        };

        size_t previous = begin;
        size_t pos = begin;
        while (pos < end)
        {
            const uint32_t code_point = code_points[pos];
            size_t next = 0;

            if (code_point == '\'' && pos + 1 < end)
            {
                const uint32_t a = code_points[pos + 1];
                if (a == 's' || a == 't' || a == 'm' || a == 'd')
                {
                    next = pos + 2;
                }
                else if (pos + 2 < end)
                {
                    const uint32_t b = code_points[pos + 2];
                    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
                    {
                        next = pos + 3;
                    }
                }
            }

            if (next == 0)
            {
                const size_t skip = code_point == ' ' ? 1 : 0;
                uint8_t type = class_at(pos + skip);
                size_t scan = pos + skip;

                if ((type & UNICODE_LETTER) != 0)
                {
                    while ((type & UNICODE_LETTER) != 0)
                    {
                        type = class_at(++scan);
                    }
                    next = scan;
                }
                else if ((type & UNICODE_NUMBER) != 0)
                {
                    while ((type & UNICODE_NUMBER) != 0)
                    {
                        type = class_at(++scan);
                    }
                    next = scan;
                }
                else if ((type & word_classes) == 0)
                {
                    while ((type & word_classes) == 0)
                    {
                        type = class_at(++scan);
                    }
                    next = scan;
                }
            }

            if (next == 0)
            {
                size_t spaces = 0;
                while ((class_at(pos + spaces) & UNICODE_WHITESPACE) != 0)
                {
                    ++spaces;
                }

                if (spaces > 1 && pos + spaces < end)
                {
                    next = pos + spaces - 1;
                }
                else if (spaces > 0)
                {
                    next = pos + spaces;
                }
                else
                {
                    next = pos + 1;
                }
            }

            if (!out.push_back(static_cast<uint32_t>(next - previous)))
            {
                return false;
            }
            previous = next;
            pos = next;
        }
    }
    return true;
}

bool Tokenizer::push_bigram(int32_t left, int32_t right)
{
    if (left < 0 || right < 0)
    {
        return true;
    }

    Bigram bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.left_symbol = m_nodes[static_cast<size_t>(left)].symbol;
    bigram.right_symbol = m_nodes[static_cast<size_t>(right)].symbol;
    if (!m_merges.find(pair_key(bigram.left_symbol, bigram.right_symbol), bigram.rank))
    {
        return true;
    }

    if (!m_heap.push_back(bigram))
    {
        return false;
    }
    size_t at = m_heap.size() - 1;
    while (at > 0)
    {
        const size_t parent = (at - 1) / 2;
        if (!bigram_before(bigram.rank, bigram.left, m_heap[parent].rank, m_heap[parent].left))
        {
            break;
        }
        m_heap[at] = m_heap[parent];
        at = parent;
    }
    m_heap[at] = bigram;
    return true;
}

Tokenizer::Bigram Tokenizer::pop_bigram()
{
    const Bigram top = m_heap[0];
    const Bigram last = m_heap[m_heap.size() - 1];
    const size_t count = m_heap.size() - 1;
    m_heap.resize_uninitialized(count);

    size_t at = 0;
    for (;;)
    {
        size_t child = at * 2 + 1;
        if (child >= count)
        {
            break;
        }
        if (child + 1 < count &&
            bigram_before(m_heap[child + 1].rank, m_heap[child + 1].left, m_heap[child].rank, m_heap[child].left))
        {
            ++child;
        }
        if (!bigram_before(m_heap[child].rank, m_heap[child].left, last.rank, last.left))
        {
            break;
        }
        m_heap[at] = m_heap[child];
        at = child;
    }
    if (count > 0)
    {
        m_heap[at] = last;
    }
    return top;
}

bool Tokenizer::encode_word(const uint32_t* code_points, size_t count, Array<int32_t>& out)
{
    m_nodes.clear();
    m_heap.clear();

    for (size_t i = 0; i < count; ++i)
    {
        char bytes[4];
        const size_t length = utf8_encode(code_points[i], bytes);
        for (size_t b = 0; b < length; ++b)
        {
            Node node;
            node.symbol = m_byte_symbol[static_cast<uint8_t>(bytes[b])];
            node.prev = static_cast<int32_t>(m_nodes.size()) - 1;
            node.next = static_cast<int32_t>(m_nodes.size()) + 1;
            if (!m_nodes.push_back(node))
            {
                return false;
            }
        }
    }
    if (m_nodes.empty())
    {
        return true;
    }
    m_nodes[m_nodes.size() - 1].next = -1;

    for (size_t i = 1; i < m_nodes.size(); ++i)
    {
        if (!push_bigram(static_cast<int32_t>(i) - 1, static_cast<int32_t>(i)))
        {
            return false;
        }
    }

    while (!m_heap.empty())
    {
        const Bigram bigram = pop_bigram();
        Node& left = m_nodes[static_cast<size_t>(bigram.left)];
        Node& right = m_nodes[static_cast<size_t>(bigram.right)];

        // stale entries: one side was merged away or has grown since the push
        if (left.symbol != bigram.left_symbol || right.symbol != bigram.right_symbol)
        {
            continue;
        }

        left.symbol = m_merged[bigram.rank];
        right.symbol = DEAD;
        left.next = right.next;
        if (right.next >= 0)
        {
            m_nodes[static_cast<size_t>(right.next)].prev = bigram.left;
        }

        if (!push_bigram(left.prev, bigram.left) || !push_bigram(bigram.left, left.next))
        {
            return false;
        }
    }

    for (const Node& node : m_nodes)
    {
        if (node.symbol == DEAD)
        {
            continue;
        }

        const Symbol& symbol = m_symbols[node.symbol];
        if (symbol.token != TOKEN_NONE)
        {
            if (!out.push_back(symbol.token))
            {
                return false;
            }
            continue;
        }

        // not a token: llama.cpp falls back to the tokens of its single bytes
        for (const char c : symbol.text)
        {
            const int32_t token = m_byte_token[static_cast<uint8_t>(c)];
            if (token != TOKEN_NONE && !out.push_back(token))
            {
                return false;
            }
        }
    }
    return true;
}

bool Tokenizer::append_piece(int32_t token, bool special, bool strip_space, Array<char>& out) const
{
    if (token < 0 || static_cast<size_t>(token) >= m_tokens.size())
    {
        return true;
    }

    const Token& entry = m_tokens[static_cast<size_t>(token)];
    if (!special && is_special_type(entry.type))
    {
        return true;
    }

    const size_t start = out.size();

    if (is_special_type(entry.type) || entry.type == TokenType::UserDefined)
    {
        for (const char c : entry.text)
        {
            if (!out.push_back(c))
            {
                return false;
            }
        }
    }
    else if (entry.type == TokenType::Normal)
    {
        for (size_t offset = 0; offset < entry.text.size();)
        {
            const uint32_t code_point = utf8_decode(entry.text.data(), entry.text.size(), offset);
            if (code_point < BYTE_CODE_POINTS && BYTE_TABLES.byte[code_point] >= 0)
            {
                if (!out.push_back(static_cast<char>(BYTE_TABLES.byte[code_point])))
                {
                    return false;
                }
                continue;
            }

            // same marker llama.cpp writes for a character outside the byte alphabet
            char encoded[4];
            const size_t length = utf8_encode(code_point, encoded);
            char marker[32];
            int written = std::snprintf(marker, sizeof(marker), "[UNK_BYTE_0x");
            for (size_t i = 0; i < length; ++i)
            {
                written += std::snprintf(marker + written, sizeof(marker) - static_cast<size_t>(written), "%02x",
                                         static_cast<unsigned>(static_cast<uint8_t>(encoded[i])));
            }
            for (int i = 0; i < written; ++i)
            {
                if (!out.push_back(marker[i]))
                {
                    return false;
                }
            }
            for (const char c : entry.text)
            {
                if (!out.push_back(c))
                {
                    return false;
                }
            }
            if (!out.push_back(']'))
            {
                return false;
            }
        }
    }
    else if (entry.type == TokenType::Byte && entry.text.size() >= 5)
    {
        // "<0xNN>"
        unsigned value = 0;
        for (size_t i = 3; i < 5; ++i)
        {
            const char c = entry.text[i];
            unsigned digit = 0;
            if (c >= '0' && c <= '9')
            {
                digit = static_cast<unsigned>(c - '0');
            }
            else if (c >= 'a' && c <= 'f')
            {
                digit = static_cast<unsigned>(c - 'a' + 10);
            }
            else if (c >= 'A' && c <= 'F')
            {
                digit = static_cast<unsigned>(c - 'A' + 10);
            }
            else
            {
                break;
            }
            value = value * 16 + digit;
        }
        if (!out.push_back(static_cast<char>(value)))
        {
            return false;
        }
    }

    if (strip_space && out.size() > start && out[start] == ' ')
    {
        std::memmove(out.data() + start, out.data() + start + 1, out.size() - start - 1);
        out.resize_uninitialized(out.size() - 1);
    }
    return true;
}

bool Tokenizer::token_to_piece(int32_t token, bool special, Array<char>& out) const
{
    return append_piece(token, special, false, out);
}

bool Tokenizer::decode(const int32_t* tokens, size_t count, bool remove_special, bool unparse_special,
                       Array<char>& out) const
{
    const size_t start = out.size();
    bool strip_space = m_add_space_prefix;

    if (remove_special && m_add_bos && count > 0 && tokens[0] == m_bos)
    {
        strip_space = false;
        ++tokens;
        --count;
    }
    if (remove_special && m_add_eos && count > 0 && tokens[count - 1] == m_eos)
    {
        --count;
    }

    for (size_t i = 0; i < count; ++i)
    {
        if (!append_piece(tokens[i], unparse_special, strip_space, out))
        {
            return false;
        }
        strip_space = false;
    }

    if (!m_clean_spaces)
    {
        return true;
    }

    // the three passes of llama.cpp's clean_up_tokenization_spaces, in place
    char* text = out.data() + start;
    size_t total = out.size() - start;

    const size_t total1 = total;
    total = total != 0 ? 1 : 0;
    for (size_t i = 1; i < total1; ++i)
    {
        const char x = text[i];
        if (text[i - 1] == ' ' && (x == '?' || x == '!' || x == '.' || x == ','))
        {
            --total;
        }
        text[total++] = x;
    }

    const size_t total2 = total;
    total = total != 0 ? 1 : 0;
    for (size_t i = 1; i < total2; ++i)
    {
        const char x = text[i];
        if (x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ')
        {
            --total;
            text[++i] = '\0';
        }
        text[total++] = x;
    }

    const size_t total3 = total;
    total = total != 0 ? 1 : 0;
    for (size_t i = 1; i < total3; ++i)
    {
        const char x = text[i];
        if (text[i - 1] == ' ' && x == '\'' && i + 1 < total3)
        {
            const char x1 = text[i + 1];
            if (x1 == 's' || x1 == 'm')
            {
                --total;
            }
            else if (x1 != 't' && x1 != 'd' && i + 2 < total3)
            {
                const char x2 = text[i + 2];
                if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e'))
                {
                    --total;
                }
            }
        }
        text[total++] = x;
    }

    out.resize_uninitialized(start + total);
    return true;
}

}
