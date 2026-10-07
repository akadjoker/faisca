#pragma once

#include "faisca/arena.h"
#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/gguf.h"
#include "faisca/int_map.h"
#include "faisca/string_map.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace faisca
{

enum class TokenType : int32_t
{
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
};

constexpr int32_t TOKEN_NONE = -1;

// token texts are views into the GGUF file, which must stay open.
// encode() uses scratch buffers of the object: one Tokenizer per thread
class Tokenizer
{
public:
    Tokenizer() = default;

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    bool load(const GgufFile& file, Error& error);

    void release();

    bool is_loaded() const
    {
        return m_tokens.size() != 0;
    }

    size_t vocab_size() const
    {
        return m_tokens.size();
    }

    std::string_view token_text(int32_t token) const
    {
        return m_tokens[static_cast<size_t>(token)].text;
    }

    TokenType token_type(int32_t token) const
    {
        return m_tokens[static_cast<size_t>(token)].type;
    }

    int32_t find_token(std::string_view text) const;

    int32_t bos() const
    {
        return m_bos;
    }

    int32_t eos() const
    {
        return m_eos;
    }

    bool add_bos() const
    {
        return m_add_bos;
    }

    bool add_eos() const
    {
        return m_add_eos;
    }

    // appends to out, false only when out of memory
    bool encode(std::string_view text, bool add_special, bool parse_special, Array<int32_t>& out);

    bool token_to_piece(int32_t token, bool special, Array<char>& out) const;

    bool decode(const int32_t* tokens, size_t count, bool remove_special, bool unparse_special,
                Array<char>& out) const;

private:
    enum class PreTokenizer
    {
        Gpt2,
        SmolLm,
    };

    struct Token
    {
        std::string_view text;
        TokenType type;
    };

    struct Symbol
    {
        std::string_view text;
        int32_t token;
    };

    struct Fragment
    {
        int32_t token; // TOKEN_NONE for raw text
        size_t offset;
        size_t length;
    };

    struct Node
    {
        uint32_t symbol;
        int32_t prev;
        int32_t next;
    };

    struct Bigram
    {
        uint32_t rank;
        int32_t left;
        int32_t right;
        uint32_t left_symbol;
        uint32_t right_symbol;
    };

    bool intern(std::string_view text, bool copy, uint32_t& symbol);
    bool append_piece(int32_t token, bool special, bool strip_space, Array<char>& out) const;
    bool load_tokens(const GgufFile& file, Error& error);
    bool load_merges(const GgufFile& file, Error& error);
    bool load_special_tokens(const GgufFile& file, Error& error);

    bool partition(std::string_view text, bool parse_special);
    bool encode_text(std::string_view text, Array<int32_t>& out);
    bool split_numbers(const Array<uint32_t>& in, Array<uint32_t>& out) const;
    bool split_gpt2(const Array<uint32_t>& in, Array<uint32_t>& out) const;
    bool encode_word(const uint32_t* code_points, size_t count, Array<int32_t>& out);
    bool push_bigram(int32_t left, int32_t right);
    Bigram pop_bigram();

    Arena m_arena;
    Array<Token> m_tokens;
    Array<Symbol> m_symbols;
    StringMap m_symbol_index;
    IntMap m_merges; // (left symbol, right symbol) to rank
    Array<uint32_t> m_merged; // rank to merged symbol
    Array<int32_t> m_special;

    uint32_t m_byte_symbol[256] = {};
    int32_t m_byte_token[256] = {};

    PreTokenizer m_pre = PreTokenizer::Gpt2;
    bool m_clean_spaces = false;
    bool m_add_space_prefix = false;
    bool m_add_bos = false;
    bool m_add_eos = false;
    int32_t m_bos = TOKEN_NONE;
    int32_t m_eos = TOKEN_NONE;

    Array<Fragment> m_fragments;
    Array<Fragment> m_fragments_next;
    Array<uint32_t> m_code_points;
    Array<uint32_t> m_words;
    Array<uint32_t> m_words_next;
    Array<Node> m_nodes;
    Array<Bigram> m_heap;
    Array<char> m_scratch;
};

}
