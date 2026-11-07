#pragma once

#include "faisca/arena.h"
#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/gguf.h"
#include "faisca/int_map.h"
#include "faisca/json.h"
#include "faisca/mapped_file.h"
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

// token texts are views into the GGUF file, which must stay open. Two models: byte level
// BPE (tokenizer.ggml.model gpt2) and SentencePiece (llama: the vocabulary's pieces merged
// by score, spaces as U+2581, bytes <0xXX> for what no piece covers).
// encode() uses scratch buffers and a word cache of the object: one Tokenizer per thread
class Tokenizer
{
public:
    Tokenizer() = default;

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    bool load(const GgufFile& file, Error& error);
    // a Hugging Face tokenizer.json (byte level BPE with the Qwen2 pre-tokenizer); the file is
    // kept mapped, token texts point into it
    bool load_hf(const char* path, Error& error);

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

    // encode() remembers the tokens of every word it sees, up to a few MB
    void clear_word_cache()
    {
        forget_words();
    }

    bool token_to_piece(int32_t token, bool special, Array<char>& out) const;

    bool decode(const int32_t* tokens, size_t count, bool remove_special, bool unparse_special,
                Array<char>& out) const;

private:
    enum class PreTokenizer
    {
        Gpt2,
        SmolLm,
        Llama3,
        Qwen2,
    };

    enum class Model
    {
        Bpe,
        Spm,
    };

    struct SpmSymbol
    {
        uint32_t offset;
        uint32_t length;
        int32_t prev;
        int32_t next;
    };

    struct SpmBigram
    {
        float score;
        int32_t left;
        int32_t right;
        uint32_t size;
    };

    struct Token
    {
        std::string_view text;
        TokenType type;
        uint32_t piece_offset;
        uint32_t piece_length;
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

    struct CachedWord
    {
        uint32_t offset;
        uint32_t count;
    };

    bool intern(std::string_view text, bool copy, uint32_t& symbol);
    bool append_piece(int32_t token, bool special, bool strip_space, Array<char>& out) const;
    bool load_tokens(const GgufFile& file, Error& error);
    bool build_pieces(Error& error);
    bool load_merges(const GgufFile& file, Error& error);
    bool load_special_tokens(const GgufFile& file, Error& error);
    bool set_vocabulary(const Array<std::string_view>& texts, const Array<int32_t>& types, Error& error);
    bool set_merges(const Array<std::string_view>& merges, Error& error);
    bool collect_special_tokens(Error& error);

    bool partition(std::string_view text, bool parse_special);
    bool encode_text(std::string_view text, Array<int32_t>& out);
    bool split_numbers(const Array<uint32_t>& in, Array<uint32_t>& out) const;
    bool split_gpt2(const Array<uint32_t>& in, Array<uint32_t>& out) const;
    // max_digits: 3 for the llama 3 regex, 1 for qwen2's
    bool split_llama3(const Array<uint32_t>& in, Array<uint32_t>& out, size_t max_digits) const;
    bool encode_word(const uint32_t* code_points, size_t count, Array<int32_t>& out);
    bool encode_bytes(std::string_view word, Array<int32_t>& out);
    bool encode_short(std::string_view word, Array<int32_t>& out);
    bool emit_symbol(uint32_t symbol, Array<int32_t>& out) const;
    bool remember_word(std::string_view word, const int32_t* tokens, size_t count);
    void forget_words();
    bool push_bigram(int32_t left, int32_t right);
    Bigram pop_bigram();
    bool encode_spm(std::string_view text, bool space_prefix, Array<int32_t>& out);
    bool spm_try_bigram(int32_t left, int32_t right);
    SpmBigram spm_pop();

    MappedFile m_hf_file;
    JsonDocument m_hf_json;
    Arena m_arena;
    Array<Token> m_tokens;
    Array<Symbol> m_symbols;
    StringMap m_symbol_index;
    IntMap m_merges; // (left symbol, right symbol) to rank
    Array<uint32_t> m_merged; // rank to merged symbol
    Array<int32_t> m_special;
    Array<char> m_pieces; // decoded text of every token, back to back

    uint32_t m_byte_symbol[256] = {};
    int32_t m_byte_token[256] = {};
    uint8_t m_ascii_classes[128] = {};

    PreTokenizer m_pre = PreTokenizer::Gpt2;
    Model m_model = Model::Bpe;
    Array<float> m_scores;
    int32_t m_unk = TOKEN_NONE;
    Array<char> m_spm_text;
    Array<SpmSymbol> m_spm_symbols;
    Array<SpmBigram> m_spm_heap;
    bool m_ignore_merges = false;
    Array<char> m_mapped_word;
    bool m_clean_spaces = false;
    bool m_add_space_prefix = false;
    bool m_add_bos = false;
    bool m_add_eos = false;
    int32_t m_bos = TOKEN_NONE;
    int32_t m_eos = TOKEN_NONE;

    Array<Fragment> m_fragments;
    Array<Fragment> m_fragments_next;
    Array<uint32_t> m_code_points;
    Array<uint8_t> m_classes;
    Array<uint32_t> m_words;
    Array<uint32_t> m_words_next;
    Array<Node> m_nodes;
    Array<Bigram> m_heap;
    Array<char> m_scratch;
    Array<char> m_word_bytes;

    StringMap m_word_cache;
    Array<CachedWord> m_cached_words;
    Array<int32_t> m_cached_tokens;
    Arena m_cache_arena;
};

}
