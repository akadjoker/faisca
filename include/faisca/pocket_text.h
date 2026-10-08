#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/gguf.h"
#include "faisca/string_map.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// the text rules of a Pocket TTS pack (pocket.text.* keys)
struct PocketTextRules
{
    bool pad_with_spaces = false;
    bool remove_semicolons = false;
    bool append_terminal_punctuation = true;
    bool capitalize_first_letter = true;
    Array<uint32_t> replace_from; // code points
    Array<uint32_t> replace_to_offsets; // into replace_to, one more than replace_from
    Array<uint32_t> replace_to;
};

// SentencePiece unigram model of the pack: U+2581 for spaces, Viterbi over the summed piece
// scores per space delimited segment, byte pieces <0xXX> for characters no piece covers
class PocketTokenizer
{
public:
    PocketTokenizer() = default;

    PocketTokenizer(const PocketTokenizer&) = delete;
    PocketTokenizer& operator=(const PocketTokenizer&) = delete;

    bool load(const GgufFile& file, Error& error);

    size_t size() const
    {
        return m_offsets.size() == 0 ? 0 : m_offsets.size() - 1;
    }

    bool encode(const char* text, size_t length, Array<int32_t>& out) const;

    // ids [begin, end) back to text (the leading space of the first piece dropped, byte
    // pieces reassembled); appended to out
    bool decode(const int32_t* ids, size_t count, Array<char>& out) const;

    const char* piece(int32_t id, size_t& length) const;

private:
    bool encode_segment(const char* s, size_t n, Array<int32_t>& out) const;

    Array<char> m_bytes;
    Array<uint32_t> m_offsets;
    Array<float> m_scores;
    StringMap m_ids;
    int32_t m_byte_ids[256];
    size_t m_max_length = 0;
    float m_unk_score = 0.0f;
};

// text_chunking.py of the reference: the prepared text, the frames after EOS guess (3 for
// at most 4 words, else 1); false on empty text
bool pocket_prepare_text(const PocketTextRules& rules, const char* text, size_t length, Array<char>& out, int& guess,
                         Error& error);

// the prepared text cut into chunks of at most max_tokens tokens: chunk i is
// chunks[offsets[i], offsets[i + 1]) without a terminator
bool pocket_chunk_text(const PocketTextRules& rules, const PocketTokenizer& tokenizer, const char* text, size_t length,
                       size_t max_tokens, Array<char>& chunks, Array<uint32_t>& offsets, Error& error);

}
