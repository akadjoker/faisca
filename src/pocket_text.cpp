#include "faisca/pocket_text.h"

#include "faisca/unicode.h"

#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

const char METASPACE[] = "\xe2\x96\x81";
constexpr size_t METASPACE_LENGTH = 3;

// "<0xXX>" pieces
bool byte_piece(const char* p, size_t length, unsigned int& byte)
{
    if (length != 6 || p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>')
    {
        return false;
    }
    byte = 0;
    for (size_t i = 3; i < 5; ++i)
    {
        const char c = p[i];
        unsigned int v = 0;
        if (c >= '0' && c <= '9')
        {
            v = static_cast<unsigned int>(c - '0');
        }
        else if (c >= 'A' && c <= 'F')
        {
            v = static_cast<unsigned int>(c - 'A' + 10);
        }
        else if (c >= 'a' && c <= 'f')
        {
            v = static_cast<unsigned int>(c - 'a' + 10);
        }
        else
        {
            return false;
        }
        byte = byte * 16 + v;
    }
    return true;
}

bool is_space(uint32_t c)
{
    return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

bool in_set(uint32_t c, const uint32_t* set)
{
    for (; *set != 0; ++set)
    {
        if (*set == c)
        {
            return true;
        }
    }
    return false;
}

const uint32_t TERMINAL[] = {'.', '!', '?', 0x2026, 0};
const uint32_t WEAK[] = {',', ';', ':', '-', 0x2013, 0x2014, 0};
const uint32_t CLOSERS[] = {'"', '\'', 0x201d, 0x2019, ')', ']', 0xbb, 0};
const uint32_t SUBS[] = {',', ';', ':', 0};

uint32_t to_upper(uint32_t c)
{
    if (c >= 'a' && c <= 'z')
    {
        return c - 32;
    }
    if (c >= 0xe0 && c <= 0xfe && c != 0xf7)
    {
        return c - 32;
    }
    if (c == 0xff)
    {
        return 0x178;
    }
    if (c >= 0x100 && c <= 0x17f)
    {
        const bool odd_lower = (c >= 0x100 && c <= 0x137) || (c >= 0x14a && c <= 0x177);
        const bool even_lower = (c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17e);
        if (odd_lower && (c & 1))
        {
            return c - 1;
        }
        if (even_lower && !(c & 1))
        {
            return c - 1;
        }
        return c;
    }
    if (c >= 0x3b1 && c <= 0x3c9 && c != 0x3c2)
    {
        return c - 32;
    }
    if (c == 0x3c2)
    {
        return 0x3a3;
    }
    if (c >= 0x430 && c <= 0x44f)
    {
        return c - 32;
    }
    if (c >= 0x450 && c <= 0x45f)
    {
        return c - 80;
    }
    return c;
}

bool decode_points(const char* text, size_t length, Array<uint32_t>& out)
{
    out.clear();
    size_t offset = 0;
    while (offset < length)
    {
        const uint32_t cp = utf8_decode(text, length, offset);
        if (!out.push_back(cp))
        {
            return false;
        }
    }
    return true;
}

bool encode_points(const uint32_t* points, size_t count, Array<char>& out)
{
    for (size_t i = 0; i < count; ++i)
    {
        char buffer[4];
        const size_t n = utf8_encode(points[i], buffer);
        if (!out.append(buffer, n))
        {
            return false;
        }
    }
    return true;
}

size_t count_words(const uint32_t* s, size_t n)
{
    size_t words = 0;
    bool in_word = false;
    for (size_t i = 0; i < n; ++i)
    {
        if (is_space(s[i]))
        {
            in_word = false;
        }
        else if (!in_word)
        {
            in_word = true;
            words++;
        }
    }
    return words;
}

void strip(const uint32_t* s, size_t n, size_t& begin, size_t& end)
{
    begin = 0;
    end = n;
    while (begin < end && is_space(s[begin]))
    {
        begin++;
    }
    while (end > begin && is_space(s[end - 1]))
    {
        end--;
    }
}

bool ascii_digit(char c)
{
    return c >= '0' && c <= '9';
}

}

bool PocketTokenizer::load(const GgufFile& file, Error& error)
{
    const GgufArray* tokens = file.get_array("tokenizer.ggml.tokens");
    const GgufArray* scores = file.get_array("tokenizer.ggml.scores");
    if (tokens == nullptr || tokens->elem_type != GgufType::String || scores == nullptr ||
        scores->elem_type != GgufType::Float32 || scores->count != tokens->count)
    {
        error.set("tokenizer.ggml.tokens or tokenizer.ggml.scores is missing");
        return false;
    }
    Array<std::string_view> texts;
    if (!tokens->strings_to(texts) || !scores->copy_to(m_scores))
    {
        error.set("out of memory while reading the vocabulary");
        return false;
    }
    const size_t count = texts.size();
    for (int32_t& id : m_byte_ids)
    {
        id = -1;
    }
    m_bytes.clear();
    m_offsets.clear();
    m_max_length = 0;
    float min_score = 0.0f;
    if (!m_offsets.reserve(count + 1) || !m_ids.reserve(count))
    {
        error.set("out of memory");
        return false;
    }
    m_offsets.push_back_unchecked(0);
    for (size_t i = 0; i < count; ++i)
    {
        const std::string_view piece = texts[i];
        if (!m_bytes.append(piece.data(), piece.size()))
        {
            error.set("out of memory");
            return false;
        }
        m_offsets.push_back_unchecked(static_cast<uint32_t>(m_bytes.size()));
        if (m_scores[i] < min_score)
        {
            min_score = m_scores[i];
        }
        unsigned int byte = 0;
        if (byte_piece(piece.data(), piece.size(), byte) && byte < 256)
        {
            m_byte_ids[byte] = static_cast<int32_t>(i);
            continue;
        }
        if (piece == "<unk>" || piece == "<s>" || piece == "</s>" || piece == "<pad>")
        {
            continue;
        }
        if (piece.size() > m_max_length)
        {
            m_max_length = piece.size();
        }
    }
    // keys point into m_bytes, which no longer moves
    for (size_t i = 0; i < count; ++i)
    {
        const std::string_view piece(m_bytes.data() + m_offsets[i], m_offsets[i + 1] - m_offsets[i]);
        unsigned int byte = 0;
        if (byte_piece(piece.data(), piece.size(), byte) || piece == "<unk>" ||
            piece == "<s>" || piece == "</s>" || piece == "<pad>")
        {
            continue;
        }
        const StringMap::Insert insert = m_ids.insert(piece, static_cast<uint32_t>(i));
        if (insert == StringMap::Insert::Failed)
        {
            error.set("out of memory");
            return false;
        }
    }
    m_unk_score = min_score - 10.0f;
    return true;
}

const char* PocketTokenizer::piece(int32_t id, size_t& length) const
{
    if (id < 0 || static_cast<size_t>(id) >= size())
    {
        length = 0;
        return "";
    }
    length = m_offsets[static_cast<size_t>(id) + 1] - m_offsets[static_cast<size_t>(id)];
    return m_bytes.data() + m_offsets[static_cast<size_t>(id)];
}

namespace
{

size_t utf8_length(unsigned char c)
{
    return c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 1;
}

}

bool PocketTokenizer::encode_segment(const char* s, size_t n, Array<int32_t>& out) const
{
    Array<float> best;
    Array<int32_t> from;
    Array<int32_t> id;
    if (!best.resize_uninitialized(n + 1) || !from.resize_uninitialized(n + 1) || !id.resize_uninitialized(n + 1))
    {
        return false;
    }
    for (size_t i = 0; i <= n; ++i)
    {
        best[i] = -1e30f;
        from[i] = -1;
        id[i] = -1;
    }
    best[0] = 0.0f;
    for (size_t i = 0; i < n; i += utf8_length(static_cast<unsigned char>(s[i])))
    {
        if (best[i] <= -1e30f)
        {
            continue;
        }
        const size_t cl = utf8_length(static_cast<unsigned char>(s[i]));
        bool single = false;
        for (size_t j = i + cl; j <= n && j - i <= m_max_length;)
        {
            uint32_t found = 0;
            if (m_ids.find(std::string_view(s + i, j - i), found))
            {
                const float score = best[i] + m_scores[found];
                if (from[j] < 0 || score > best[j])
                {
                    best[j] = score;
                    from[j] = static_cast<int32_t>(i);
                    id[j] = static_cast<int32_t>(found);
                }
                if (j - i == cl)
                {
                    single = true;
                }
            }
            if (j == n)
            {
                break;
            }
            j += utf8_length(static_cast<unsigned char>(s[j]));
        }
        if (!single && i + cl <= n)
        {
            const float score = best[i] + m_unk_score;
            if (from[i + cl] < 0 || score > best[i + cl])
            {
                best[i + cl] = score;
                from[i + cl] = static_cast<int32_t>(i);
                id[i + cl] = -1;
            }
        }
    }
    // the path backwards, then emitted forwards
    Array<uint32_t> ends;
    for (size_t j = n; j > 0; j = static_cast<size_t>(from[j]))
    {
        if (from[j] < 0 || !ends.push_back(static_cast<uint32_t>(j)))
        {
            return false;
        }
    }
    for (size_t k = ends.size(); k > 0; --k)
    {
        const size_t j = ends[k - 1];
        const size_t i = static_cast<size_t>(from[j]);
        if (id[j] >= 0)
        {
            if (!out.push_back(id[j]))
            {
                return false;
            }
            continue;
        }
        for (size_t b = i; b < j; ++b)
        {
            if (!out.push_back(m_byte_ids[static_cast<unsigned char>(s[b])]))
            {
                return false;
            }
        }
    }
    return true;
}

bool PocketTokenizer::encode(const char* text, size_t length, Array<int32_t>& out) const
{
    Array<char> norm;
    if (!norm.append(METASPACE, METASPACE_LENGTH))
    {
        return false;
    }
    for (size_t i = 0; i < length; ++i)
    {
        if (text[i] == ' ')
        {
            if (!norm.append(METASPACE, METASPACE_LENGTH))
            {
                return false;
            }
        }
        else if (!norm.push_back(text[i]))
        {
            return false;
        }
    }
    size_t start = 0;
    while (start < norm.size())
    {
        size_t next = start + METASPACE_LENGTH;
        while (next + METASPACE_LENGTH <= norm.size() && std::memcmp(norm.data() + next, METASPACE, METASPACE_LENGTH) != 0)
        {
            next++;
        }
        if (next + METASPACE_LENGTH > norm.size())
        {
            next = norm.size();
        }
        if (!encode_segment(norm.data() + start, next - start, out))
        {
            return false;
        }
        start = next;
    }
    return true;
}

bool PocketTokenizer::decode(const int32_t* ids, size_t count, Array<char>& out) const
{
    Array<char> bytes;
    for (size_t i = 0; i < count; ++i)
    {
        size_t length = 0;
        const char* p = piece(ids[i], length);
        unsigned int byte = 0;
        if (byte_piece(p, length, byte))
        {
            if (!bytes.push_back(static_cast<char>(byte)))
            {
                return false;
            }
            continue;
        }
        if (!out.append(bytes.data(), bytes.size()))
        {
            return false;
        }
        bytes.clear();
        bool first_space_dropped = i != 0;
        for (size_t k = 0; k < length;)
        {
            if (k + METASPACE_LENGTH <= length && std::memcmp(p + k, METASPACE, METASPACE_LENGTH) == 0)
            {
                if (!first_space_dropped)
                {
                    first_space_dropped = true;
                }
                else if (!out.push_back(' '))
                {
                    return false;
                }
                k += METASPACE_LENGTH;
            }
            else
            {
                if (!out.push_back(p[k]))
                {
                    return false;
                }
                first_space_dropped = true;
                k++;
            }
        }
    }
    return out.append(bytes.data(), bytes.size());
}

bool pocket_prepare_text(const PocketTextRules& rules, const char* text, size_t length, Array<char>& out, int& guess,
                         Error& error)
{
    Array<uint32_t> points;
    if (!decode_points(text, length, points))
    {
        error.set("out of memory");
        return false;
    }
    size_t begin = 0;
    size_t end = 0;
    strip(points.data(), points.size(), begin, end);
    Array<uint32_t> work;
    if (!work.append(points.data() + begin, end - begin))
    {
        error.set("out of memory");
        return false;
    }
    if (rules.replace_from.size() != 0)
    {
        Array<uint32_t> replaced;
        for (size_t i = 0; i < work.size(); ++i)
        {
            bool hit = false;
            for (size_t k = 0; k < rules.replace_from.size(); ++k)
            {
                if (rules.replace_from[k] == work[i])
                {
                    const uint32_t a = rules.replace_to_offsets[k];
                    const uint32_t b = rules.replace_to_offsets[k + 1];
                    if (!replaced.append(rules.replace_to.data() + a, b - a))
                    {
                        error.set("out of memory");
                        return false;
                    }
                    hit = true;
                    break;
                }
            }
            if (!hit && !replaced.push_back(work[i]))
            {
                error.set("out of memory");
                return false;
            }
        }
        // " ".join(split()): whitespace runs collapsed, then a sentence mark followed by , ; : keeps the mark
        Array<uint32_t> joined;
        bool pending_space = false;
        bool in_word = false;
        for (size_t i = 0; i < replaced.size(); ++i)
        {
            if (is_space(replaced[i]))
            {
                if (in_word)
                {
                    pending_space = true;
                }
                in_word = false;
                continue;
            }
            if (pending_space && !joined.push_back(' '))
            {
                error.set("out of memory");
                return false;
            }
            pending_space = false;
            in_word = true;
            if (!joined.push_back(replaced[i]))
            {
                error.set("out of memory");
                return false;
            }
        }
        work.clear();
        for (size_t i = 0; i < joined.size(); ++i)
        {
            if (!work.push_back(joined[i]))
            {
                error.set("out of memory");
                return false;
            }
            if (in_set(joined[i], TERMINAL))
            {
                size_t j = i + 1;
                while (j < joined.size() && is_space(joined[j]))
                {
                    j++;
                }
                if (j < joined.size() && in_set(joined[j], SUBS))
                {
                    i = j;
                }
            }
        }
    }
    if (work.size() == 0)
    {
        return false;
    }
    for (size_t i = 0; i < work.size(); ++i)
    {
        if (work[i] == '\n' || work[i] == '\r')
        {
            work[i] = ' ';
        }
    }
    // "  " -> " ", one pass left to right as str.replace
    {
        Array<uint32_t> squeezed;
        for (size_t i = 0; i < work.size(); ++i)
        {
            if (work[i] == ' ' && i + 1 < work.size() && work[i + 1] == ' ')
            {
                if (!squeezed.push_back(' '))
                {
                    error.set("out of memory");
                    return false;
                }
                i++;
                continue;
            }
            if (!squeezed.push_back(work[i]))
            {
                error.set("out of memory");
                return false;
            }
        }
        work.swap(squeezed);
    }
    if (rules.remove_semicolons)
    {
        for (size_t i = 0; i < work.size(); ++i)
        {
            if (work[i] == ';')
            {
                work[i] = ',';
            }
        }
    }
    guess = count_words(work.data(), work.size()) <= 4 ? 3 : 1;
    if (rules.capitalize_first_letter)
    {
        work[0] = to_upper(work[0]);
    }
    if (rules.append_terminal_punctuation)
    {
        size_t e = work.size();
        while (e > 0 && (work[e - 1] == ' ' || in_set(work[e - 1], CLOSERS)))
        {
            e--;
        }
        if (e == 0 || !in_set(work[e - 1], TERMINAL))
        {
            // the closers after the core, stripped of spaces
            Array<uint32_t> closers;
            {
                size_t cb = 0;
                size_t ce = 0;
                strip(work.data() + e, work.size() - e, cb, ce);
                if (!closers.append(work.data() + e + cb, ce - cb))
                {
                    error.set("out of memory");
                    return false;
                }
            }
            if (e > 0 && in_set(work[e - 1], WEAK))
            {
                size_t core = e;
                while (core > 0 && (work[core - 1] == ' ' || in_set(work[core - 1], WEAK)))
                {
                    core--;
                }
                Array<uint32_t> fixed;
                if (!fixed.append(work.data(), core) || !fixed.push_back('.') || !fixed.append(closers.data(), closers.size()))
                {
                    error.set("out of memory");
                    return false;
                }
                work.swap(fixed);
            }
            else if (e > 0 && !work.push_back('.'))
            {
                error.set("out of memory");
                return false;
            }
        }
    }
    if (rules.pad_with_spaces && count_words(work.data(), work.size()) < 5)
    {
        Array<uint32_t> padded;
        for (int i = 0; i < 8; ++i)
        {
            if (!padded.push_back(' '))
            {
                error.set("out of memory");
                return false;
            }
        }
        if (!padded.append(work.data(), work.size()))
        {
            error.set("out of memory");
            return false;
        }
        work.swap(padded);
    }
    out.clear();
    if (!encode_points(work.data(), work.size(), out))
    {
        error.set("out of memory");
        return false;
    }
    return true;
}

namespace
{

bool trim_bytes(const char* s, size_t n, Array<char>& out)
{
    Array<uint32_t> points;
    if (!decode_points(s, n, points))
    {
        return false;
    }
    size_t begin = 0;
    size_t end = 0;
    strip(points.data(), points.size(), begin, end);
    out.clear();
    return encode_points(points.data() + begin, end - begin, out);
}

bool is_bound(int32_t id, const int32_t* bounds, size_t n_bounds)
{
    for (size_t b = 0; b < n_bounds; ++b)
    {
        if (bounds[b] == id)
        {
            return true;
        }
    }
    return false;
}

// cut points after runs of boundary tokens: [0, ..., n]
bool boundaries(const PocketTokenizer& tok, const int32_t* ids, size_t n, const int32_t* bounds, size_t n_bounds,
                bool skip_decimal, Array<uint32_t>& out)
{
    out.clear();
    if (!out.push_back(0))
    {
        return false;
    }
    bool after = false;
    for (size_t i = 0; i < n; ++i)
    {
        if (is_bound(ids[i], bounds, n_bounds))
        {
            after = true;
            continue;
        }
        if (after)
        {
            bool decimal = false;
            if (skip_decimal)
            {
                Array<char> pre;
                Array<char> suf;
                if (!tok.decode(ids, i, pre) || !tok.decode(ids + i, n - i, suf))
                {
                    return false;
                }
                decimal = pre.size() >= 2 && pre[pre.size() - 1] == '.' && ascii_digit(pre[pre.size() - 2]) &&
                          suf.size() != 0 && ascii_digit(suf[0]);
            }
            if (!decimal && !out.push_back(static_cast<uint32_t>(i)))
            {
                return false;
            }
        }
        after = false;
    }
    return out.push_back(static_cast<uint32_t>(n));
}

struct Segment
{
    uint32_t n_tokens;
    uint32_t offset;
    uint32_t length;
};

bool push_segment(Array<Segment>& segments, Array<char>& texts, uint32_t n_tokens, const char* text, size_t length)
{
    Segment segment{n_tokens, static_cast<uint32_t>(texts.size()), static_cast<uint32_t>(length)};
    return texts.append(text, length) && segments.push_back(segment);
}

}

bool pocket_chunk_text(const PocketTextRules& rules, const PocketTokenizer& tokenizer, const char* text, size_t length,
                       size_t max_tokens, Array<char>& chunks, Array<uint32_t>& offsets, Error& error)
{
    chunks.clear();
    offsets.clear();
    if (!offsets.push_back(0))
    {
        error.set("out of memory");
        return false;
    }
    Array<char> prepared;
    int guess = 0;
    if (!pocket_prepare_text(rules, text, length, prepared, guess, error))
    {
        return error.c_str()[0] == '\0' ? true : false;
    }
    Array<char> trimmed;
    if (!trim_bytes(prepared.data(), prepared.size(), trimmed))
    {
        error.set("out of memory");
        return false;
    }
    Array<int32_t> ids;
    Array<int32_t> eos;
    Array<int32_t> sub;
    if (!tokenizer.encode(trimmed.data(), trimmed.size(), ids) || !tokenizer.encode(".!...?", 6, eos) ||
        !tokenizer.encode(",;:", 3, sub) || eos.size() < 2 || sub.size() < 2)
    {
        error.set("out of memory");
        return false;
    }
    // the first token of each is the leading metaspace piece
    const int32_t* eos_bounds = eos.data() + 1;
    const size_t n_eos = eos.size() - 1;
    const int32_t* sub_bounds = sub.data() + 1;
    const size_t n_sub = sub.size() - 1;

    Array<Segment> segments;
    Array<char> texts;
    Array<uint32_t> b;
    if (!boundaries(tokenizer, ids.data(), ids.size(), eos_bounds, n_eos, true, b))
    {
        error.set("out of memory");
        return false;
    }
    for (size_t i = 0; i + 1 < b.size(); ++i)
    {
        const uint32_t n = b[i + 1] - b[i];
        Array<char> s;
        if (!tokenizer.decode(ids.data() + b[i], n, s))
        {
            error.set("out of memory");
            return false;
        }
        if (n <= max_tokens)
        {
            if (!push_segment(segments, texts, n, s.data(), s.size()))
            {
                error.set("out of memory");
                return false;
            }
            continue;
        }
        Array<char> st;
        Array<int32_t> sid;
        Array<uint32_t> sb;
        if (!trim_bytes(s.data(), s.size(), st) || !tokenizer.encode(st.data(), st.size(), sid) ||
            !boundaries(tokenizer, sid.data(), sid.size(), sub_bounds, n_sub, false, sb))
        {
            error.set("out of memory");
            return false;
        }
        if (sb.size() > 2)
        {
            for (size_t k = 0; k + 1 < sb.size(); ++k)
            {
                Array<char> piece;
                if (!tokenizer.decode(sid.data() + sb[k], sb[k + 1] - sb[k], piece) ||
                    !push_segment(segments, texts, sb[k + 1] - sb[k], piece.data(), piece.size()))
                {
                    error.set("out of memory");
                    return false;
                }
            }
        }
        else if (!push_segment(segments, texts, n, s.data(), s.size()))
        {
            error.set("out of memory");
            return false;
        }
    }

    Array<char> current;
    uint32_t current_n = 0;
    const auto flush = [&]() -> bool
    {
        Array<char> t;
        if (!trim_bytes(current.data(), current.size(), t) || !chunks.append(t.data(), t.size()) ||
            !offsets.push_back(static_cast<uint32_t>(chunks.size())))
        {
            return false;
        }
        current.clear();
        current_n = 0;
        return true;
    };
    for (size_t i = 0; i < segments.size(); ++i)
    {
        const Segment& sg = segments[i];
        const char* s = texts.data() + sg.offset;
        if (current.size() == 0)
        {
            if (!current.append(s, sg.length))
            {
                error.set("out of memory");
                return false;
            }
            current_n = sg.n_tokens;
            continue;
        }
        if (current_n + sg.n_tokens > max_tokens)
        {
            if (!flush() || !current.append(s, sg.length))
            {
                error.set("out of memory");
                return false;
            }
            current_n = sg.n_tokens;
        }
        else
        {
            if (!current.push_back(' ') || !current.append(s, sg.length))
            {
                error.set("out of memory");
                return false;
            }
            current_n += sg.n_tokens;
        }
    }
    if (current.size() != 0 && !flush())
    {
        error.set("out of memory");
        return false;
    }
    return true;
}

}
