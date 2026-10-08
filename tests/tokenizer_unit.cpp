#include "faisca/tokenizer.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace faisca;

namespace
{

int g_checks = 0;
int g_failures = 0;

#define CHECK(condition)                                                              \
    do                                                                                \
    {                                                                                 \
        ++g_checks;                                                                   \
        if (!(condition))                                                             \
        {                                                                             \
            ++g_failures;                                                             \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        }                                                                             \
    } while (false)

constexpr int32_t NORMAL = 1;
constexpr int32_t UNKNOWN = 2;
constexpr int32_t CONTROL = 3;
constexpr int32_t USER_DEFINED = 4;
constexpr int32_t UNUSED = 5;
constexpr int32_t BYTE = 6;

// ids 0 to 255 are the byte tokens, in byte order
constexpr int32_t T_SPACE_H = 256;
constexpr int32_t T_EL = 257;
constexpr int32_t T_LO_FIRST = 258;
constexpr int32_t T_ELLO = 259;
constexpr int32_t T_SPACE_HELLO = 260;
constexpr int32_t T_HE = 261;
constexpr int32_t T_HELLO = 262;
constexpr int32_t T_SPACE_W = 263;
constexpr int32_t T_OR = 264;
constexpr int32_t T_LD = 265;
constexpr int32_t T_SPACE_WOR = 266;
constexpr int32_t T_SPACE_WORLD = 267;
constexpr int32_t T_12 = 268;
constexpr int32_t T_SPACE_1 = 269;
constexpr int32_t T_APOSTROPHE_T = 270;
constexpr int32_t T_TWO_SPACES = 271;
constexpr int32_t T_CJK = 272;
constexpr int32_t T_EOT = 273;
constexpr int32_t T_IM_START = 274;
constexpr int32_t T_TAG = 275;
constexpr int32_t T_TAG_X = 276;
constexpr int32_t T_UNK = 277;
constexpr int32_t T_UNUSED = 278;
constexpr int32_t T_BYTE_A = 279;
constexpr int32_t T_EMPTY = 280;
constexpr int32_t T_LO = 281;
constexpr int32_t VOCAB_SIZE = 282;

std::string utf8(uint32_t code_point)
{
    std::string out;
    if (code_point <= 0x7f)
    {
        out.push_back(static_cast<char>(code_point));
    }
    else if (code_point <= 0x7ff)
    {
        out.push_back(static_cast<char>(0xc0 | (code_point >> 6)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    }
    else
    {
        out.push_back(static_cast<char>(0xe0 | (code_point >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    }
    return out;
}

bool printable_byte(uint32_t byte)
{
    return (byte >= 0x21 && byte <= 0x7e) || (byte >= 0xa1 && byte <= 0xac) || (byte >= 0xae && byte <= 0xff);
}

// the GPT-2 byte to code point table
std::string byte_text(uint32_t byte)
{
    if (printable_byte(byte))
    {
        return utf8(byte);
    }
    uint32_t next = 256;
    for (uint32_t b = 0; b < byte; ++b)
    {
        if (!printable_byte(b))
        {
            ++next;
        }
    }
    return utf8(next);
}

struct Spec
{
    std::string model = "gpt2";
    std::string pre = "gpt-2";
    bool with_model = true;
    bool with_pre = true;
    bool with_tokens = true;
    bool with_types = true;
    bool with_merges = true;
    bool tokens_as_ints = false;
    bool short_types = false;
    bool with_bos_eos_ids = true;
    bool add_bos = false;
    bool add_eos = false;
    bool add_space_prefix = false;
    size_t token_count = static_cast<size_t>(VOCAB_SIZE);
};

std::vector<std::pair<std::string, int32_t>> make_tokens()
{
    std::vector<std::pair<std::string, int32_t>> tokens;
    for (uint32_t byte = 0; byte < 256; ++byte)
    {
        tokens.emplace_back(byte_text(byte), NORMAL);
    }
    const char* space = "\xc4\xa0";
    tokens.emplace_back(std::string(space) + "h", NORMAL);
    tokens.emplace_back("el", NORMAL);
    tokens.emplace_back("lo", NORMAL);
    tokens.emplace_back("ello", NORMAL);
    tokens.emplace_back(std::string(space) + "hello", NORMAL);
    tokens.emplace_back("he", NORMAL);
    tokens.emplace_back("hello", NORMAL);
    tokens.emplace_back(std::string(space) + "w", NORMAL);
    tokens.emplace_back("or", NORMAL);
    tokens.emplace_back("ld", NORMAL);
    tokens.emplace_back(std::string(space) + "wor", NORMAL);
    tokens.emplace_back(std::string(space) + "world", NORMAL);
    tokens.emplace_back("12", NORMAL);
    tokens.emplace_back(std::string(space) + "1", NORMAL);
    tokens.emplace_back("'t", NORMAL);
    tokens.emplace_back(std::string(space) + space, NORMAL);
    tokens.emplace_back("\xe6\x97\xa5", NORMAL);
    tokens.emplace_back("<|endoftext|>", CONTROL);
    tokens.emplace_back("<|im_start|>", CONTROL);
    tokens.emplace_back("<tag>", USER_DEFINED);
    tokens.emplace_back("<tag>x", USER_DEFINED);
    tokens.emplace_back("<unk>", UNKNOWN);
    tokens.emplace_back("<unused>", UNUSED);
    tokens.emplace_back("<0x41>", BYTE);
    tokens.emplace_back("", NORMAL);
    tokens.emplace_back("lo", NORMAL);
    return tokens;
}

std::vector<std::string> make_merges()
{
    const std::string space = "\xc4\xa0";
    return {
        space + " h", "e l", "l o", "el lo", space + "h ello", "h e", "h ello", space + " w", "o r", "l d",
        space + "w or", space + "wor ld", "1 2", space + " 1", "' t", space + " " + space, space + " x",
        "no_space_here", "",
    };
}

class Image
{
public:
    template <typename T>
    Image& put(T value)
    {
        const size_t at = bytes.size();
        bytes.resize(at + sizeof(T));
        std::memcpy(bytes.data() + at, &value, sizeof(T));
        return *this;
    }

    Image& str(std::string_view text)
    {
        put<uint64_t>(text.size());
        bytes.insert(bytes.end(), text.begin(), text.end());
        return *this;
    }

    Image& key(std::string_view name, GgufType type)
    {
        str(name);
        put<uint32_t>(static_cast<uint32_t>(type));
        return *this;
    }

    Image& array(GgufType elem_type, uint64_t count)
    {
        put<uint32_t>(static_cast<uint32_t>(elem_type));
        put<uint64_t>(count);
        return *this;
    }

    std::vector<uint8_t> bytes;
};

Image make_image(const Spec& spec)
{
    const auto tokens = make_tokens();
    const auto merges = make_merges();

    uint64_t kv_count = 1;
    kv_count += spec.with_model ? 1 : 0;
    kv_count += spec.with_pre ? 1 : 0;
    kv_count += spec.with_tokens ? 1 : 0;
    kv_count += spec.with_types ? 1 : 0;
    kv_count += spec.with_merges ? 1 : 0;
    kv_count += spec.with_bos_eos_ids ? 2 : 0;
    kv_count += spec.add_bos ? 1 : 0;
    kv_count += spec.add_eos ? 1 : 0;
    kv_count += spec.add_space_prefix ? 1 : 0;

    Image image;
    image.bytes.insert(image.bytes.end(), {'G', 'G', 'U', 'F'});
    image.put<uint32_t>(3);
    image.put<uint64_t>(0);
    image.put<uint64_t>(kv_count);

    image.key("general.architecture", GgufType::String).str("llama");
    if (spec.with_model)
    {
        image.key("tokenizer.ggml.model", GgufType::String).str(spec.model);
    }
    if (spec.with_pre)
    {
        image.key("tokenizer.ggml.pre", GgufType::String).str(spec.pre);
    }
    if (spec.with_tokens)
    {
        if (spec.tokens_as_ints)
        {
            image.key("tokenizer.ggml.tokens", GgufType::Array).array(GgufType::Int32, 1).put<int32_t>(0);
        }
        else
        {
            image.key("tokenizer.ggml.tokens", GgufType::Array).array(GgufType::String, spec.token_count);
            for (size_t i = 0; i < spec.token_count; ++i)
            {
                image.str(tokens[i].first);
            }
        }
    }
    if (spec.with_types)
    {
        const size_t count = spec.short_types ? spec.token_count - 1 : spec.token_count;
        image.key("tokenizer.ggml.token_type", GgufType::Array).array(GgufType::Int32, count);
        for (size_t i = 0; i < count; ++i)
        {
            image.put<int32_t>(tokens[i].second);
        }
    }
    if (spec.with_merges)
    {
        image.key("tokenizer.ggml.merges", GgufType::Array).array(GgufType::String, merges.size());
        for (const std::string& merge : merges)
        {
            image.str(merge);
        }
    }
    if (spec.with_bos_eos_ids)
    {
        image.key("tokenizer.ggml.bos_token_id", GgufType::UInt32).put<uint32_t>(static_cast<uint32_t>(T_EOT));
        image.key("tokenizer.ggml.eos_token_id", GgufType::UInt32).put<uint32_t>(static_cast<uint32_t>(T_EOT));
    }
    if (spec.add_bos)
    {
        image.key("tokenizer.ggml.add_bos_token", GgufType::Bool).put<uint8_t>(1);
    }
    if (spec.add_eos)
    {
        image.key("tokenizer.ggml.add_eos_token", GgufType::Bool).put<uint8_t>(1);
    }
    if (spec.add_space_prefix)
    {
        image.key("tokenizer.ggml.add_space_prefix", GgufType::Bool).put<uint8_t>(1);
    }
    return image;
}

struct Loaded
{
    Image image;
    GgufFile file;
    Tokenizer tokenizer;
    Error error;

    explicit Loaded(const Spec& spec) : image(make_image(spec))
    {
        if (!file.parse(image.bytes.data(), image.bytes.size(), error) || !tokenizer.load(file, error))
        {
            std::fprintf(stderr, "  load failed: %s\n", error.c_str());
        }
    }
};

std::vector<int32_t> encode(Tokenizer& tokenizer, std::string_view text, bool add_special, bool parse_special)
{
    Array<int32_t> out;
    if (!tokenizer.encode(text, add_special, parse_special, out))
    {
        return {-2};
    }
    return std::vector<int32_t>(out.begin(), out.end());
}

std::string decode(const Tokenizer& tokenizer, const std::vector<int32_t>& tokens, bool remove_special,
                   bool unparse_special)
{
    Array<char> out;
    if (!tokenizer.decode(tokens.data(), tokens.size(), remove_special, unparse_special, out))
    {
        return "<decode failed>";
    }
    return std::string(out.begin(), out.end());
}

void print_tokens(const char* label, const std::vector<int32_t>& tokens)
{
    std::fprintf(stderr, "    %s:", label);
    for (const int32_t token : tokens)
    {
        std::fprintf(stderr, " %d", token);
    }
    std::fprintf(stderr, "\n");
}

void check_encode(Tokenizer& tokenizer, std::string_view text, bool add_special, bool parse_special,
                  const std::vector<int32_t>& expected, int line)
{
    const std::vector<int32_t> actual = encode(tokenizer, text, add_special, parse_special);
    ++g_checks;
    if (actual != expected)
    {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: encode mismatch for '%.*s'\n", __FILE__, line, static_cast<int>(text.size()),
                     text.data());
        print_tokens("expected", expected);
        print_tokens("actual  ", actual);
    }
}

#define CHECK_ENCODE(tokenizer, text, add_special, parse_special, ...) \
    check_encode(tokenizer, text, add_special, parse_special, __VA_ARGS__, __LINE__)

void check_decode(const Tokenizer& tokenizer, int line, const std::vector<int32_t>& tokens, bool remove_special,
                  bool unparse_special, std::string_view expected)
{
    const std::string actual = decode(tokenizer, tokens, remove_special, unparse_special);
    ++g_checks;
    if (actual != expected)
    {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: decode mismatch\n", __FILE__, line);
        std::fprintf(stderr, "    expected: '%.*s'\n", static_cast<int>(expected.size()), expected.data());
        std::fprintf(stderr, "    actual:   '%s'\n", actual.c_str());
    }
}

#define CHECK_DECODE(tokenizer, ...) check_decode(tokenizer, __LINE__, __VA_ARGS__)

void check_rejected(const Spec& spec, const char* expected, int line)
{
    const Image image = make_image(spec);
    Error error;
    GgufFile file;
    ++g_checks;
    if (!file.parse(image.bytes.data(), image.bytes.size(), error))
    {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: the image itself was rejected: %s\n", __FILE__, line, error.c_str());
        return;
    }

    Tokenizer tokenizer;
    if (tokenizer.load(file, error))
    {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: vocabulary was accepted, expected an error with '%s'\n", __FILE__, line, expected);
        return;
    }
    if (std::strstr(error.c_str(), expected) == nullptr)
    {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: error '%s' does not mention '%s'\n", __FILE__, line, error.c_str(), expected);
    }
    CHECK(!tokenizer.is_loaded());
    CHECK(tokenizer.vocab_size() == 0);
}

#define CHECK_REJECTED(spec, expected) check_rejected(spec, expected, __LINE__)

void test_load_errors()
{
    Spec spec;
    spec.with_model = false;
    CHECK_REJECTED(spec, "tokenizer.ggml.model is missing");

    spec = Spec();
    spec.model = "llama";
    CHECK_REJECTED(spec, "not supported");

    spec = Spec();
    spec.with_pre = false;
    CHECK_REJECTED(spec, "tokenizer.ggml.pre is missing");

    spec = Spec();
    spec.pre = "deepseek-llm";
    CHECK_REJECTED(spec, "not supported yet");

    spec = Spec();
    spec.with_tokens = false;
    CHECK_REJECTED(spec, "tokenizer.ggml.tokens");

    spec = Spec();
    spec.tokens_as_ints = true;
    CHECK_REJECTED(spec, "tokenizer.ggml.tokens");

    spec = Spec();
    spec.short_types = true;
    CHECK_REJECTED(spec, "tokenizer.ggml.token_type");

    spec = Spec();
    spec.with_merges = false;
    CHECK_REJECTED(spec, "tokenizer.ggml.merges");
}

void test_vocabulary()
{
    Loaded loaded{Spec()};
    const Tokenizer& tokenizer = loaded.tokenizer;
    CHECK(tokenizer.is_loaded());
    CHECK(tokenizer.vocab_size() == static_cast<size_t>(VOCAB_SIZE));

    CHECK(tokenizer.token_text(T_HELLO) == "hello");
    CHECK(tokenizer.token_text(T_SPACE_HELLO) == "\xc4\xa0hello");
    CHECK(tokenizer.token_text(T_EMPTY) == "[EMPTY_280]");
    CHECK(tokenizer.token_text(0) == "\xc4\x80");
    CHECK(tokenizer.token_text('a') == "a");

    CHECK(tokenizer.token_type(T_HELLO) == TokenType::Normal);
    CHECK(tokenizer.token_type(T_EOT) == TokenType::Control);
    CHECK(tokenizer.token_type(T_TAG) == TokenType::UserDefined);
    CHECK(tokenizer.token_type(T_UNK) == TokenType::Unknown);
    CHECK(tokenizer.token_type(T_UNUSED) == TokenType::Unused);
    CHECK(tokenizer.token_type(T_BYTE_A) == TokenType::Byte);

    CHECK(tokenizer.find_token("hello") == T_HELLO);
    CHECK(tokenizer.find_token("<|endoftext|>") == T_EOT);
    CHECK(tokenizer.find_token("[EMPTY_280]") == T_EMPTY);
    CHECK(tokenizer.find_token("lo") == T_LO);
    CHECK(tokenizer.token_text(T_LO_FIRST) == "lo");
    CHECK(tokenizer.token_text(T_LO) == "lo");
    CHECK(tokenizer.find_token("") == TOKEN_NONE);
    CHECK(tokenizer.find_token("missing") == TOKEN_NONE);
    CHECK(tokenizer.find_token("\xc4\xa0x") == TOKEN_NONE);

    CHECK(tokenizer.bos() == T_EOT);
    CHECK(tokenizer.eos() == T_EOT);
    CHECK(!tokenizer.add_bos());
    CHECK(!tokenizer.add_eos());

    Spec spec;
    spec.with_bos_eos_ids = false;
    Loaded defaults{spec};
    CHECK(defaults.tokenizer.bos() == 11);
    CHECK(defaults.tokenizer.eos() == 11);

    spec.token_count = 8;
    Loaded tiny{spec};
    CHECK(tiny.tokenizer.is_loaded());
    CHECK(tiny.tokenizer.vocab_size() == 8);
    CHECK(tiny.tokenizer.bos() == TOKEN_NONE);
    CHECK(tiny.tokenizer.eos() == TOKEN_NONE);

    spec = Spec();
    spec.with_types = false;
    Loaded untyped{spec};
    CHECK(untyped.tokenizer.token_type(T_EOT) == TokenType::Normal);
    CHECK(untyped.tokenizer.token_type(T_BYTE_A) == TokenType::Normal);

    Spec flags;
    flags.add_bos = true;
    flags.add_eos = true;
    Loaded with_flags{flags};
    CHECK(with_flags.tokenizer.add_bos());
    CHECK(with_flags.tokenizer.add_eos());

    with_flags.tokenizer.release();
    CHECK(!with_flags.tokenizer.is_loaded());
    CHECK(with_flags.tokenizer.vocab_size() == 0);
    CHECK(with_flags.tokenizer.bos() == TOKEN_NONE);
    CHECK(!with_flags.tokenizer.add_bos());
    CHECK(with_flags.tokenizer.find_token("hello") == TOKEN_NONE);

    CHECK(with_flags.tokenizer.load(with_flags.file, with_flags.error));
    CHECK(with_flags.tokenizer.vocab_size() == static_cast<size_t>(VOCAB_SIZE));
    CHECK(with_flags.tokenizer.find_token("hello") == T_HELLO);
}

void test_encode_gpt2()
{
    Loaded loaded{Spec()};
    Tokenizer& tokenizer = loaded.tokenizer;

    CHECK_ENCODE(tokenizer, "", false, false, {});
    CHECK_ENCODE(tokenizer, "", true, true, {});
    CHECK_ENCODE(tokenizer, "hello", false, false, {T_HELLO});
    CHECK_ENCODE(tokenizer, " hello world", false, false, {T_SPACE_HELLO, T_SPACE_WORLD});
    CHECK_ENCODE(tokenizer, "hello  world", false, false, {T_HELLO, ' ', T_SPACE_WORLD});
    CHECK_ENCODE(tokenizer, "hello   ", false, false, {T_HELLO, T_TWO_SPACES, ' '});
    CHECK_ENCODE(tokenizer, "  ", false, false, {T_TWO_SPACES});
    CHECK_ENCODE(tokenizer, "don't", false, false, {'d', 'o', 'n', T_APOSTROPHE_T});
    CHECK_ENCODE(tokenizer, "lo", false, false, {T_LO});
    CHECK_ENCODE(tokenizer, "he", false, false, {T_HE});
    CHECK_ENCODE(tokenizer, "el", false, false, {T_EL});
    CHECK_ENCODE(tokenizer, "ello", false, false, {T_ELLO});
    CHECK_ENCODE(tokenizer, " h", false, false, {T_SPACE_H});
    CHECK_ENCODE(tokenizer, " w", false, false, {T_SPACE_W});
    CHECK_ENCODE(tokenizer, "or", false, false, {T_OR});
    CHECK_ENCODE(tokenizer, "ld", false, false, {T_LD});
    CHECK_ENCODE(tokenizer, " wor", false, false, {T_SPACE_WOR});
    CHECK_ENCODE(tokenizer, " worlds", false, false, {T_SPACE_WORLD, 's'});
    CHECK_ENCODE(tokenizer, "hello12", false, false, {T_HELLO, T_12});
    CHECK_ENCODE(tokenizer, " 12", false, false, {' ', T_12});
    CHECK_ENCODE(tokenizer, " 1", false, false, {T_SPACE_1});
    CHECK_ENCODE(tokenizer, "a.b", false, false, {'a', '.', 'b'});
    CHECK_ENCODE(tokenizer, "\n\n", false, false, {'\n', '\n'});

    // a merge whose result is not a token: only the bytes that are tokens survive
    CHECK_ENCODE(tokenizer, " x", false, false, {'x'});

    // malformed UTF-8 is replaced by U+FFFD before the byte encoding
    CHECK_ENCODE(tokenizer, "\xff", false, false, {0xef, 0xbf, 0xbd});
    CHECK_ENCODE(tokenizer, "\xc3\xa9", false, false, {0xc3, 0xa9});
    CHECK_ENCODE(tokenizer, "\xe6\x97\xa5", false, false, {0xe6, 0x97, 0xa5});

    // add_special does nothing without add_bos_token and add_eos_token
    CHECK_ENCODE(tokenizer, "hello", true, false, {T_HELLO});

    Array<int32_t> out;
    CHECK(out.push_back(-7));
    CHECK(tokenizer.encode("hello", false, false, out));
    CHECK(tokenizer.encode(" world", false, false, out));
    CHECK(out.size() == 3 && out[0] == -7 && out[1] == T_HELLO && out[2] == T_SPACE_WORLD);

    // words longer than 64 bytes take the heap based merge loop
    for (const size_t repeats : {12, 13, 20, 200})
    {
        std::string long_word;
        std::vector<int32_t> expected;
        for (size_t i = 0; i < repeats; ++i)
        {
            long_word += "hello";
            expected.push_back(T_HELLO);
        }
        CHECK_ENCODE(tokenizer, long_word, false, false, expected);

        std::string spaces(repeats * 5, ' ');
        expected.assign(repeats * 5 / 2, T_TWO_SPACES);
        if ((repeats * 5) % 2 == 1)
        {
            expected.push_back(' ');
        }
        CHECK_ENCODE(tokenizer, spaces, false, false, expected);

        std::string letters(repeats * 5, 'a');
        expected.assign(repeats * 5, 'a');
        CHECK_ENCODE(tokenizer, letters, false, false, expected);
    }

    std::string long_text;
    for (int i = 0; i < 2000; ++i)
    {
        long_text += " hello world";
    }
    const std::vector<int32_t> tokens = encode(tokenizer, long_text, false, false);
    CHECK(tokens.size() == 4000);
    bool alternating = tokens.size() == 4000;
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        alternating = alternating && tokens[i] == (i % 2 == 0 ? T_SPACE_HELLO : T_SPACE_WORLD);
    }
    CHECK(alternating);
}

void test_encode_smollm()
{
    Spec spec;
    spec.pre = "smollm";
    Loaded loaded{spec};
    Tokenizer& tokenizer = loaded.tokenizer;

    CHECK_ENCODE(tokenizer, "hello", false, false, {T_HELLO});
    CHECK_ENCODE(tokenizer, " hello world", false, false, {T_SPACE_HELLO, T_SPACE_WORLD});
    CHECK_ENCODE(tokenizer, "hello12", false, false, {T_HELLO, '1', '2'});
    CHECK_ENCODE(tokenizer, " 12", false, false, {' ', '1', '2'});
    CHECK_ENCODE(tokenizer, " 1", false, false, {' ', '1'});
    CHECK_ENCODE(tokenizer, "1 2", false, false, {'1', ' ', '2'});
    CHECK_ENCODE(tokenizer, "don't", false, false, {'d', 'o', 'n', T_APOSTROPHE_T});
}

void test_special_tokens()
{
    Loaded loaded{Spec()};
    Tokenizer& tokenizer = loaded.tokenizer;

    CHECK_ENCODE(tokenizer, "<|im_start|>hello<tag>x<unk>", false, true, {T_IM_START, T_HELLO, T_TAG_X, T_UNK});
    CHECK_ENCODE(tokenizer, "<|im_start|>hello<tag>x<unk>", false, false,
                 {'<', '|', 'i', 'm', '_', 's', 't', 'a', 'r', 't', '|', '>', T_HELLO, T_TAG_X, '<', 'u', 'n', 'k',
                  '>'});
    CHECK_ENCODE(tokenizer, "<tag><tag>x", false, true, {T_TAG, T_TAG_X});
    CHECK_ENCODE(tokenizer, "<tag><tag>x", false, false, {T_TAG, T_TAG_X});
    CHECK_ENCODE(tokenizer, "<tag>", false, false, {T_TAG});
    CHECK_ENCODE(tokenizer, "<tag", false, true, {'<', 't', 'a', 'g'});
    CHECK_ENCODE(tokenizer, "<|endoftext|><|endoftext|>", false, true, {T_EOT, T_EOT});
    CHECK_ENCODE(tokenizer, "hello <|endoftext|> hello", false, true, {T_HELLO, ' ', T_EOT, T_SPACE_HELLO});
    CHECK_ENCODE(tokenizer, "<unused>", false, true, {'<', 'u', 'n', 'u', 's', 'e', 'd', '>'});
    CHECK_ENCODE(tokenizer, "<0x41>", false, true, {'<', '0', 'x', '4', '1', '>'});
    CHECK_ENCODE(tokenizer, "[EMPTY_280]", false, true, {'[', 'E', 'M', 'P', 'T', 'Y', '_', '2', '8', '0', ']'});

    Spec spec;
    spec.add_bos = true;
    spec.add_eos = true;
    Loaded with_flags{spec};
    CHECK_ENCODE(with_flags.tokenizer, "hello", true, true, {T_EOT, T_HELLO, T_EOT});
    CHECK_ENCODE(with_flags.tokenizer, "hello", false, true, {T_HELLO});
    CHECK_ENCODE(with_flags.tokenizer, "", true, true, {T_EOT, T_EOT});
    CHECK_ENCODE(with_flags.tokenizer, "<|endoftext|>", true, true, {T_EOT, T_EOT, T_EOT});

    spec.with_bos_eos_ids = false;
    spec.token_count = 8;
    Loaded no_bos{spec};
    CHECK(no_bos.tokenizer.add_bos());
    CHECK_ENCODE(no_bos.tokenizer, "", true, true, {});
}

void test_decode()
{
    Loaded loaded{Spec()};
    const Tokenizer& tokenizer = loaded.tokenizer;

    CHECK_DECODE(tokenizer, {}, false, false, "");
    CHECK_DECODE(tokenizer, {T_SPACE_HELLO, T_SPACE_WORLD}, false, false, " hello world");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', T_SPACE_WORLD}, false, false, "hello  world");
    CHECK_DECODE(tokenizer, {0xef, 0xbf, 0xbd}, false, false, "\xef\xbf\xbd");
    CHECK_DECODE(tokenizer, {0, 1, 2}, false, false, std::string("\x00\x01\x02", 3));
    CHECK_DECODE(tokenizer, {T_BYTE_A}, false, false, "A");
    CHECK_DECODE(tokenizer, {T_CJK}, false, false, "[UNK_BYTE_0xe697a5\xe6\x97\xa5]");
    CHECK_DECODE(tokenizer, {T_EMPTY}, false, false, "[EMPTY_280]");
    CHECK_DECODE(tokenizer, {T_UNUSED}, false, true, "");
    CHECK_DECODE(tokenizer, {-1, 9999, T_HELLO, VOCAB_SIZE}, false, true, "hello");

    CHECK_DECODE(tokenizer, {T_EOT, T_HELLO, T_TAG, T_UNK}, false, false, "hello<tag>");
    CHECK_DECODE(tokenizer, {T_EOT, T_HELLO, T_TAG, T_UNK}, false, true, "<|endoftext|>hello<tag><unk>");

    // add_bos_token is not set, so remove_special leaves the ends alone
    CHECK_DECODE(tokenizer, {T_EOT, T_HELLO, T_EOT}, true, true, "<|endoftext|>hello<|endoftext|>");

    // gpt-2 cleans spaces before punctuation and contractions
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '.'}, false, false, "hello.");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '?', ' ', '!', ' ', ','}, false, false, "hello?!,");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '\'', 's'}, false, false, "hello's");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '\'', 'm'}, false, false, "hello'm");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '\'', 'r', 'e'}, false, false, "hello're");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '\'', 'v', 'e'}, false, false, "hello've");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '\'', 't'}, false, false, "hello 't");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '\'', 'd'}, false, false, "hello 'd");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '\'', ' ', 's'}, false, false, "hello's");
    CHECK_DECODE(tokenizer, {T_HELLO, ' ', '\'', ' ', 'x'}, false, false, "hello'x");
    CHECK_DECODE(tokenizer, {' ', '.'}, false, false, ".");
    CHECK_DECODE(tokenizer, {'.'}, false, false, ".");

    Array<char> out;
    CHECK(out.push_back('>'));
    CHECK(tokenizer.decode(std::vector<int32_t>{T_HELLO, ' ', '.'}.data(), 3, false, false, out));
    CHECK(std::string(out.begin(), out.end()) == ">hello.");

    out.clear();
    CHECK(tokenizer.token_to_piece(T_SPACE_HELLO, false, out));
    CHECK(std::string(out.begin(), out.end()) == " hello");
    out.clear();
    CHECK(tokenizer.token_to_piece(T_EOT, false, out));
    CHECK(out.empty());
    CHECK(tokenizer.token_to_piece(T_EOT, true, out));
    CHECK(std::string(out.begin(), out.end()) == "<|endoftext|>");
    out.clear();
    CHECK(tokenizer.token_to_piece(T_TAG, false, out));
    CHECK(std::string(out.begin(), out.end()) == "<tag>");

    Spec spec;
    spec.pre = "smollm";
    Loaded smollm{spec};
    CHECK_DECODE(smollm.tokenizer, {T_HELLO, ' ', '.'}, false, false, "hello .");
    CHECK_DECODE(smollm.tokenizer, {T_HELLO, ' ', '\'', 's'}, false, false, "hello 's");

    spec = Spec();
    spec.add_bos = true;
    spec.add_eos = true;
    Loaded with_flags{spec};
    CHECK_DECODE(with_flags.tokenizer, {T_EOT, T_HELLO, T_EOT}, true, true, "hello");
    CHECK_DECODE(with_flags.tokenizer, {T_EOT, T_HELLO, T_EOT}, false, true, "<|endoftext|>hello<|endoftext|>");
    CHECK_DECODE(with_flags.tokenizer, {T_EOT, T_HELLO, T_EOT}, false, false, "hello");
    CHECK_DECODE(with_flags.tokenizer, {T_EOT}, true, true, "");
    CHECK_DECODE(with_flags.tokenizer, {T_HELLO, T_EOT, T_HELLO}, true, true, "hello<|endoftext|>hello");

    spec = Spec();
    spec.add_space_prefix = true;
    Loaded prefixed{spec};
    CHECK_DECODE(prefixed.tokenizer, {T_SPACE_HELLO, T_SPACE_WORLD}, false, false, "hello world");
    CHECK_DECODE(prefixed.tokenizer, {T_HELLO, T_SPACE_WORLD}, false, false, "hello world");
    CHECK_DECODE(prefixed.tokenizer, {T_EOT, T_SPACE_HELLO}, false, false, " hello");
    CHECK_DECODE(prefixed.tokenizer, {T_EOT, T_SPACE_HELLO}, false, true, "<|endoftext|> hello");
    CHECK_DECODE(prefixed.tokenizer, {T_TWO_SPACES}, false, false, " ");

    spec.add_bos = true;
    Loaded prefixed_bos{spec};
    CHECK_DECODE(prefixed_bos.tokenizer, {T_EOT, T_SPACE_HELLO}, true, false, " hello");
    CHECK_DECODE(prefixed_bos.tokenizer, {T_SPACE_HELLO}, true, false, "hello");
}

void test_round_trip()
{
    Loaded loaded{Spec()};
    Tokenizer& tokenizer = loaded.tokenizer;

    const char* texts[] = {
        "hello world", " hello  world ", "don't", "12 34", "a\tb\nc", "<|im_start|>hello<|endoftext|>", "\xff\xfe",
        "\xc3\xa9\xe6\x97\xa5", "[EMPTY_280]", "'", " '", "' '",
    };
    for (const char* text : texts)
    {
        const std::vector<int32_t> tokens = encode(tokenizer, text, false, true);
        std::string expected = text;
        if (std::strcmp(text, "\xff\xfe") == 0)
        {
            expected = "\xef\xbf\xbd\xef\xbf\xbd";
        }
        CHECK_DECODE(tokenizer, tokens, false, true, expected);
    }
}

// without the duplicate "lo": llama.cpp asserts that every token text is unique
bool write_image(const char* path)
{
    Spec spec;
    spec.token_count = static_cast<size_t>(VOCAB_SIZE) - 1;
    const Image image = make_image(spec);
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(image.bytes.data()), static_cast<std::streamsize>(image.bytes.size()));
    return stream.good();
}

}

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--write-gguf") == 0)
    {
        if (!write_image(argv[2]))
        {
            std::fprintf(stderr, "tokenizer_unit: cannot write '%s'\n", argv[2]);
            return 1;
        }
        return 0;
    }

    test_load_errors();
    test_vocabulary();
    test_encode_gpt2();
    test_encode_smollm();
    test_special_tokens();
    test_decode();
    test_round_trip();

    std::printf("%s tokenizer_unit: %d checks, %d failed\n", g_failures == 0 ? "ok  " : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
