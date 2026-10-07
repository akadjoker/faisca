#include "faisca/tokenizer.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

using namespace faisca;

namespace
{

void print_hex(const Array<char>& bytes)
{
    for (const char c : bytes)
    {
        std::printf("%02x", static_cast<unsigned>(static_cast<uint8_t>(c)));
    }
}

}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: tokenizer_dump <model.gguf> <cases.bin>\n");
        return 2;
    }

    Error error;
    GgufFile file;
    Tokenizer tokenizer;
    if (!file.open(argv[1], error) || !tokenizer.load(file, error))
    {
        std::fprintf(stderr, "tokenizer_dump: %s\n", error.c_str());
        return 1;
    }

    std::ifstream stream(argv[2], std::ios::binary);
    const std::vector<char> cases((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    Array<int32_t> tokens;
    Array<char> text;
    size_t at = 0;
    while (at + 8 <= cases.size())
    {
        uint32_t flags = 0;
        uint32_t length = 0;
        std::memcpy(&flags, cases.data() + at, 4);
        std::memcpy(&length, cases.data() + at + 4, 4);
        at += 8;
        if (at + length > cases.size())
        {
            break;
        }

        tokens.clear();
        if (!tokenizer.encode(std::string_view(cases.data() + at, length), (flags & 1) != 0, (flags & 2) != 0, tokens))
        {
            return 1;
        }
        at += length;

        std::printf("E");
        for (const int32_t token : tokens)
        {
            std::printf(" %" PRId32, token);
        }

        text.clear();
        tokenizer.decode(tokens.data(), tokens.size(), false, true, text);
        std::printf("|D ");
        print_hex(text);

        text.clear();
        tokenizer.decode(tokens.data(), tokens.size(), true, false, text);
        std::printf("|d ");
        print_hex(text);
        std::printf("\n");
    }
    return 0;
}
