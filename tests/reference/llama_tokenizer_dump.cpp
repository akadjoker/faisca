#include "llama.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <vector>

namespace
{

void print_hex(const char* bytes, int32_t count)
{
    for (int32_t i = 0; i < count; ++i)
    {
        std::printf("%02x", static_cast<unsigned>(static_cast<uint8_t>(bytes[i])));
    }
}

void quiet(ggml_log_level, const char*, void*)
{
}

}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: llama_tokenizer_dump <model.gguf> <cases.bin>\n");
        return 2;
    }

    llama_log_set(quiet, nullptr);
    llama_backend_init();

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model* model = llama_model_load_from_file(argv[1], params);
    if (model == nullptr)
    {
        std::fprintf(stderr, "llama_tokenizer_dump: cannot load '%s'\n", argv[1]);
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::ifstream stream(argv[2], std::ios::binary);
    const std::vector<char> cases((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    std::vector<llama_token> tokens;
    std::vector<char> text;
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

        tokens.resize(static_cast<size_t>(length) + 16);
        int32_t count = -1;
        try
        {
            count = llama_tokenize(vocab, cases.data() + at, static_cast<int32_t>(length), tokens.data(),
                                   static_cast<int32_t>(tokens.size()), (flags & 1) != 0, (flags & 2) != 0);
        }
        catch (const std::exception&)
        {
            count = -1;
        }
        at += length;
        if (count < 0)
        {
            std::printf("FAILED\n");
            continue;
        }

        std::printf("E");
        for (int32_t i = 0; i < count; ++i)
        {
            std::printf(" %" PRId32, tokens[static_cast<size_t>(i)]);
        }

        text.resize(static_cast<size_t>(count) * 512 + 64);
        int32_t written = llama_detokenize(vocab, tokens.data(), count, text.data(), static_cast<int32_t>(text.size()),
                                           false, true);
        std::printf("|D ");
        print_hex(text.data(), written < 0 ? 0 : written);

        written = llama_detokenize(vocab, tokens.data(), count, text.data(), static_cast<int32_t>(text.size()), true,
                                   false);
        std::printf("|d ");
        print_hex(text.data(), written < 0 ? 0 : written);
        std::printf("\n");
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
