// Encodes text with a Hugging Face tokenizer.json (special tokens in the text parsed), one id per
// line, for comparison with the transformers tokenizer.
// Usage: hf_tokenize <tokenizer.json> <file with the text>
#include "faisca/tokenizer.h"

#include <cstdio>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: hf_tokenize <tokenizer.json> <text file>\n");
        return 2;
    }
    faisca::Error error;
    faisca::Tokenizer tokenizer;
    if (!tokenizer.load_hf(argv[1], error))
    {
        std::printf("hf_tokenize: %s\n", error.message);
        return 1;
    }
    FILE* file = std::fopen(argv[2], "rb");
    if (file == nullptr)
    {
        std::printf("cannot open %s\n", argv[2]);
        return 1;
    }
    std::vector<char> text;
    char buffer[4096];
    size_t n;
    while ((n = std::fread(buffer, 1, sizeof(buffer), file)) > 0)
    {
        text.insert(text.end(), buffer, buffer + n);
    }
    std::fclose(file);
    faisca::Array<int32_t> ids;
    if (!tokenizer.encode(std::string_view(text.data(), text.size()), false, true, ids))
    {
        std::printf("hf_tokenize: out of memory\n");
        return 1;
    }
    for (size_t i = 0; i < ids.size(); ++i)
    {
        std::printf("%d\n", ids[i]);
    }
    return 0;
}
