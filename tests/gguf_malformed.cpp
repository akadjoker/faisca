#include "faisca/gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace faisca;

namespace
{

uint64_t touch_everything(const GgufFile& file)
{
    uint64_t sum = 0;

    for (const GgufKeyValue& kv : file.metadata())
    {
        sum += checksum64(reinterpret_cast<const uint8_t*>(kv.key.data()), kv.key.size());

        if (kv.value.type == GgufType::String)
        {
            sum += checksum64(reinterpret_cast<const uint8_t*>(kv.value.str.data()), kv.value.str.size());
        }
        else if (kv.value.type == GgufType::Array)
        {
            sum += checksum64(kv.value.array.data, static_cast<size_t>(kv.value.array.byte_size));

            Array<std::string_view> items;
            if (kv.value.array.strings_to(items))
            {
                for (const std::string_view item : items)
                {
                    sum += checksum64(reinterpret_cast<const uint8_t*>(item.data()), item.size());
                }
            }

            Array<int32_t> numbers;
            if (kv.value.array.copy_to(numbers))
            {
                for (const int32_t number : numbers)
                {
                    sum += static_cast<uint64_t>(static_cast<int64_t>(number));
                }
            }
        }
    }

    for (const GgufTensor& tensor : file.tensors())
    {
        sum += checksum64(reinterpret_cast<const uint8_t*>(tensor.name.data()), tensor.name.size());
        sum += checksum64(tensor.data, static_cast<size_t>(tensor.n_bytes));
    }

    return sum;
}

struct Counters
{
    uint64_t rejected = 0;
    uint64_t accepted = 0;
};

bool try_image(const uint8_t* image, size_t size, Counters& counters)
{
    // exact size, so ASan reports any read past the end of the file
    uint8_t* copy = static_cast<uint8_t*>(std::malloc(size == 0 ? 1 : size));
    if (copy == nullptr)
    {
        std::fprintf(stderr, "out of memory\n");
        std::exit(1);
    }
    if (size != 0)
    {
        std::memcpy(copy, image, size);
    }

    bool accepted = false;
    {
        Error error;
        GgufFile file;
        if (file.parse(copy, size, error))
        {
            touch_everything(file);
            counters.accepted += 1;
            accepted = true;
        }
        else
        {
            if (error.empty() || file.is_open())
            {
                std::fprintf(stderr, "rejected without an error message, or left open\n");
                std::exit(1);
            }
            counters.rejected += 1;
        }
    }

    std::free(copy);
    return accepted;
}

bool read_file(const std::string& path, std::vector<uint8_t>& out)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return true;
}

template <typename T>
void poke(std::vector<uint8_t>& image, size_t offset, T value)
{
    std::memcpy(image.data() + offset, &value, sizeof(T));
}

bool test_file(const std::string& path)
{
    std::vector<uint8_t> image;
    if (!read_file(path, image))
    {
        std::fprintf(stderr, "cannot read '%s'\n", path.c_str());
        return false;
    }

    Error error;
    GgufFile valid;
    if (!valid.parse(image.data(), image.size(), error))
    {
        std::fprintf(stderr, "'%s' should be valid: %s\n", path.c_str(), error.c_str());
        return false;
    }
    touch_everything(valid);

    uint64_t needed = 24;
    for (const GgufTensor& tensor : valid.tensors())
    {
        const uint64_t end = valid.data_offset() + tensor.offset + tensor.n_bytes;
        needed = end > needed ? end : needed;
    }
    const size_t header_size = static_cast<size_t>(
        valid.data_offset() < image.size() ? valid.data_offset() : image.size());

    bool ok = true;

    Counters truncation;
    for (size_t length = 0; length < image.size(); ++length)
    {
        if (length > header_size + 256 && length % 97 != 0 && length + 64 < image.size())
        {
            continue;
        }
        const bool accepted = try_image(image.data(), length, truncation);
        if (accepted && length < needed)
        {
            std::fprintf(stderr, "'%s': accepted a file truncated to %zu bytes, %" PRIu64 " are needed\n",
                         path.c_str(), length, needed);
            ok = false;
        }
    }

    Counters mutation;
    std::mt19937_64 rng(0xfa15ca);
    std::vector<uint8_t> damaged;
    for (int iteration = 0; iteration < 20000 && header_size > 0; ++iteration)
    {
        damaged = image;
        const int edits = 1 + static_cast<int>(rng() % 3);
        for (int e = 0; e < edits; ++e)
        {
            const size_t offset = static_cast<size_t>(rng() % header_size);
            switch (rng() % 4)
            {
                case 0:
                    damaged[offset] = static_cast<uint8_t>(rng());
                    break;
                case 1:
                    damaged[offset] = 0xff;
                    break;
                case 2:
                    damaged[offset] = 0x00;
                    break;
                default:
                    damaged[offset] = static_cast<uint8_t>(damaged[offset] ^ (1u << (rng() % 8)));
                    break;
            }
        }
        try_image(damaged.data(), damaged.size(), mutation);
    }

    Counters targeted;
    const uint64_t huge_counts[] = {0xffffffffffffffffull, 0x8000000000000000ull, 0x00000000ffffffffull,
                                    static_cast<uint64_t>(image.size()), 1ull << 40};
    for (const uint64_t count : huge_counts)
    {
        damaged = image;
        poke(damaged, 8, count);
        if (try_image(damaged.data(), damaged.size(), targeted))
        {
            std::fprintf(stderr, "'%s': accepted tensor count %" PRIu64 "\n", path.c_str(), count);
            ok = false;
        }

        damaged = image;
        poke(damaged, 16, count);
        if (try_image(damaged.data(), damaged.size(), targeted))
        {
            std::fprintf(stderr, "'%s': accepted metadata count %" PRIu64 "\n", path.c_str(), count);
            ok = false;
        }
    }

    const uint32_t bad_versions[] = {0u, 1u, 4u, 0x03000000u, 0xffffffffu};
    for (const uint32_t version : bad_versions)
    {
        damaged = image;
        poke(damaged, 4, version);
        if (try_image(damaged.data(), damaged.size(), targeted))
        {
            std::fprintf(stderr, "'%s': accepted version %u\n", path.c_str(), version);
            ok = false;
        }
    }

    damaged = image;
    damaged[0] = 'X';
    if (try_image(damaged.data(), damaged.size(), targeted))
    {
        std::fprintf(stderr, "'%s': accepted a bad magic\n", path.c_str());
        ok = false;
    }

    std::printf("%s %s\n", ok ? "ok  " : "FAIL", path.c_str());
    std::printf("     truncations: %" PRIu64 " rejected, %" PRIu64 " accepted\n", truncation.rejected,
                truncation.accepted);
    std::printf("     mutations:   %" PRIu64 " rejected, %" PRIu64 " accepted\n", mutation.rejected,
                mutation.accepted);
    std::printf("     targeted:    %" PRIu64 " rejected, %" PRIu64 " accepted\n", targeted.rejected,
                targeted.accepted);
    return ok;
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: gguf_malformed <valid.gguf> [more.gguf ...]\n");
        return 2;
    }

    bool ok = true;
    for (int i = 1; i < argc; ++i)
    {
        ok = test_file(argv[i]) && ok;
    }
    return ok ? 0 : 1;
}
