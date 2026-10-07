#include "faisca/gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string_view>

using namespace faisca;

namespace
{

void print_text(std::string_view text)
{
    std::fwrite(text.data(), 1, text.size(), stdout);
}

uint64_t double_bits(double value)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void print_value(const GgufValue& value)
{
    switch (value.type)
    {
        case GgufType::UInt8:
        case GgufType::UInt16:
        case GgufType::UInt32:
        case GgufType::UInt64:
        case GgufType::Bool:
            std::printf("%" PRIu64, value.u64);
            return;
        case GgufType::Int8:
        case GgufType::Int16:
        case GgufType::Int32:
        case GgufType::Int64:
            std::printf("%" PRId64, value.i64);
            return;
        case GgufType::Float32:
        case GgufType::Float64:
            std::printf("%" PRIu64, double_bits(value.f64));
            return;
        case GgufType::String:
            std::printf("%zu:%" PRIu64, value.str.size(),
                        checksum64(reinterpret_cast<const uint8_t*>(value.str.data()), value.str.size()));
            return;
        case GgufType::Array:
            std::printf("%u:%" PRIu64 ":%" PRIu64 ":%" PRIu64, static_cast<uint32_t>(value.array.elem_type),
                        value.array.count, value.array.byte_size,
                        checksum64(value.array.data, static_cast<size_t>(value.array.byte_size)));
            return;
    }
}

}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: gguf_dump <file.gguf>\n");
        return 2;
    }

    Error error;
    GgufFile file;
    if (!file.open(argv[1], error))
    {
        std::fprintf(stderr, "gguf_dump: %s\n", error.c_str());
        return 1;
    }

    std::printf("version\t%u\n", file.version());
    std::printf("alignment\t%u\n", file.alignment());
    std::printf("data_offset\t%" PRIu64 "\n", file.data_offset());

    for (const GgufKeyValue& kv : file.metadata())
    {
        std::printf("kv\t");
        print_text(kv.key);
        std::printf("\t%u\t", static_cast<uint32_t>(kv.value.type));
        print_value(kv.value);
        std::printf("\n");
    }

    for (const GgufTensor& tensor : file.tensors())
    {
        std::printf("tensor\t");
        print_text(tensor.name);
        std::printf("\t%u\t%u\t", static_cast<uint32_t>(tensor.type), tensor.n_dims);
        for (uint32_t d = 0; d < tensor.n_dims; ++d)
        {
            std::printf("%s%" PRIu64, d == 0 ? "" : ",", tensor.dims[d]);
        }
        std::printf("\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n", tensor.offset, tensor.n_elements,
                    tensor.n_bytes, checksum64(tensor.data, static_cast<size_t>(tensor.n_bytes)));
    }

    return 0;
}
