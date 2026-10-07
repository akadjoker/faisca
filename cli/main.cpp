#include "faisca/gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace
{

using namespace faisca;

constexpr const char* FAISCA_VERSION = "0.3.0-wip";
constexpr size_t PREVIEW_ITEMS = 6;

void print_usage()
{
    std::printf(
        "faisca %s\n"
        "\n"
        "usage:\n"
        "  faisca info <model.gguf> [--meta] [--tensors]\n"
        "  faisca version\n"
        "\n"
        "info options:\n"
        "  --meta      list every metadata key\n"
        "  --tensors   list every tensor\n",
        FAISCA_VERSION);
}

struct Text
{
    char text[64];
};

Text human_bytes(uint64_t bytes)
{
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4)
    {
        value /= 1024.0;
        ++unit;
    }

    Text out;
    if (unit == 0)
    {
        std::snprintf(out.text, sizeof(out.text), "%" PRIu64 " B", bytes);
    }
    else
    {
        std::snprintf(out.text, sizeof(out.text), "%.2f %s", value, units[unit]);
    }
    return out;
}

Text human_count(uint64_t count)
{
    Text out;
    if (count >= 1000000000ull)
    {
        std::snprintf(out.text, sizeof(out.text), "%.2f B", static_cast<double>(count) / 1e9);
    }
    else if (count >= 1000000ull)
    {
        std::snprintf(out.text, sizeof(out.text), "%.2f M", static_cast<double>(count) / 1e6);
    }
    else
    {
        std::snprintf(out.text, sizeof(out.text), "%" PRIu64, count);
    }
    return out;
}

int print_text(std::string_view text, size_t max_length)
{
    int written = 0;
    const size_t limit = text.size() < max_length ? text.size() : max_length;
    for (size_t i = 0; i < limit; ++i)
    {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '\n')
        {
            written += std::printf("\\n");
        }
        else if (c == '\t')
        {
            written += std::printf("\\t");
        }
        else if (c < 0x20 || c == 0x7f)
        {
            written += std::printf("?");
        }
        else
        {
            written += std::printf("%c", static_cast<char>(c));
        }
    }
    if (text.size() > max_length)
    {
        written += std::printf("...");
    }
    return written;
}

void print_padded(std::string_view text, size_t max_length, int width)
{
    const int written = print_text(text, max_length);
    std::printf("%*s", written < width ? width - written : 0, "");
}

template <typename T, typename PrintOne>
void print_numbers(const GgufArray& array, PrintOne print_one)
{
    Array<T> values;
    if (!array.copy_to(values))
    {
        return;
    }

    const size_t shown = values.size() < PREVIEW_ITEMS ? values.size() : PREVIEW_ITEMS;
    for (size_t i = 0; i < shown; ++i)
    {
        std::printf("%s", i == 0 ? " = " : ", ");
        print_one(values[i]);
    }
    if (values.size() > shown)
    {
        std::printf(", ...");
    }
}

void print_array(const GgufArray& array)
{
    std::printf("%s[%" PRIu64 "]", gguf_type_name(array.elem_type), array.count);

    const auto print_unsigned = [](uint64_t value)
    {
        std::printf("%" PRIu64, value);
    };
    const auto print_signed = [](int64_t value)
    {
        std::printf("%" PRId64, value);
    };
    const auto print_float = [](double value)
    {
        std::printf("%g", value);
    };

    switch (array.elem_type)
    {
        case GgufType::String:
        {
            Array<std::string_view> items;
            if (!array.strings_to(items))
            {
                return;
            }
            const size_t shown = items.size() < PREVIEW_ITEMS ? items.size() : PREVIEW_ITEMS;
            for (size_t i = 0; i < shown; ++i)
            {
                std::printf("%s", i == 0 ? " = \"" : ", \"");
                print_text(items[i], 16);
                std::printf("\"");
            }
            if (items.size() > shown)
            {
                std::printf(", ...");
            }
            return;
        }
        case GgufType::UInt8:
            print_numbers<uint8_t>(array, print_unsigned);
            return;
        case GgufType::Int8:
            print_numbers<int8_t>(array, print_signed);
            return;
        case GgufType::UInt16:
            print_numbers<uint16_t>(array, print_unsigned);
            return;
        case GgufType::Int16:
            print_numbers<int16_t>(array, print_signed);
            return;
        case GgufType::UInt32:
            print_numbers<uint32_t>(array, print_unsigned);
            return;
        case GgufType::Int32:
            print_numbers<int32_t>(array, print_signed);
            return;
        case GgufType::UInt64:
            print_numbers<uint64_t>(array, print_unsigned);
            return;
        case GgufType::Int64:
            print_numbers<int64_t>(array, print_signed);
            return;
        case GgufType::Float32:
            print_numbers<float>(array, print_float);
            return;
        case GgufType::Float64:
            print_numbers<double>(array, print_float);
            return;
        case GgufType::Bool:
        case GgufType::Array:
            return;
    }
}

void print_value(const GgufValue& value)
{
    switch (value.type)
    {
        case GgufType::UInt8:
        case GgufType::UInt16:
        case GgufType::UInt32:
        case GgufType::UInt64:
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
            std::printf("%g", value.f64);
            return;
        case GgufType::Bool:
            std::printf("%s", value.u64 != 0 ? "true" : "false");
            return;
        case GgufType::String:
            std::printf("\"");
            print_text(value.str, 72);
            std::printf("\"");
            return;
        case GgufType::Array:
            print_array(value.array);
            return;
    }
}

void print_summary_key(const GgufFile& file, const char* label, std::string_view prefix, const char* suffix)
{
    char key[160];
    const int length = std::snprintf(key, sizeof(key), "%.*s%s", static_cast<int>(prefix.size()), prefix.data(), suffix);
    if (length < 0 || static_cast<size_t>(length) >= sizeof(key))
    {
        return;
    }

    const GgufValue* value = file.find(std::string_view(key, static_cast<size_t>(length)));
    if (value != nullptr)
    {
        std::printf("  %-18s ", label);
        print_value(*value);
        std::printf("\n");
    }
}

int command_info(int argc, char** argv)
{
    const char* path = nullptr;
    bool show_meta = false;
    bool show_tensors = false;

    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--meta") == 0)
        {
            show_meta = true;
        }
        else if (std::strcmp(argv[i], "--tensors") == 0)
        {
            show_tensors = true;
        }
        else if (argv[i][0] == '-' && argv[i][1] != '\0')
        {
            std::fprintf(stderr, "faisca: unknown option '%s'\n", argv[i]);
            return 2;
        }
        else if (path == nullptr)
        {
            path = argv[i];
        }
        else
        {
            std::fprintf(stderr, "faisca: unexpected argument '%s'\n", argv[i]);
            return 2;
        }
    }

    if (path == nullptr)
    {
        std::fprintf(stderr, "faisca: info needs a model file\n");
        return 2;
    }

    Error error;
    GgufFile file;
    if (!file.open(path, error))
    {
        std::fprintf(stderr, "faisca: %s\n", error.c_str());
        return 1;
    }

    constexpr uint32_t type_slots = 64;
    uint64_t type_counts[type_slots] = {};
    uint64_t total_elements = 0;
    uint64_t total_bytes = 0;
    for (const GgufTensor& tensor : file.tensors())
    {
        total_elements += tensor.n_elements;
        total_bytes += tensor.n_bytes;
        const uint32_t id = static_cast<uint32_t>(tensor.type);
        if (id < type_slots)
        {
            type_counts[id] += 1;
        }
    }

    std::printf("file\n");
    std::printf("  %-18s %s\n", "path", path);
    std::printf("  %-18s %s\n", "size", human_bytes(file.file_size()).text);
    std::printf("  %-18s v%" PRIu32 ", alignment %" PRIu32 "\n", "gguf", file.version(), file.alignment());
    std::printf("  %-18s %zu keys\n", "metadata", file.metadata().size());
    std::printf("  %-18s %zu, %s parameters, %s\n", "tensors", file.tensors().size(), human_count(total_elements).text,
                human_bytes(total_bytes).text);

    bool first_type = true;
    for (uint32_t id = 0; id < type_slots; ++id)
    {
        if (type_counts[id] == 0)
        {
            continue;
        }
        if (first_type)
        {
            std::printf("  %-18s ", "tensor types");
        }
        std::printf("%s%s x %" PRIu64, first_type ? "" : ", ", ggml_type_info(static_cast<GgmlType>(id))->name,
                    type_counts[id]);
        first_type = false;
    }
    if (!first_type)
    {
        std::printf("\n");
    }

    std::string_view architecture;
    if (file.get_string("general.architecture", architecture))
    {
        std::printf("\nmodel\n");
        print_summary_key(file, "name", "general", ".name");
        print_summary_key(file, "architecture", "general", ".architecture");
        print_summary_key(file, "context length", architecture, ".context_length");
        print_summary_key(file, "embedding length", architecture, ".embedding_length");
        print_summary_key(file, "blocks", architecture, ".block_count");
        print_summary_key(file, "feed forward", architecture, ".feed_forward_length");
        print_summary_key(file, "heads", architecture, ".attention.head_count");
        print_summary_key(file, "kv heads", architecture, ".attention.head_count_kv");
        print_summary_key(file, "rope freq base", architecture, ".rope.freq_base");
        print_summary_key(file, "rms epsilon", architecture, ".attention.layer_norm_rms_epsilon");
        print_summary_key(file, "tokenizer", "tokenizer.ggml", ".model");
        print_summary_key(file, "vocabulary", "tokenizer.ggml", ".tokens");
    }

    if (show_meta)
    {
        std::printf("\nmetadata\n");
        for (const GgufKeyValue& kv : file.metadata())
        {
            std::printf("  ");
            print_padded(kv.key, 96, 44);
            std::printf(" %-5s ", gguf_type_name(kv.value.type));
            print_value(kv.value);
            std::printf("\n");
        }
    }

    if (show_tensors)
    {
        std::printf("\ntensors\n");
        for (const GgufTensor& tensor : file.tensors())
        {
            char shape[112];
            size_t at = 0;
            shape[at++] = '[';
            for (uint32_t d = 0; d < tensor.n_dims; ++d)
            {
                const int written = std::snprintf(shape + at, sizeof(shape) - at - 2, "%s%" PRIu64, d == 0 ? "" : ", ",
                                                  tensor.dims[d]);
                if (written > 0)
                {
                    at += static_cast<size_t>(written);
                }
            }
            shape[at++] = ']';
            shape[at] = '\0';

            std::printf("  ");
            print_padded(tensor.name, 96, 40);
            std::printf(" %-8s %-24s %12s  @%" PRIu64 "\n", ggml_type_info(tensor.type)->name, shape,
                        human_bytes(tensor.n_bytes).text, tensor.offset);
        }
    }

    return 0;
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        print_usage();
        return 2;
    }

    const char* command = argv[1];
    if (std::strcmp(command, "info") == 0)
    {
        return command_info(argc, argv);
    }
    if (std::strcmp(command, "version") == 0 || std::strcmp(command, "--version") == 0)
    {
        std::printf("faisca %s\n", FAISCA_VERSION);
        return 0;
    }
    if (std::strcmp(command, "help") == 0 || std::strcmp(command, "--help") == 0 || std::strcmp(command, "-h") == 0)
    {
        print_usage();
        return 0;
    }

    std::fprintf(stderr, "faisca: unknown command '%s'\n", command);
    print_usage();
    return 2;
}
