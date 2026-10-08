#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/ops.h"
#include "faisca/runner.h"
#include "faisca/session.h"
#include "faisca/speculative.h"
#if defined(FAISCA_VULKAN)
#include "faisca/vulkan_runner.h"
#endif
#include "faisca/tokenizer.h"
#include "faisca/audio.h"
#include "faisca/whisper_model.h"
#include "faisca/pocket_model.h"
#include "faisca/pocket_runner.h"
#include "faisca/whisper_runner.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
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
        "  faisca chat <model.gguf> [-n <tokens>] [-c <context>] [-t <threads>] [--session <file>] [--kv-f32]\n"
        "  faisca generate <model.gguf> -p <prompt> [-n <tokens>] [-c <context>] [-t <threads>] [--raw] [--f32] [--kv-f32]\n"
        "                                [--draft <draft.gguf>] [--draft-n <tokens>] [--vulkan]\n"
        "  faisca transcribe <model.bin> <audio.wav> [--language <code>] [-t <threads>] [--f32]\n"
        "  faisca version\n"
        "\n"
        "info options:\n"
        "  --meta      list every metadata key\n"
        "  --tensors   list every tensor\n"
        "\n"
        "generate options (greedy):\n"
        "  -p <text>   the user message, wrapped in the chat template unless --raw\n"
        "  -n <count>  tokens to generate (default 128)\n"
        "  -c <size>   context size in tokens (default 2048)\n"
        "  -t <count>  threads (default 1)\n"
        "  --raw       feed the prompt as is\n"
        "  --f32       convert the weights to float32 at load instead of computing on Q8_0\n"
        "  --kv-f32    keep the KV cache in float32 instead of f16\n"
        "  --draft     a smaller model of the same vocabulary that proposes tokens (speculative decoding)\n"
        "  --draft-n   tokens proposed per round, default 3\n"
        "  --vulkan    run on the Vulkan device instead of the CPU (float32 activations, no draft)\n"
        "\n"
        "chat reads one user message per line from standard input and keeps the conversation in the\n"
        "KV cache between turns; --session loads the conversation from the file if it exists and\n"
        "saves it after every reply.\n"
        "\n"
        "transcribe runs a Whisper model (whisper.cpp ggml file) on a 16 kHz 16 bit WAV, greedy,\n"
        "without timestamps; --language is the two letter code (default en, pt for Portuguese);\n"
        "the matrices run on Q8_0 unless --f32 keeps the file's values.\n"
        "\n"
        "speak synthesizes -p (or standard input) with a Pocket TTS pack (pockettts.cpp GGUF) and one\n"
        "of its voices into a 24 kHz 16 bit WAV (default out.wav); --seed picks the noise (default 0).\n",
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

double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// the chat template the vocabulary asks for: ChatML (SmolLM2) or Llama 3
struct ChatTemplate
{
    const char* first_user;
    const char* next_user;
    const char* assistant;
};

ChatTemplate chat_template(Tokenizer& tokenizer)
{
    Array<int32_t> probe;
    if (tokenizer.encode("<|start_header_id|>", false, true, probe) && probe.size() == 1 &&
        tokenizer.token_type(probe[0]) == TokenType::Control)
    {
        return {"<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"};
    }
    return {"<|im_start|>user\n", "<|im_end|>\n<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"};
}

bool append_text(Array<char>& out, const char* text)
{
    return out.append(text, std::strlen(text));
}

int command_generate(int argc, char** argv)
{
    const char* path = nullptr;
    const char* prompt = nullptr;
    int count = 128;
    long context = 2048;
    bool raw = false;
    bool dequantize = false;
    bool kv_f16 = true;
    int threads = 1;
    const char* draft_path = nullptr;
    int draft_n = 3;
    bool vulkan = false;

    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc)
        {
            prompt = argv[++i];
        }
        else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            count = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc)
        {
            context = std::atol(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc)
        {
            threads = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--raw") == 0)
        {
            raw = true;
        }
        else if (std::strcmp(argv[i], "--f32") == 0)
        {
            dequantize = true;
        }
        else if (std::strcmp(argv[i], "--kv-f32") == 0)
        {
            kv_f16 = false;
        }
        else if (std::strcmp(argv[i], "--draft") == 0 && i + 1 < argc)
        {
            draft_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--draft-n") == 0 && i + 1 < argc)
        {
            draft_n = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--vulkan") == 0)
        {
            vulkan = true;
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

    if (path == nullptr || prompt == nullptr)
    {
        std::fprintf(stderr, "faisca: generate needs a model file and -p <prompt>\n");
        return 2;
    }
    if (count < 1 || context < 1 || context > 0x7fffffffL || draft_n < 1)
    {
        std::fprintf(stderr, "faisca: -n, -c and --draft-n must be positive\n");
        return 2;
    }

    if (vulkan && draft_path != nullptr)
    {
        std::fprintf(stderr, "faisca: --draft is not available with --vulkan\n");
        return 2;
    }
#if !defined(FAISCA_VULKAN)
    if (vulkan)
    {
        std::fprintf(stderr, "faisca: built without the Vulkan backend\n");
        return 1;
    }
#endif

    Error error;
    GgufFile file;
    Tokenizer tokenizer;
    LlamaModel model;
    ThreadPool pool;
    Runner runner;
#if defined(FAISCA_VULKAN)
    VulkanDevice device;
    VulkanRunner vulkan_runner;
#endif

    const auto load_start = std::chrono::steady_clock::now();
    if (!file.open(path, error) || !tokenizer.load(file, error))
    {
        std::fprintf(stderr, "faisca: %s\n", error.c_str());
        return 1;
    }
#if defined(FAISCA_VULKAN)
    if (vulkan && (!device.init(error) || !vulkan_runner.init(device, file, static_cast<uint32_t>(context), error)))
    {
        std::fprintf(stderr, "faisca: %s\n", error.c_str());
        return 1;
    }
#endif
    if (!vulkan && (!pool.start(threads, error) || !model.load(file, error, dequantize, &pool) ||
                    !runner.init(model, static_cast<uint32_t>(context), pool, error, kv_f16)))
    {
        std::fprintf(stderr, "faisca: %s\n", error.c_str());
        return 1;
    }
    const auto run_forward = [&](const int32_t* ids, size_t n, Array<float>& out) -> bool
    {
#if defined(FAISCA_VULKAN)
        if (vulkan)
        {
            return vulkan_runner.forward(ids, n, out, error);
        }
#endif
        return runner.forward(ids, n, out, error);
    };
    GgufFile draft_file;
    LlamaModel draft_model;
    Runner draft_runner;
    Speculative speculative;
    if (draft_path != nullptr &&
        (!draft_file.open(draft_path, error) || !draft_model.load(draft_file, error, dequantize, &pool) ||
         !draft_runner.init(draft_model, static_cast<uint32_t>(context), pool, error, kv_f16) ||
         !speculative.init(runner, draft_runner, static_cast<size_t>(draft_n), error)))
    {
        std::fprintf(stderr, "faisca: %s\n", error.c_str());
        return 1;
    }
    const double load_seconds = seconds_since(load_start);

    Array<char> text;
    if (raw)
    {
        append_text(text, prompt);
    }
    else
    {
        const ChatTemplate chat = chat_template(tokenizer);
        append_text(text, chat.first_user);
        append_text(text, prompt);
        append_text(text, chat.assistant);
    }

    Array<int32_t> tokens;
    if (!tokenizer.encode(std::string_view(text.data(), text.size()), true, true, tokens) || tokens.empty())
    {
        std::fprintf(stderr, "faisca: the prompt produced no tokens\n");
        return 1;
    }
    if (tokens.size() + static_cast<size_t>(count) > static_cast<size_t>(context))
    {
        std::fprintf(stderr, "faisca: %zu prompt tokens plus %d generated do not fit a context of %ld\n", tokens.size(),
                     count, context);
        return 1;
    }

    Array<float> logits;
    Array<int32_t> pending;
    int32_t best = 0;
    const auto prompt_start = std::chrono::steady_clock::now();
    if (draft_path != nullptr ? !speculative.prompt(tokens.data(), tokens.size(), best, error)
                              : !run_forward(tokens.data(), tokens.size(), logits))
    {
        std::fprintf(stderr, "faisca: %s\n", error.c_str());
        return 1;
    }
    const double prompt_seconds = seconds_since(prompt_start);

    const auto generate_start = std::chrono::steady_clock::now();
    Array<char> piece;
    int generated = 0;
    size_t next_pending = 0;
    for (; generated < count; ++generated)
    {
        if (draft_path == nullptr)
        {
            best = 0;
            for (size_t t = 1; t < logits.size(); ++t)
            {
                if (logits[t] > logits[static_cast<size_t>(best)])
                {
                    best = static_cast<int32_t>(t);
                }
            }
        }
        else if (generated > 0)
        {
            // best was produced by the target; the rounds queue the tokens after it
            if (next_pending == pending.size())
            {
                pending.clear();
                next_pending = 0;
                if (!speculative.round(best, pending, error))
                {
                    std::fprintf(stderr, "\nfaisca: %s\n", error.c_str());
                    return 1;
                }
            }
            best = pending[next_pending++];
        }
        if (best == tokenizer.eos() || tokenizer.token_type(best) == TokenType::Control)
        {
            break;
        }

        piece.clear();
        tokenizer.token_to_piece(best, false, piece);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);

        if (draft_path == nullptr && generated + 1 < count && !run_forward(&best, 1, logits))
        {
            std::fprintf(stderr, "\nfaisca: %s\n", error.c_str());
            return 1;
        }
    }
    const double generate_seconds = seconds_since(generate_start);
    std::printf("\n");

    const char* backend = simd_name();
    int backend_threads = vulkan ? 0 : runner.threads();
    size_t weight_mib = vulkan ? 0 : model.weight_bytes() >> 20;
#if defined(FAISCA_VULKAN)
    if (vulkan)
    {
        backend = device.device_name();
        weight_mib = vulkan_runner.weight_bytes() >> 20;
    }
#endif
    std::fprintf(stderr,
                 "\n%s, %d threads, %zu MiB of weights per token, load %.2f s, prompt %zu tokens in %.2f s (%.1f tok/s), "
                 "%d tokens generated in %.2f s (%.1f tok/s)\n",
                 backend, backend_threads, weight_mib, load_seconds, tokens.size(), prompt_seconds,
                 static_cast<double>(tokens.size()) / prompt_seconds, generated, generate_seconds,
                 generated > 0 ? static_cast<double>(generated) / generate_seconds : 0.0);
    if (draft_path != nullptr)
    {
        std::fprintf(stderr, "draft: %zu rounds, %zu tokens proposed, %zu accepted (%.0f%%)\n", speculative.rounds(),
                     speculative.drafted(), speculative.accepted(),
                     speculative.drafted() > 0 ? 100.0 * static_cast<double>(speculative.accepted()) / static_cast<double>(speculative.drafted()) : 0.0);
    }
    return 0;
}

}

int command_chat(int argc, char** argv)
{
    const char* path = nullptr;
    const char* session_path = nullptr;
    int count = 256;
    long context = 2048;
    int threads = 1;
    bool kv_f16 = true;

    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            count = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc)
        {
            context = std::atol(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc)
        {
            threads = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--session") == 0 && i + 1 < argc)
        {
            session_path = argv[++i];
        }
        else if (std::strcmp(argv[i], "--kv-f32") == 0)
        {
            kv_f16 = false;
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
        std::fprintf(stderr, "faisca: chat needs a model file\n");
        return 2;
    }
    if (count < 1 || context < 1 || context > 0x7fffffffL)
    {
        std::fprintf(stderr, "faisca: -n and -c must be positive\n");
        return 2;
    }

    Error error;
    GgufFile file;
    Tokenizer tokenizer;
    LlamaModel model;
    ThreadPool pool;
    Runner runner;
    Session session;
    if (!file.open(path, error) || !tokenizer.load(file, error) || !pool.start(threads, error) ||
        !model.load(file, error, false, &pool) || !runner.init(model, static_cast<uint32_t>(context), pool, error, kv_f16))
    {
        std::fprintf(stderr, "faisca: %s\n", error.c_str());
        return 1;
    }
    std::string_view name;
    file.get_string("general.name", name);
    char model_name[128] = {};
    std::snprintf(model_name, sizeof(model_name), "%.*s", static_cast<int>(name.size()), name.data());
    if (!session.init(runner, model_name, error))
    {
        std::fprintf(stderr, "faisca: %s\n", error.c_str());
        return 1;
    }
    if (session_path != nullptr)
    {
        FILE* probe = std::fopen(session_path, "rb");
        if (probe != nullptr)
        {
            std::fclose(probe);
            if (!session.load(session_path, error))
            {
                std::fprintf(stderr, "faisca: %s\n", error.c_str());
                return 1;
            }
            std::fprintf(stderr, "faisca: session '%s' loaded, %" PRIu32 " tokens of conversation\n", session_path,
                         session.position());
        }
    }

    Array<int32_t> turn;
    Array<int32_t> conversation;
    Array<float> logits;
    Array<char> text;
    Array<char> piece;
    char line[4096];
    for (;;)
    {
        std::fprintf(stderr, "> ");
        std::fflush(stderr);
        if (std::fgets(line, sizeof(line), stdin) == nullptr)
        {
            break;
        }
        size_t length = std::strlen(line);
        while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r'))
        {
            line[--length] = '\0';
        }
        if (length == 0)
        {
            continue;
        }

        // the turn as tokens appended to what the cache holds, never re-tokenizing the past
        text.clear();
        const bool first = session.tokens().empty();
        const ChatTemplate chat = chat_template(tokenizer);
        append_text(text, first ? chat.first_user : chat.next_user);
        append_text(text, line);
        append_text(text, chat.assistant);
        turn.clear();
        if (!tokenizer.encode(std::string_view(text.data(), text.size()), first, true, turn) || turn.empty())
        {
            std::fprintf(stderr, "faisca: the message produced no tokens\n");
            continue;
        }
        conversation.clear();
        if (!conversation.append(session.tokens().data(), session.tokens().size()) || !conversation.append(turn.data(), turn.size()))
        {
            std::fprintf(stderr, "faisca: out of memory\n");
            return 1;
        }
        if (conversation.size() + static_cast<size_t>(count) > runner.context_size())
        {
            std::fprintf(stderr, "faisca: the conversation of %zu tokens plus %d do not fit a context of %" PRIu32 "\n",
                         conversation.size(), count, runner.context_size());
            break;
        }
        const auto start = std::chrono::steady_clock::now();
        size_t reused = 0;
        if (!session.feed(conversation.data(), conversation.size(), logits, error, &reused))
        {
            std::fprintf(stderr, "faisca: %s\n", error.c_str());
            return 1;
        }
        const double prompt_seconds = seconds_since(start);

        const auto generate_start = std::chrono::steady_clock::now();
        int generated = 0;
        for (; generated < count; ++generated)
        {
            int32_t best = 0;
            for (size_t t = 1; t < logits.size(); ++t)
            {
                if (logits[t] > logits[static_cast<size_t>(best)])
                {
                    best = static_cast<int32_t>(t);
                }
            }
            if (best == tokenizer.eos() || tokenizer.token_type(best) == TokenType::Control)
            {
                break;
            }
            piece.clear();
            tokenizer.token_to_piece(best, false, piece);
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
            if (!session.push(best, logits, error))
            {
                std::fprintf(stderr, "\nfaisca: %s\n", error.c_str());
                return 1;
            }
        }
        const double generate_seconds = seconds_since(generate_start);
        std::printf("\n");
        std::fprintf(stderr, "[%zu tokens reused, %zu processed in %.2f s, %d generated in %.2f s (%.1f tok/s)]\n", reused,
                     conversation.size() - reused, prompt_seconds, generated, generate_seconds,
                     generated > 0 ? static_cast<double>(generated) / generate_seconds : 0.0);
        if (session_path != nullptr && !session.save(session_path, error))
        {
            std::fprintf(stderr, "faisca: %s\n", error.c_str());
            return 1;
        }
    }
    return 0;
}

int command_transcribe(int argc, char** argv)
{
    const char* language = "en";
    int threads = 1;
    bool quantize = true;
    const char* positional[2] = {nullptr, nullptr};
    int n_positional = 0;
    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--language") == 0 && i + 1 < argc)
        {
            language = argv[++i];
        }
        else if (std::strcmp(argv[i], "--f32") == 0)
        {
            quantize = false;
        }
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc)
        {
            threads = std::atoi(argv[++i]);
        }
        else if (argv[i][0] == '-')
        {
            std::fprintf(stderr, "faisca transcribe: unknown option '%s'\n", argv[i]);
            return 2;
        }
        else if (n_positional < 2)
        {
            positional[n_positional++] = argv[i];
        }
    }
    if (n_positional != 2 || threads < 1)
    {
        print_usage();
        return 2;
    }
    const int32_t language_id = faisca::WhisperModel::language_id(language);
    if (language_id < 0)
    {
        std::fprintf(stderr, "faisca transcribe: unknown language '%s'\n", language);
        return 2;
    }

    faisca::Error error;
    faisca::WhisperModel model;
    faisca::ThreadPool pool;
    faisca::WhisperRunner runner;
    faisca::Array<float> samples;
    const auto t0 = std::chrono::steady_clock::now();
    if (!model.load(positional[0], error, quantize) || !pool.start(threads, error) || !runner.init(model, pool, error))
    {
        std::fprintf(stderr, "faisca transcribe: %s\n", error.c_str());
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (!faisca::read_wav(positional[1], samples, error))
    {
        std::fprintf(stderr, "faisca transcribe: %s\n", error.c_str());
        return 1;
    }
    faisca::Array<int32_t> tokens;
    if (!runner.transcribe_audio(samples.data(), samples.size(), language_id, tokens, error))
    {
        std::fprintf(stderr, "faisca transcribe: %s\n", error.c_str());
        return 1;
    }
    const auto t2 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        size_t length = 0;
        const char* bytes = model.token(tokens[i], length);
        std::fwrite(bytes, 1, length, stdout);
    }
    std::printf("\n");
    const double load = std::chrono::duration<double>(t1 - t0).count();
    const double run = std::chrono::duration<double>(t2 - t1).count();
    const double audio = static_cast<double>(samples.size()) / static_cast<double>(faisca::WHISPER_SAMPLE_RATE);
    std::fprintf(stderr, "load %.2f s, %.1f s of audio in %.2f s (%.2fx real time), %zu tokens, %d threads, %s\n", load, audio, run,
                 audio / run, tokens.size(), threads, model.quantized() ? "Q8_0" : "float32");
    return 0;
}

int command_speak(int argc, char** argv)
{
    const char* text = nullptr;
    const char* out = "out.wav";
    int threads = 1;
    bool quantize = true;
    uint32_t seed = 0;
    const char* positional[3] = {nullptr, nullptr, nullptr};
    int n_positional = 0;
    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc)
        {
            text = argv[++i];
        }
        else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            out = argv[++i];
        }
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        }
        else if (std::strcmp(argv[i], "--f32") == 0)
        {
            quantize = false;
        }
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc)
        {
            threads = std::atoi(argv[++i]);
        }
        else if (argv[i][0] == '-')
        {
            std::fprintf(stderr, "faisca speak: unknown option '%s'\n", argv[i]);
            return 2;
        }
        else if (n_positional < 3)
        {
            positional[n_positional++] = argv[i];
        }
    }
    if (n_positional != 3 || threads < 1)
    {
        print_usage();
        return 2;
    }
    faisca::Array<char> input;
    if (text == nullptr)
    {
        char block[4096];
        size_t n = 0;
        while ((n = std::fread(block, 1, sizeof(block), stdin)) > 0)
        {
            if (!input.append(block, n))
            {
                std::fprintf(stderr, "faisca speak: out of memory\n");
                return 1;
            }
        }
        if (!input.push_back('\0'))
        {
            return 1;
        }
        text = input.data();
    }

    faisca::Error error;
    faisca::PocketModel model;
    faisca::PocketVoice voice;
    faisca::ThreadPool pool;
    faisca::PocketRunner runner;
    const auto t0 = std::chrono::steady_clock::now();
    if (!model.load(positional[0], error, quantize) || !voice.load(positional[1], positional[2], model, error) ||
        !pool.start(threads, error) || !runner.init(model, pool, error))
    {
        std::fprintf(stderr, "faisca speak: %s\n", error.c_str());
        return 1;
    }
    runner.set_seed(seed);
    const auto t1 = std::chrono::steady_clock::now();
    faisca::Array<float> samples;
    if (!runner.synthesize(voice, text, std::strlen(text), samples, error))
    {
        std::fprintf(stderr, "faisca speak: %s\n", error.c_str());
        return 1;
    }
    const auto t2 = std::chrono::steady_clock::now();
    const uint32_t rate = model.hparams().sample_rate;
    if (!faisca::write_wav(out, samples.data(), samples.size(), rate, error))
    {
        std::fprintf(stderr, "faisca speak: %s\n", error.c_str());
        return 1;
    }
    const double load = std::chrono::duration<double>(t1 - t0).count();
    const double run = std::chrono::duration<double>(t2 - t1).count();
    const double audio = static_cast<double>(samples.size()) / static_cast<double>(rate);
    std::fprintf(stderr, "load %.2f s, %.2f s of audio in %.2f s (%.2fx real time), %zu chunks, %zu frames, %d threads, %s -> %s\n",
                 load, audio, run, audio / run, runner.stats().chunks, runner.stats().frames, threads, quantize ? "Q8_0" : "float32",
                 out);
    return 0;
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
    if (std::strcmp(command, "chat") == 0)
    {
        return command_chat(argc, argv);
    }
    if (std::strcmp(command, "generate") == 0)
    {
        return command_generate(argc, argv);
    }
    if (std::strcmp(command, "transcribe") == 0)
    {
        return command_transcribe(argc, argv);
    }
    if (std::strcmp(command, "speak") == 0)
    {
        return command_speak(argc, argv);
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
