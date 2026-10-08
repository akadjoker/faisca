// Runs whisper.cpp (the patched copy in /home/user/ref/whisper.cpp: eval callback and named
// tensors) on a 16 kHz mono WAV and writes every named tensor plus the mel window, then the
// greedy tokens.
//
// usage: whisper_layer_dump [--language en] [--threads N] [--max-tokens N] <model.bin> <audio.wav> <out.bin>
//
// Dump format: {u32 name_length, name, u32 dims = 2, u64 ne[2] = {rows, cols}, f32 data} per
// tensor; decoder tensors carry "@step" with the decode call they came from.

#include "ggml-backend.h"
#include "ggml.h"
#include "whisper.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{

struct Traced
{
    uint64_t rows = 0;
    uint64_t cols = 0;
    std::vector<float> values;
};

struct Capture
{
    std::vector<std::string> order;
    std::map<std::string, Traced> tensors;
    int step = 0;
};

bool wanted(const char* name)
{
    if (std::strstr(name, " (") != nullptr)
    {
        return false; // views and transposes inherit the name with a suffix
    }
    static const char* prefixes[] = {"mel", "conv1", "embd_conv", "e_", "cross_", "d_", "logits"};
    for (const char* prefix : prefixes)
    {
        if (std::strncmp(name, prefix, std::strlen(prefix)) == 0)
        {
            return true;
        }
    }
    return false;
}

void store(Capture& capture, const char* name, struct ggml_tensor* t);

// the mel input is never a computed node: it is taken from the im2col that reads it
bool reads_mel(struct ggml_tensor* t)
{
    return t->op == GGML_OP_IM2COL && t->src[1] != nullptr && std::strcmp(ggml_get_name(t->src[1]), "mel") == 0;
}

bool eval_callback(struct ggml_tensor* t, bool ask, void* user)
{
    const char* name = ggml_get_name(t);
    Capture& capture = *static_cast<Capture*>(user);
    if (reads_mel(t))
    {
        if (!ask)
        {
            store(capture, "mel", t->src[1]);
        }
        return true;
    }
    if (!wanted(name))
    {
        return false;
    }
    if (ask)
    {
        return true;
    }
    store(capture, name, t);
    return true;
}

void store(Capture& capture, const char* name, struct ggml_tensor* t)
{
    if (!ggml_is_contiguous(t))
    {
        std::fprintf(stderr, "whisper_layer_dump: %s is not contiguous\n", name);
        std::exit(1);
    }
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    Traced traced;
    traced.cols = static_cast<uint64_t>(t->ne[0]);
    traced.rows = n / traced.cols;
    traced.values.resize(n);
    if (t->type == GGML_TYPE_F32)
    {
        ggml_backend_tensor_get(t, traced.values.data(), 0, n * sizeof(float));
    }
    else if (t->type == GGML_TYPE_F16)
    {
        std::vector<ggml_fp16_t> half(n);
        ggml_backend_tensor_get(t, half.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i)
        {
            traced.values[i] = ggml_fp16_to_fp32(half[i]);
        }
    }
    else
    {
        std::fprintf(stderr, "whisper_layer_dump: %s has type %d\n", name, static_cast<int>(t->type));
        std::exit(1);
    }
    std::string key = name;
    const bool decoder = name[0] == 'd' || std::strcmp(name, "logits") == 0;
    if (decoder)
    {
        key += "@" + std::to_string(capture.step);
    }
    if (capture.tensors.find(key) == capture.tensors.end())
    {
        capture.order.push_back(key);
    }
    capture.tensors[key] = std::move(traced);
    if (std::strcmp(name, "logits") == 0)
    {
        capture.step++;
    }
}

bool read_wav(const char* path, std::vector<float>& samples)
{
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr)
    {
        return false;
    }
    std::vector<uint8_t> bytes;
    uint8_t buffer[65536];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), f)) > 0)
    {
        bytes.insert(bytes.end(), buffer, buffer + got);
    }
    std::fclose(f);
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
    {
        return false;
    }
    size_t pos = 12;
    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t rate = 0;
    uint16_t bits = 0;
    while (pos + 8 <= bytes.size())
    {
        uint32_t size = 0;
        std::memcpy(&size, bytes.data() + pos + 4, 4);
        const uint8_t* chunk = bytes.data() + pos + 8;
        if (std::memcmp(bytes.data() + pos, "fmt ", 4) == 0 && size >= 16)
        {
            std::memcpy(&format, chunk, 2);
            std::memcpy(&channels, chunk + 2, 2);
            std::memcpy(&rate, chunk + 4, 4);
            std::memcpy(&bits, chunk + 14, 2);
        }
        else if (std::memcmp(bytes.data() + pos, "data", 4) == 0)
        {
            if (format != 1 || bits != 16 || rate != 16000 || channels == 0)
            {
                std::fprintf(stderr, "whisper_layer_dump: need 16 kHz 16 bit PCM (format %u, %u bits, %u Hz)\n", format, bits, rate);
                return false;
            }
            const size_t frames = size / (2 * channels);
            samples.resize(frames);
            for (size_t i = 0; i < frames; ++i)
            {
                int32_t sum = 0;
                for (uint16_t c = 0; c < channels; ++c)
                {
                    int16_t v = 0;
                    std::memcpy(&v, chunk + (i * channels + c) * 2, 2);
                    sum += v;
                }
                samples[i] = static_cast<float>(sum) / (32768.0f * static_cast<float>(channels));
            }
            return true;
        }
        pos += 8 + size + (size & 1);
    }
    return false;
}

}

int main(int argc, char** argv)
{
    const char* language = "en";
    int threads = 4;
    int max_tokens = 0;
    while (argc > 1 && argv[1][0] == '-')
    {
        if (std::strcmp(argv[1], "--language") == 0 && argc > 2)
        {
            language = argv[2];
            argv += 2;
            argc -= 2;
        }
        else if (std::strcmp(argv[1], "--threads") == 0 && argc > 2)
        {
            threads = std::atoi(argv[2]);
            argv += 2;
            argc -= 2;
        }
        else if (std::strcmp(argv[1], "--max-tokens") == 0 && argc > 2)
        {
            max_tokens = std::atoi(argv[2]);
            argv += 2;
            argc -= 2;
        }
        else
        {
            std::fprintf(stderr, "whisper_layer_dump: unknown option '%s'\n", argv[1]);
            return 2;
        }
    }
    if (argc != 4)
    {
        std::fprintf(stderr, "usage: whisper_layer_dump [--language en] [--threads N] [--max-tokens N] <model.bin> <audio.wav> <out.bin>\n");
        return 2;
    }

    std::vector<float> samples;
    if (!read_wav(argv[2], samples))
    {
        std::fprintf(stderr, "whisper_layer_dump: cannot read '%s'\n", argv[2]);
        return 1;
    }

    Capture capture;
    whisper_set_eval_callback(eval_callback, &capture);

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    cparams.flash_attn = false;
    whisper_context* ctx = whisper_init_from_file_with_params(argv[1], cparams);
    if (ctx == nullptr)
    {
        std::fprintf(stderr, "whisper_layer_dump: cannot load '%s'\n", argv[1]);
        return 1;
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = language;
    params.detect_language = false;
    params.n_threads = threads;
    params.no_timestamps = true;
    params.single_segment = true;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.temperature_inc = 0.0f;
    params.max_tokens = max_tokens;
    if (whisper_full(ctx, params, samples.data(), static_cast<int>(samples.size())) != 0)
    {
        std::fprintf(stderr, "whisper_layer_dump: whisper_full failed\n");
        return 1;
    }

    FILE* out = std::fopen(argv[3], "wb");
    if (out == nullptr)
    {
        std::fprintf(stderr, "whisper_layer_dump: cannot write '%s'\n", argv[3]);
        return 1;
    }
    for (const std::string& key : capture.order)
    {
        const Traced& traced = capture.tensors[key];
        const uint32_t name_length = static_cast<uint32_t>(key.size());
        const uint32_t dims = 2;
        const uint64_t ne[2] = {traced.rows, traced.cols};
        std::fwrite(&name_length, 4, 1, out);
        std::fwrite(key.data(), 1, key.size(), out);
        std::fwrite(&dims, 4, 1, out);
        std::fwrite(ne, 8, 2, out);
        std::fwrite(traced.values.data(), 4, traced.values.size(), out);
    }
    std::fclose(out);
    std::fprintf(stderr, "whisper_layer_dump: %zu tensors written, %d decode steps\n", capture.order.size(), capture.step);

    std::printf("tokens:");
    std::string text;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int s = 0; s < n_segments; ++s)
    {
        const int n_tokens = whisper_full_n_tokens(ctx, s);
        for (int i = 0; i < n_tokens; ++i)
        {
            const whisper_token id = whisper_full_get_token_id(ctx, s, i);
            std::printf(" %d", id);
        }
        text += whisper_full_get_segment_text(ctx, s);
    }
    std::printf("\ntext:%s\n", text.c_str());
    whisper_free(ctx);
    return 0;
}
