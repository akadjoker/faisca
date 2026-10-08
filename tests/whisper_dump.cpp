// Runs the faisca Whisper runner on a 16 kHz WAV and writes every traced tensor in the format of
// whisper_layer_dump, then the greedy tokens.
//
// usage: whisper_dump [--language en] [--threads N] [--f32] <model.bin> <audio.wav> <out.bin>

#include "faisca/audio.h"
#include "faisca/thread.h"
#include "faisca/whisper_model.h"
#include "faisca/whisper_runner.h"

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
    size_t n_state = 0;
    size_t n_audio_ctx = 0;
    size_t n_vocab = 0;
};

void trace(void* user, const char* name, int layer, const float* data, size_t count)
{
    Capture& capture = *static_cast<Capture*>(user);
    std::string key = name;
    if (layer >= 0)
    {
        key += "-" + std::to_string(layer);
    }
    const bool decoder = name[0] == 'd' || std::strcmp(name, "logits") == 0;
    if (decoder)
    {
        key += "@" + std::to_string(capture.step);
    }
    Traced traced;
    if (std::strcmp(name, "mel") == 0)
    {
        traced.rows = count / (2 * capture.n_audio_ctx);
    }
    else if (std::strcmp(name, "conv1") == 0 || std::strcmp(name, "embd_conv") == 0)
    {
        traced.rows = capture.n_state; // channel major
    }
    else if (std::strcmp(name, "logits") == 0)
    {
        traced.rows = count / capture.n_vocab;
    }
    else
    {
        traced.rows = count / capture.n_state / (std::strstr(name, "ffn_up") != nullptr || std::strstr(name, "ffn_gelu") != nullptr ? 4 : 1);
    }
    traced.cols = count / traced.rows;
    traced.values.assign(data, data + count);
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

}

int main(int argc, char** argv)
{
    const char* language = "en";
    int threads = 4;
    bool quantize = true;
    while (argc > 1 && argv[1][0] == '-')
    {
        if (std::strcmp(argv[1], "--f32") == 0)
        {
            quantize = false;
            argv += 1;
            argc -= 1;
            continue;
        }
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
        else
        {
            std::fprintf(stderr, "whisper_dump: unknown option '%s'\n", argv[1]);
            return 2;
        }
    }
    if (argc != 4)
    {
        std::fprintf(stderr, "usage: whisper_dump [--language en] [--threads N] [--f32] <model.bin> <audio.wav> <out.bin>\n");
        return 2;
    }
    faisca::Error error;
    faisca::WhisperModel model;
    faisca::ThreadPool pool;
    faisca::WhisperRunner runner;
    faisca::Array<float> samples;
    if (!model.load(argv[1], error, quantize) || !pool.start(threads, error) || !runner.init(model, pool, error) ||
        !faisca::read_wav(argv[2], samples, error))
    {
        std::fprintf(stderr, "whisper_dump: %s\n", error.c_str());
        return 1;
    }
    const int32_t language_id = faisca::WhisperModel::language_id(language);
    if (language_id < 0)
    {
        std::fprintf(stderr, "whisper_dump: unknown language '%s'\n", language);
        return 1;
    }
    Capture capture;
    capture.n_state = static_cast<size_t>(model.hparams().n_audio_state);
    capture.n_audio_ctx = static_cast<size_t>(model.hparams().n_audio_ctx);
    capture.n_vocab = static_cast<size_t>(model.hparams().n_vocab);
    runner.set_trace(trace, &capture);

    faisca::Array<int32_t> tokens;
    if (!runner.transcribe_audio(samples.data(), samples.size(), language_id, tokens, error))
    {
        std::fprintf(stderr, "whisper_dump: %s\n", error.c_str());
        return 1;
    }

    FILE* out = std::fopen(argv[3], "wb");
    if (out == nullptr)
    {
        std::fprintf(stderr, "whisper_dump: cannot write '%s'\n", argv[3]);
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
    std::fprintf(stderr, "whisper_dump: %zu tensors written, %d decode steps\n", capture.order.size(), capture.step);

    std::printf("tokens:");
    std::string text;
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        std::printf(" %d", tokens[i]);
        size_t length = 0;
        const char* bytes = model.token(tokens[i], length);
        text.append(bytes, length);
    }
    std::printf(" %d\ntext:%s\n", model.tokens().eot, text.c_str());
    return 0;
}
