// Runs the faisca Pocket TTS runner and writes the tensors pockettts.cpp dumps with --dump
// (prompt-ids, text-embed, step0-hidden, step0-latent, eos, latents, output-audio) in its
// format: int32 ndims, int32 shape, float32 data, plus the per layer tensors of the first
// prefill (pre_dbg_*) and of step 0 (dbg_*) that the patched reference writes with
// PT_DUMP_NODES=<dir>.
//
// usage: pocket_dump [--threads N] [--f32] [--seed S] <model.gguf> <voices.gguf> <voice> <text> <out dir>

#include "faisca/pocket_model.h"
#include "faisca/pocket_runner.h"
#include "faisca/thread.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{

struct Capture
{
    std::string dir;
    bool hidden_done = false;
    size_t dim = 0;
    size_t ldim = 0;
    std::chrono::steady_clock::time_point first_hidden;
    std::chrono::steady_clock::time_point last_hidden;
    std::chrono::steady_clock::time_point text_embed;
    std::chrono::steady_clock::time_point latents;
};

void write(const std::string& path, const std::vector<int32_t>& shape, const float* data, size_t count)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(1);
    }
    const int32_t nd = static_cast<int32_t>(shape.size());
    std::fwrite(&nd, 4, 1, f);
    std::fwrite(shape.data(), 4, shape.size(), f);
    std::fwrite(data, 4, count, f);
    std::fclose(f);
}

void trace(void* user, const char* name, int layer, const float* data, size_t count)
{
    Capture& c = *static_cast<Capture*>(user);
    std::string key = name;
    std::vector<int32_t> shape;
    const auto now = std::chrono::steady_clock::now();
    if (key == "hidden")
    {
        if (!c.hidden_done)
        {
            c.first_hidden = now;
        }
        c.last_hidden = now;
    }
    else if (key == "text-embed" && !c.hidden_done)
    {
        c.text_embed = now;
    }
    else if (key == "latents")
    {
        c.latents = now;
    }
    if (key == "prompt-ids")
    {
        std::fprintf(stderr, "chunk %d: %zu tokens\n", layer, count);
        if (layer != 0)
        {
            return;
        }
        layer = -1;
    }
    if (layer >= 0)
    {
        // per layer tensors of the first prefill (layer + 100) and of step 0, named as the
        // PT_DUMP_NODES patch of the reference writes them
        if (c.hidden_done)
        {
            return;
        }
        const std::string prefix = layer >= 100 ? "/pre_dbg_" : "/dbg_";
        write(c.dir + prefix + key + "_" + std::to_string(layer % 100) + ".bin", {static_cast<int32_t>(count)}, data, count);
        return;
    }
    if (key == "hidden")
    {
        if (c.hidden_done)
        {
            return;
        }
        c.hidden_done = true;
        key = "step0-hidden";
    }
    if (key == "text-embed")
    {
        shape = {static_cast<int32_t>(count / c.dim), static_cast<int32_t>(c.dim)};
    }
    else if (key == "latents")
    {
        shape = {static_cast<int32_t>(count / c.ldim), static_cast<int32_t>(c.ldim)};
    }
    else
    {
        shape = {static_cast<int32_t>(count)};
    }
    write(c.dir + "/" + key + ".bin", shape, data, count);
}

}

int main(int argc, char** argv)
{
    int threads = 1;
    bool f32 = false;
    uint32_t seed = 0;
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
        {
            threads = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        }
        else if (std::strcmp(argv[i], "--f32") == 0)
        {
            f32 = true;
        }
        else
        {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() != 5)
    {
        std::fprintf(stderr, "usage: pocket_dump [--threads N] [--f32] [--seed S] <model.gguf> <voices.gguf> <voice> <text> <out dir>\n");
        return 1;
    }
    faisca::Error error;
    faisca::PocketModel model;
    const auto t0 = std::chrono::steady_clock::now();
    if (!model.load(positional[0], error, !f32))
    {
        std::fprintf(stderr, "load: %s\n", error.message);
        return 1;
    }
    faisca::PocketVoice voice;
    if (!voice.load(positional[1], positional[2], model, error))
    {
        std::fprintf(stderr, "voice: %s\n", error.message);
        return 1;
    }
    faisca::ThreadPool pool;
    if (!pool.start(threads, error))
    {
        std::fprintf(stderr, "threads: %s\n", error.message);
        return 1;
    }
    faisca::PocketRunner runner;
    if (!runner.init(model, pool, error))
    {
        std::fprintf(stderr, "init: %s\n", error.message);
        return 1;
    }
    Capture capture;
    capture.dir = positional[4];
    capture.dim = model.hparams().dim;
    capture.ldim = model.hparams().ldim;
    runner.set_trace(trace, &capture);
    runner.set_seed(seed);
    const auto t1 = std::chrono::steady_clock::now();
    faisca::Array<float> samples;
    if (!runner.synthesize(voice, positional[3], std::strlen(positional[3]), samples, error))
    {
        std::fprintf(stderr, "synthesize: %s\n", error.message);
        return 1;
    }
    const auto t2 = std::chrono::steady_clock::now();
    write(capture.dir + "/output-audio.bin", {static_cast<int32_t>(samples.size())}, samples.data(), samples.size());
    const double load_s = std::chrono::duration<double>(t1 - t0).count();
    const double synth_s = std::chrono::duration<double>(t2 - t1).count();
    const double audio_s = static_cast<double>(samples.size()) / model.hparams().sample_rate;
    std::printf("load %.2f s, %zu chunks, %zu frames, %zu steps, %.2f s of audio in %.2f s (%.1fx real time)\n", load_s,
                runner.stats().chunks, runner.stats().frames, runner.stats().steps, audio_s, synth_s, audio_s / synth_s);
    const auto seconds = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
    {
        return std::chrono::duration<double>(b - a).count();
    };
    std::printf("first chunk: prefill %.3f s, steps %.3f s (%.2f ms per step), decode %.3f s (%.2f ms per frame)\n",
                seconds(t1, capture.text_embed), seconds(capture.text_embed, capture.last_hidden),
                1000.0 * seconds(capture.text_embed, capture.last_hidden) / static_cast<double>(runner.stats().steps),
                seconds(capture.latents, t2), 1000.0 * seconds(capture.latents, t2) / static_cast<double>(runner.stats().frames));
    return 0;
}
