// Times the Runner on a prompt of token ids followed by greedy generation: the prompt in one
// forward call (batches of Runner::BATCH), then one token at a time. Prints the best of the
// runs, like llama-bench's pp and tg numbers.
//
// usage: bench_runner [--threads N] [--runs R] [--generate N] [--f32] <model.gguf> <ids.txt>

#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/runner.h"
#include "faisca/thread.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char** argv)
{
    int threads = 1;
    int runs = 3;
    int generate = 32;
    bool f32 = false;
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
        {
            threads = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
        {
            runs = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc)
        {
            generate = std::atoi(argv[++i]);
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
    if (positional.size() != 2)
    {
        std::fprintf(stderr, "usage: bench_runner [--threads N] [--runs R] [--generate N] [--f32] <model.gguf> <ids.txt>\n");
        return 2;
    }
    std::vector<int32_t> ids;
    std::ifstream stream(positional[1]);
    int32_t id = 0;
    while (stream >> id)
    {
        ids.push_back(id);
    }
    if (ids.empty())
    {
        std::fprintf(stderr, "no token ids in %s\n", positional[1]);
        return 2;
    }

    faisca::Error error;
    faisca::GgufFile file;
    faisca::LlamaModel model;
    faisca::ThreadPool pool;
    faisca::Runner runner;
    const auto t0 = std::chrono::steady_clock::now();
    if (!file.open(positional[0], error) || !pool.start(threads, error) || !model.load(file, error, f32, &pool) ||
        !runner.init(model, static_cast<uint32_t>(ids.size() + generate + 1), pool, error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const double load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double best_prompt = 0.0;
    double best_generate = 0.0;
    faisca::Array<float> logits;
    for (int run = 0; run < runs; ++run)
    {
        runner.reset();
        const auto p0 = std::chrono::steady_clock::now();
        if (!runner.forward(ids.data(), ids.size(), logits, error))
        {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        const auto p1 = std::chrono::steady_clock::now();
        int32_t next = 0;
        for (int g = 0; g < generate; ++g)
        {
            next = 0;
            for (size_t i = 1; i < logits.size(); ++i)
            {
                if (logits[i] > logits[static_cast<size_t>(next)])
                {
                    next = static_cast<int32_t>(i);
                }
            }
            if (!runner.forward(&next, 1, logits, error))
            {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
        }
        const auto p2 = std::chrono::steady_clock::now();
        const double prompt = static_cast<double>(ids.size()) / std::chrono::duration<double>(p1 - p0).count();
        const double gen = static_cast<double>(generate) / std::chrono::duration<double>(p2 - p1).count();
        best_prompt = prompt > best_prompt ? prompt : best_prompt;
        best_generate = gen > best_generate ? gen : best_generate;
        std::printf("run %d: prompt %.1f tok/s, generation %.1f tok/s\n", run + 1, prompt, gen);
    }
    std::printf("%s: %zu prompt tokens then %d, %d threads, load %.2f s, best prompt %.1f tok/s, best generation %.1f tok/s\n",
                positional[0], ids.size(), generate, threads, load, best_prompt, best_generate);
    return 0;
}
