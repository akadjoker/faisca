// Times the CPU or GPU runner on token ids followed by greedy generation: the prompt in one
// forward call (batches of Runner::BATCH), then one token at a time. Prints the best of the
// runs, like llama-bench's pp and tg numbers.
//
// usage: bench_runner [--threads N] [--runs R] [--generate N] [--f32] [--kv-f32] [--vulkan | --cuda] <model.gguf> <ids.txt>

#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/runner.h"
#include "faisca/thread.h"
#if defined(FAISCA_VULKAN)
#include "faisca/vulkan_runner.h"
#endif
#if defined(FAISCA_CUDA)
#include "faisca/cuda_runner.h"
#endif

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
    bool vulkan = false;
    bool cuda = false;
    bool kv_f16 = true;
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
        else if (std::strcmp(argv[i], "--vulkan") == 0)
        {
            vulkan = true;
        }
        else if (std::strcmp(argv[i], "--cuda") == 0)
        {
            cuda = true;
        }
        else if (std::strcmp(argv[i], "--kv-f32") == 0)
        {
            kv_f16 = false;
        }
        else
        {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() != 2)
    {
        std::fprintf(stderr, "usage: bench_runner [--threads N] [--runs R] [--generate N] [--f32] [--kv-f32] [--vulkan | --cuda] <model.gguf> <ids.txt>\n");
        return 2;
    }
    if (threads < 1 || runs < 1 || generate < 1)
    {
        std::fprintf(stderr, "threads, runs and generate must be positive\n");
        return 2;
    }
    if (vulkan && f32)
    {
        std::fprintf(stderr, "--f32 is not supported by the Vulkan runner\n");
        return 2;
    }
    if (vulkan && cuda)
    {
        std::fprintf(stderr, "choose only one GPU backend\n");
        return 2;
    }
#if !defined(FAISCA_CUDA)
    if (cuda)
    {
        std::fprintf(stderr, "bench_runner: built without CUDA\n");
        return 1;
    }
#endif
#if !defined(FAISCA_VULKAN)
    if (vulkan)
    {
        std::fprintf(stderr, "bench_runner: built without the Vulkan backend\n");
        return 1;
    }
#endif
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
#if defined(FAISCA_VULKAN)
    faisca::VulkanDevice device;
    faisca::VulkanRunner vulkan_runner;
#endif
#if defined(FAISCA_CUDA)
    faisca::CudaDevice cuda_device;
    faisca::CudaRunner cuda_runner;
#endif
    const uint32_t n_ctx = static_cast<uint32_t>(ids.size() + static_cast<size_t>(generate) + 1);
    const auto t0 = std::chrono::steady_clock::now();
    if (!file.open(positional[0], error))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
#if defined(FAISCA_VULKAN)
    if (vulkan && (!device.init(error) || !vulkan_runner.init(device, file, n_ctx, error)))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
#endif
#if defined(FAISCA_CUDA)
    if (cuda && (!cuda_device.init(error) || !cuda_runner.init(cuda_device, file, n_ctx, error, f32, kv_f16)))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
#endif
    if (!vulkan && !cuda && (!pool.start(threads, error) || !model.load(file, error, f32, &pool) ||
                    !runner.init(model, n_ctx, pool, error, kv_f16)))
    {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    const double load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#if defined(FAISCA_CUDA)
    if (cuda)
    {
        std::printf("CUDA device %s, weights %zu bytes, KV cache %s\n",
                    cuda_device.device_name(), cuda_runner.weight_bytes(), kv_f16 ? "f16" : "float32");
    }
#endif
#if defined(FAISCA_VULKAN)
    if (vulkan)
    {
        std::printf("Vulkan device %s, weights %zu bytes, KV cache float32\n",
                    device.device_name(), vulkan_runner.weight_bytes());
    }
#endif
    const auto run_forward = [&](const int32_t* tokens, size_t count, faisca::Array<float>& out) -> bool
    {
#if defined(FAISCA_CUDA)
        if (cuda)
        {
            return cuda_runner.forward(tokens, count, out, error);
        }
#endif
#if defined(FAISCA_VULKAN)
        if (vulkan)
        {
            return vulkan_runner.forward(tokens, count, out, error);
        }
#endif
        return runner.forward(tokens, count, out, error);
    };

    double best_prompt = 0.0;
    double best_generate = 0.0;
    faisca::Array<float> logits;
    for (int run = 0; run < runs; ++run)
    {
#if defined(FAISCA_CUDA)
        if (cuda)
        {
            cuda_runner.reset();
        }
        else
#endif
#if defined(FAISCA_VULKAN)
        if (vulkan)
        {
            vulkan_runner.reset();
        }
        else
#endif
        {
            runner.reset();
        }
        const auto p0 = std::chrono::steady_clock::now();
        if (!run_forward(ids.data(), ids.size(), logits))
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
            if (!run_forward(&next, 1, logits))
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
    if (vulkan || cuda)
    {
        std::printf("%s: %zu prompt tokens then %d, %s, load %.2f s, best prompt %.1f tok/s, best generation %.1f tok/s\n",
                    positional[0], ids.size(), generate, cuda ? "CUDA" : "Vulkan", load, best_prompt, best_generate);
    }
    else
    {
        std::printf("%s: %zu prompt tokens then %d, %d threads, load %.2f s, best prompt %.1f tok/s, best generation %.1f tok/s\n",
                    positional[0], ids.size(), generate, threads, load, best_prompt, best_generate);
    }
    return 0;
}
