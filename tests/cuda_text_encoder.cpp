// CosmosTextCudaEncoder against a float32 PyTorch run of tools/omnidreams/text_embed.py
// (--float32) of the same prompt, next to the bf16 PyTorch run; per region and per layer.
// Usage: cuda_text_encoder <model dir> <prompt file> <float32 .pt> [bf16 .pt]
#include "faisca/text_encoder_cuda.h"
#include "faisca/torch_ckpt.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

float bf16(uint16_t v)
{
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: cuda_text_encoder <model dir> <prompt file> <float32 .pt> [bf16 .pt]\n");
        return 2;
    }
    std::string prompt;
    {
        FILE* file = std::fopen(argv[2], "rb");
        if (file == nullptr)
        {
            std::printf("cannot open %s\n", argv[2]);
            return 1;
        }
        char buffer[4096];
        size_t n;
        while ((n = std::fread(buffer, 1, sizeof(buffer), file)) > 0)
        {
            prompt.append(buffer, n);
        }
        std::fclose(file);
        while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == ' ' || prompt.back() == '\r'))
        {
            prompt.pop_back();
        }
        while (!prompt.empty() && (prompt.front() == ' ' || prompt.front() == '\n'))
        {
            prompt.erase(prompt.begin());
        }
    }
    faisca::Error error;
    faisca::CkptFile reference, low;
    if (!reference.open(argv[3], error) || (argc > 4 && !low.open(argv[4], error)))
    {
        std::printf("%s\n", error.message);
        return 1;
    }
    const faisca::CkptTensor* ref = reference.find("");
    const faisca::CkptTensor* ref_bf16 = argc > 4 ? low.find("") : nullptr;
    const size_t count = size_t(faisca::CosmosTextCudaEncoder::TOKENS) * faisca::CosmosTextCudaEncoder::OUT_WIDTH;
    if (ref == nullptr || ref->dtype != faisca::CkptDtype::F32 || ref->n_elements != count)
    {
        std::printf("%s: expected one float32 tensor of 512 x 100352\n", argv[3]);
        return 1;
    }
    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::printf("cuda_text_encoder: skipped, %s\n", error.message);
        return 77;
    }
    faisca::CosmosTextCudaEncoder encoder;
    faisca::Array<uint16_t> out;
    auto start = std::chrono::steady_clock::now();
    if (!encoder.init(device, argv[1], error))
    {
        std::printf("cuda_text_encoder: %s\n", error.message);
        return 1;
    }
    const double init_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (int run = 0; run < 2; ++run)
    {
        start = std::chrono::steady_clock::now();
        if (!encoder.encode(prompt.c_str(), out, error))
        {
            std::printf("cuda_text_encoder: %s\n", error.message);
            return 1;
        }
        std::printf("init %.1f s, encode %.2f s\n", init_s,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    size_t prompt_tokens = 0;
    for (size_t i = 0; i < encoder.tokens().size(); ++i)
    {
        prompt_tokens += encoder.tokens()[i] != encoder.tokens()[faisca::CosmosTextCudaEncoder::TOKENS - 1] ? 1 : 0;
    }
    const float* r = reinterpret_cast<const float*>(ref->data);
    const uint16_t* b = ref_bf16 != nullptr ? reinterpret_cast<const uint16_t*>(ref_bf16->data) : nullptr;
    const uint32_t W = faisca::CosmosTextCudaEncoder::WIDTH, OUT = faisca::CosmosTextCudaEncoder::OUT_WIDTH;
    const auto region = [&](size_t t0, size_t t1, uint32_t l0, uint32_t l1, double& ours, double& theirs) {
        double e = 0, eb = 0, n = 0;
        for (size_t t = t0; t < t1; ++t)
        {
            for (size_t c = size_t(l0) * W; c < size_t(l1) * W; ++c)
            {
                const double x = r[t * OUT + c];
                const double d = bf16(out[t * OUT + c]) - x;
                e += d * d;
                n += x * x;
                if (b != nullptr)
                {
                    const double db = bf16(b[t * OUT + c]) - x;
                    eb += db * db;
                }
            }
        }
        ours = std::sqrt(e / n);
        theirs = std::sqrt(eb / n);
    };
    double ours, theirs;
    region(0, 512, 0, 28, ours, theirs);
    std::printf("%zu prompt tokens; rel L2 to float32: all ours %.3e torch bf16 %.3e\n", prompt_tokens, ours, theirs);
    region(0, prompt_tokens, 0, 28, ours, theirs);
    std::printf("  prompt tokens: ours %.3e, torch bf16 %.3e\n", ours, theirs);
    region(prompt_tokens, 512, 0, 28, ours, theirs);
    std::printf("  padding:       ours %.3e, torch bf16 %.3e\n", ours, theirs);
    std::printf("  per layer (prompt tokens, ours / torch bf16):");
    for (uint32_t l = 0; l < 28; l += 3)
    {
        region(0, prompt_tokens, l, l + 1, ours, theirs);
        std::printf(" %u: %.1e/%.1e", l, ours, theirs);
    }
    region(0, prompt_tokens, 27, 28, ours, theirs);
    std::printf(" 27: %.1e/%.1e\n", ours, theirs);
    return 0;
}
