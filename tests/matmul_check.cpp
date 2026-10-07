#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// usage: matmul_check <model.gguf> <llama_dump.bin> [--rel 1e-5]
//
// Feeds every matrix of every block with the exact input llama.cpp computed
// (from a llama_layer_dump file) and compares faisca's product with the output
// llama.cpp computed. This checks the kernels alone: a full forward pass with
// quantized activations drifts because 1e-7 differences before the
// quantization round to different integers.

namespace
{

struct Tensor
{
    std::vector<uint64_t> shape;
    std::vector<float> values;
};

std::map<std::string, Tensor> read_dump(const char* path)
{
    std::map<std::string, Tensor> tensors;
    FILE* file = std::fopen(path, "rb");
    if (file == nullptr)
    {
        return tensors;
    }
    for (;;)
    {
        uint32_t name_length = 0;
        if (std::fread(&name_length, 4, 1, file) != 1)
        {
            break;
        }
        std::string name(name_length, '\0');
        uint32_t dims = 0;
        if (std::fread(name.data(), 1, name_length, file) != name_length || std::fread(&dims, 4, 1, file) != 1)
        {
            break;
        }
        Tensor tensor;
        tensor.shape.resize(dims);
        uint64_t count = 1;
        for (uint32_t d = 0; d < dims; ++d)
        {
            if (std::fread(&tensor.shape[d], 8, 1, file) != 1)
            {
                break;
            }
            count *= tensor.shape[d];
        }
        tensor.values.resize(count);
        if (std::fread(tensor.values.data(), 4, count, file) != count)
        {
            break;
        }
        tensors[name] = std::move(tensor);
    }
    std::fclose(file);
    return tensors;
}

int g_failures = 0;
int g_checks = 0;
double g_worst = 0.0;
std::string g_worst_name;

void check(const faisca::Weight& w, const Tensor& input, const Tensor& output, const char* label, double limit,
           bool verbose)
{
    const size_t n_tokens = input.values.size() / w.n_in;
    if (input.values.size() != n_tokens * w.n_in || output.values.size() != n_tokens * w.n_out)
    {
        std::printf("  %-28s shapes do not match the weight [%zu x %zu]\n", label, w.n_in, w.n_out);
        ++g_failures;
        return;
    }

    std::vector<uint8_t> xq((w.n_in / faisca::Q8_0_BLOCK) * faisca::Q8_0_BLOCK_BYTES);
    std::vector<float> out(w.n_out);
    double error = 0.0;
    double norm = 0.0;
    for (size_t t = 0; t < n_tokens; ++t)
    {
        const float* x = input.values.data() + t * w.n_in;
        if (w.type == faisca::GgmlType::Q8_0)
        {
            faisca::quantize_q8_0(x, w.n_in, xq.data());
            faisca::matvec_q8_0(w.data, xq.data(), w.n_in, w.n_out, out.data());
        }
        else
        {
            faisca::matvec(w.f32(), x, w.n_in, w.n_out, out.data());
        }
        const float* expected = output.values.data() + t * w.n_out;
        for (size_t o = 0; o < w.n_out; ++o)
        {
            const double diff = static_cast<double>(out[o]) - expected[o];
            error += diff * diff;
            norm += static_cast<double>(expected[o]) * expected[o];
        }
    }
    const double rel = norm > 0.0 ? std::sqrt(error / norm) : std::sqrt(error);
    ++g_checks;
    const bool bad = rel > limit;
    g_failures += bad ? 1 : 0;
    if (rel > g_worst)
    {
        g_worst = rel;
        g_worst_name = label;
    }
    if (bad || verbose)
    {
        std::printf("  %-28s %zu tokens  rel L2 %.3e%s\n", label, n_tokens, rel, bad ? "  FAIL" : "");
    }
}

const Tensor* find(const std::map<std::string, Tensor>& dump, const std::string& name)
{
    const auto it = dump.find(name);
    return it == dump.end() ? nullptr : &it->second;
}

}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: matmul_check <model.gguf> <llama_dump.bin> [--rel 1e-5] [--verbose]\n");
        return 2;
    }
    double limit = 1e-5;
    bool verbose = false;
    for (int i = 3; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--rel") == 0 && i + 1 < argc)
        {
            limit = std::atof(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--verbose") == 0)
        {
            verbose = true;
        }
    }

    faisca::Error error;
    faisca::GgufFile file;
    faisca::LlamaModel model;
    if (!file.open(argv[1], error) || !model.load(file, error))
    {
        std::fprintf(stderr, "matmul_check: %s\n", error.c_str());
        return 1;
    }

    const std::map<std::string, Tensor> dump = read_dump(argv[2]);
    if (dump.empty())
    {
        std::fprintf(stderr, "matmul_check: cannot read '%s'\n", argv[2]);
        return 1;
    }

    const uint32_t n_layer = model.hparams().n_layer;
    for (uint32_t l = 0; l < n_layer; ++l)
    {
        const faisca::LlamaLayer& layer = model.layer(l);
        const std::string suffix = "-" + std::to_string(l);
        const Tensor* attn_norm = find(dump, "attn_norm" + suffix);
        const Tensor* q = find(dump, "Qcur" + suffix);
        const Tensor* k = find(dump, "Kcur" + suffix);
        const Tensor* v = find(dump, "Vcur" + suffix);
        const Tensor* ffn_norm = find(dump, "ffn_norm" + suffix);
        const Tensor* gate = find(dump, "ffn_gate" + suffix);
        const Tensor* up = find(dump, "ffn_up" + suffix);
        const Tensor* swiglu = find(dump, "ffn_swiglu" + suffix);
        const Tensor* down = find(dump, "ffn_out" + suffix);
        if (attn_norm == nullptr || q == nullptr || k == nullptr || v == nullptr || ffn_norm == nullptr ||
            gate == nullptr || up == nullptr || swiglu == nullptr || down == nullptr)
        {
            std::printf("  block %u: tensors missing in the dump\n", l);
            ++g_failures;
            continue;
        }
        check(layer.wq, *attn_norm, *q, ("blk." + std::to_string(l) + ".attn_q").c_str(), limit, verbose);
        check(layer.wk, *attn_norm, *k, ("blk." + std::to_string(l) + ".attn_k").c_str(), limit, verbose);
        check(layer.wv, *attn_norm, *v, ("blk." + std::to_string(l) + ".attn_v").c_str(), limit, verbose);
        check(layer.ffn_gate, *ffn_norm, *gate, ("blk." + std::to_string(l) + ".ffn_gate").c_str(), limit, verbose);
        check(layer.ffn_up, *ffn_norm, *up, ("blk." + std::to_string(l) + ".ffn_up").c_str(), limit, verbose);
        check(layer.ffn_down, *swiglu, *down, ("blk." + std::to_string(l) + ".ffn_down").c_str(), limit, verbose);
    }

    const Tensor* result_norm = find(dump, "result_norm");
    const Tensor* result_output = find(dump, "result_output");
    if (result_norm != nullptr && result_output != nullptr)
    {
        check(model.output(), *result_norm, *result_output, "output", limit, verbose);
    }

    std::printf("%s %s: %d matrices checked with llama.cpp's inputs, worst rel L2 %.3e (%s), %d failures\n",
                g_failures == 0 ? "ok  " : "FAIL", argv[1], g_checks, g_worst, g_worst_name.c_str(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
