#include "llama.h"

#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

// usage: llama_layer_dump [--kv-f16] <model.gguf> <out.bin> <generate N> <token ids...>
//
// Runs the prompt with the KV cache in float32, without flash attention and
// with one thread, writes the named intermediate tensors to out.bin, then
// prints N greedily generated tokens. Record: u32 name length, name, u32 dims,
// u64 ne[dims], float32 data.

namespace
{

const std::set<std::string> WANTED = {
    "inp_embd", "inp_scaled", "attn_norm", "Qcur", "Kcur", "Vcur", "Qcur_normed", "Kcur_normed", "attn_out",
    "attn_post_norm", "sa_out", "ffn_inp", "ffn_norm", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_gelu", "ffn_out",
    "ffn_post_norm", "l_out", "result_norm", "result_output",
};

struct Capture
{
    FILE* out = nullptr;
    bool recording = true;
    std::map<std::string, int> seen;
    std::vector<uint8_t> buffer;
    size_t records = 0;
};

std::string base_name(const char* name)
{
    std::string text(name);
    const size_t dash = text.rfind('-');
    if (dash != std::string::npos)
    {
        bool digits = dash + 1 < text.size();
        for (size_t i = dash + 1; i < text.size(); ++i)
        {
            digits = digits && text[i] >= '0' && text[i] <= '9';
        }
        if (digits)
        {
            text.resize(dash);
        }
    }
    return text;
}

bool eval_callback(ggml_tensor* t, bool ask, void* user_data)
{
    Capture* capture = static_cast<Capture*>(user_data);
    if (!capture->recording)
    {
        return false;
    }
    const bool wanted = WANTED.count(base_name(t->name)) != 0;
    if (ask)
    {
        return wanted;
    }
    if (!wanted)
    {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t))
    {
        std::fprintf(stderr, "llama_layer_dump: skipping %s (%s, %s)\n", t->name, ggml_type_name(t->type),
                     ggml_is_contiguous(t) ? "contiguous" : "not contiguous");
        return true;
    }

    std::string name(t->name);
    const int times = ++capture->seen[name];
    if (times > 1)
    {
        name += "#" + std::to_string(times);
    }

    const size_t bytes = ggml_nbytes(t);
    capture->buffer.resize(bytes);
    ggml_backend_tensor_get(t, capture->buffer.data(), 0, bytes);

    uint32_t dims = static_cast<uint32_t>(ggml_n_dims(t));
    const uint32_t name_length = static_cast<uint32_t>(name.size());
    std::fwrite(&name_length, 4, 1, capture->out);
    std::fwrite(name.data(), 1, name.size(), capture->out);
    std::fwrite(&dims, 4, 1, capture->out);
    for (uint32_t d = 0; d < dims; ++d)
    {
        const uint64_t ne = static_cast<uint64_t>(t->ne[d]);
        std::fwrite(&ne, 8, 1, capture->out);
    }
    std::fwrite(capture->buffer.data(), 1, bytes, capture->out);
    ++capture->records;
    return true;
}

void quiet(ggml_log_level level, const char* text, void*)
{
    if (level == GGML_LOG_LEVEL_ERROR)
    {
        std::fputs(text, stderr);
    }
}

}

int main(int argc, char** argv)
{
    bool kv_f16 = false;
    while (argc > 1 && argv[1][0] == '-')
    {
        if (std::strcmp(argv[1], "--kv-f16") == 0)
        {
            kv_f16 = true;
        }
        else
        {
            std::fprintf(stderr, "llama_layer_dump: unknown option '%s'\n", argv[1]);
            return 2;
        }
        ++argv;
        --argc;
    }
    if (argc < 5)
    {
        std::fprintf(stderr, "usage: llama_layer_dump [--kv-f16] <model.gguf> <out.bin> <generate N> <token ids...>\n");
        return 2;
    }

    const int generate = std::atoi(argv[3]);
    std::vector<llama_token> prompt;
    for (int i = 4; i < argc; ++i)
    {
        prompt.push_back(std::atoi(argv[i]));
    }

    Capture capture;
    capture.out = std::fopen(argv[2], "wb");
    if (capture.out == nullptr)
    {
        std::fprintf(stderr, "llama_layer_dump: cannot write '%s'\n", argv[2]);
        return 1;
    }

    llama_log_set(quiet, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr)
    {
        std::fprintf(stderr, "llama_layer_dump: cannot load '%s'\n", argv[1]);
        return 1;
    }

    const uint32_t n_ctx = static_cast<uint32_t>(prompt.size() + static_cast<size_t>(generate) + 1);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx < 32 ? 32 : n_ctx;
    ctx_params.n_batch = static_cast<uint32_t>(prompt.size());
    ctx_params.n_ubatch = static_cast<uint32_t>(prompt.size());
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.type_k = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ctx_params.type_v = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.cb_eval = eval_callback;
    ctx_params.cb_eval_user_data = &capture;
    ctx_params.no_perf = true;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr)
    {
        std::fprintf(stderr, "llama_layer_dump: cannot create the context\n");
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_init(static_cast<int32_t>(prompt.size()), 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i)
    {
        batch.token[i] = prompt[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = static_cast<int32_t>(prompt.size());

    if (llama_decode(ctx, batch) != 0)
    {
        std::fprintf(stderr, "llama_layer_dump: decode failed\n");
        return 1;
    }
    capture.recording = false;
    std::fclose(capture.out);
    std::fprintf(stderr, "llama_layer_dump: %zu tensors written\n", capture.records);

    std::printf("generated:");
    llama_pos pos = static_cast<llama_pos>(prompt.size());
    for (int g = 0; g < generate; ++g)
    {
        const float* logits = llama_get_logits_ith(ctx, -1);
        int32_t best = 0;
        for (int32_t t = 1; t < n_vocab; ++t)
        {
            if (logits[t] > logits[best])
            {
                best = t;
            }
        }
        std::printf(" %" PRId32, best);

        batch.n_tokens = 1;
        batch.token[0] = best;
        batch.pos[0] = pos++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0)
        {
            std::fprintf(stderr, "llama_layer_dump: decode failed\n");
            return 1;
        }
    }
    std::printf("\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
