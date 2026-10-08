// Encodes an image with llama.cpp's mtmd (the reference vision path) and writes the image
// embeddings as raw float32 (n_tokens x n_embd). Built against a llama.cpp build with
// libmtmd; the Q8_0 and AMX issues of this machine need the no AMX libraries.
//
// usage: llama_vision_dump [--trace trace.bin] [--layers N] [--threads N] <model.gguf> <mmproj.gguf> <image> <embd.bin>
//        llama_vision_dump --lm <lm.bin> <generate N> <model.gguf> <mmproj.gguf> <image> <prompt>
//
// --lm runs the language model on the prompt (text with mtmd's <__media__> marker where the
// image goes, wrapped by the caller in the chat template, BOS added), writes inp_embd, l_out
// and result_norm of the prompt (the rows of the three chunks joined, as runner_dump writes
// them) in the layer dump format, prints the token ids of the text chunks and the greedy
// tokens; the KV cache is float32, flash attention off, one thread.
//
// --trace writes the named intermediate tensors of the vision graph (clip.cpp's cb names,
// the post layer norm as post_ln) in the format of llama_layer_dump, with the flash
// attention of the encoder disabled so the keys and values stay float32 (without --trace
// the encoder keeps mtmd's default, flash attention, for the timing); --layers N keeps
// the per layer tensors of the first N layers only (a whole trace is about 8 GB).

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace
{

const std::set<std::string> WANTED = {
    "inp_raw", "patch_bias", "pos_embed", "pre_ln", "layer_inp_normed", "Qcur", "Kcur", "Vcur", "kqv_out",
    "attn_out", "ffn_inp", "ffn_inp_normed", "ffn_up", "ffn_up_b", "ffn_gelu", "ffn_down", "ffn_out", "layer_out",
    "norm_b",
};

struct Capture
{
    FILE* out = nullptr;
    int layers = 1 << 30;
    std::map<std::string, int> seen;
    std::vector<uint8_t> buffer;
    size_t records = 0;
};

const std::set<std::string> LM_WANTED = {"inp_embd", "l_out", "result_norm"};

struct LmCapture
{
    bool recording = true;
    std::vector<std::string> order;
    std::map<std::string, std::pair<size_t, std::vector<float>>> tensors;
};

std::string base_name(const char* name, int& layer)
{
    std::string text(name);
    layer = -1;
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
            layer = std::atoi(text.c_str() + dash + 1);
            text.resize(dash);
        }
    }
    return text;
}

bool eval_callback(ggml_tensor* t, bool ask, void* user_data)
{
    Capture* capture = static_cast<Capture*>(user_data);
    int layer = -1;
    std::string base = base_name(t->name, layer);
    if (base == "norm_b")
    {
        base = layer < 0 ? "post_ln" : "";
    }
    const bool wanted = WANTED.count(base) != 0 && layer < capture->layers;
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
        std::fprintf(stderr, "llama_vision_dump: skipping %s (%s, %s)\n", t->name, ggml_type_name(t->type),
                     ggml_is_contiguous(t) ? "contiguous" : "not contiguous");
        return true;
    }
    std::string name = base == "post_ln" ? base : std::string(t->name);
    const int times = ++capture->seen[name];
    if (times > 1)
    {
        name += "#" + std::to_string(times);
    }
    const size_t bytes = ggml_nbytes(t);
    capture->buffer.resize(bytes);
    ggml_backend_tensor_get(t, capture->buffer.data(), 0, bytes);
    const uint32_t dims = static_cast<uint32_t>(ggml_n_dims(t));
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

}

bool lm_eval_callback(ggml_tensor* t, bool ask, void* user_data)
{
    LmCapture* capture = static_cast<LmCapture*>(user_data);
    int layer = -1;
    const std::string base = base_name(t->name, layer);
    const bool wanted = capture->recording && LM_WANTED.count(base) != 0;
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
        std::fprintf(stderr, "llama_vision_dump: skipping %s (%s, %s)\n", t->name, ggml_type_name(t->type),
                     ggml_is_contiguous(t) ? "contiguous" : "not contiguous");
        return true;
    }
    const std::string name(t->name);
    auto it = capture->tensors.find(name);
    if (it == capture->tensors.end())
    {
        capture->order.push_back(name);
        it = capture->tensors.emplace(name, std::make_pair(static_cast<size_t>(t->ne[0]), std::vector<float>())).first;
    }
    std::vector<float>& values = it->second.second;
    const size_t count = ggml_nelements(t);
    const size_t at = values.size();
    values.resize(at + count);
    ggml_backend_tensor_get(t, values.data() + at, 0, count * sizeof(float));
    return true;
}

int run_lm(const char* out_path, int generate, const char* model_path, const char* mmproj_path, const char* image_path,
           const char* prompt)
{
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr)
    {
        std::fprintf(stderr, "cannot load %s\n", model_path);
        return 1;
    }
    LmCapture capture;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 1024;
    cparams.n_ubatch = 1024;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval = lm_eval_callback;
    cparams.cb_eval_user_data = &capture;
    cparams.no_perf = true;
    llama_context* lctx = llama_init_from_model(model, cparams);
    if (lctx == nullptr)
    {
        std::fprintf(stderr, "cannot create the context\n");
        return 1;
    }
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;
    params.n_threads = 1;
    params.warmup = false;
    params.print_timings = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mtmd_context* ctx = mtmd_init_from_file(mmproj_path, model, params);
    if (ctx == nullptr)
    {
        std::fprintf(stderr, "cannot load %s\n", mmproj_path);
        return 1;
    }
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_file(ctx, image_path, false, mtmd_helper_init_opt_default());
    if (wrapper.bitmap == nullptr)
    {
        std::fprintf(stderr, "cannot read %s\n", image_path);
        return 1;
    }
    mtmd_input_text text{prompt, std::strlen(prompt), true, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = {wrapper.bitmap};
    if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0)
    {
        std::fprintf(stderr, "mtmd_tokenize failed\n");
        return 1;
    }
    std::printf("prompt:");
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; ++i)
    {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT)
        {
            std::printf(" image");
            continue;
        }
        size_t n = 0;
        const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
        for (size_t k = 0; k < n; ++k)
        {
            std::printf(" %d", tokens[k]);
        }
    }
    std::printf("\n");
    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(ctx, lctx, chunks, 0, 0, 1024, true, &n_past) != 0)
    {
        std::fprintf(stderr, "mtmd_helper_eval_chunks failed\n");
        return 1;
    }
    capture.recording = false;
    FILE* out = std::fopen(out_path, "wb");
    if (out == nullptr)
    {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    for (const std::string& key : capture.order)
    {
        const auto& traced = capture.tensors[key];
        const uint32_t name_length = static_cast<uint32_t>(key.size());
        const uint32_t dims = 2;
        const uint64_t ne[2] = {traced.first, traced.second.size() / traced.first};
        std::fwrite(&name_length, 4, 1, out);
        std::fwrite(key.data(), 1, key.size(), out);
        std::fwrite(&dims, 4, 1, out);
        std::fwrite(ne, 8, 2, out);
        std::fwrite(traced.second.data(), 4, traced.second.size(), out);
    }
    std::fclose(out);
    std::fprintf(stderr, "llama_vision_dump: %zu tensors written, %d prompt positions\n", capture.order.size(), n_past);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_init(1, 0, 1);
    std::printf("generated:");
    for (int g = 0; g < generate; ++g)
    {
        const float* logits = llama_get_logits_ith(lctx, -1);
        int32_t best = 0;
        for (int32_t t = 1; t < n_vocab; ++t)
        {
            if (logits[t] > logits[best])
            {
                best = t;
            }
        }
        std::printf(" %d", best);
        batch.n_tokens = 1;
        batch.token[0] = best;
        batch.pos[0] = n_past++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (g + 1 < generate && llama_decode(lctx, batch) != 0)
        {
            std::fprintf(stderr, "decode failed\n");
            return 1;
        }
    }
    std::printf("\n");
    llama_batch_free(batch);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(wrapper.bitmap);
    mtmd_free(ctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 8 && std::strcmp(argv[1], "--lm") == 0)
    {
        return run_lm(argv[2], std::atoi(argv[3]), argv[4], argv[5], argv[6], argv[7]);
    }
    const char* trace_path = nullptr;
    int layers = 1 << 30;
    int threads = 1;
    while (argc >= 3 && argv[1][0] == '-')
    {
        if (std::strcmp(argv[1], "--trace") == 0)
        {
            trace_path = argv[2];
        }
        else if (std::strcmp(argv[1], "--layers") == 0)
        {
            layers = std::atoi(argv[2]);
        }
        else if (std::strcmp(argv[1], "--threads") == 0)
        {
            threads = std::atoi(argv[2]);
        }
        else
        {
            break;
        }
        argv += 2;
        argc -= 2;
    }
    if (argc != 5)
    {
        std::fprintf(stderr, "usage: llama_vision_dump [--trace trace.bin] [--layers N] [--threads N] <model.gguf> <mmproj.gguf> <image> <embd.bin>\n");
        return 2;
    }
    Capture capture;
    capture.layers = layers;
    if (trace_path != nullptr)
    {
        capture.out = std::fopen(trace_path, "wb");
        if (capture.out == nullptr)
        {
            std::fprintf(stderr, "llama_vision_dump: cannot write %s\n", trace_path);
            return 1;
        }
    }
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr)
    {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;
    params.n_threads = threads;
    params.warmup = false;
    params.print_timings = false;
    params.flash_attn_type = capture.out != nullptr ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (capture.out != nullptr)
    {
        params.cb_eval = eval_callback;
        params.cb_eval_user_data = &capture;
    }
    mtmd_context* ctx = mtmd_init_from_file(argv[2], model, params);
    if (ctx == nullptr)
    {
        std::fprintf(stderr, "cannot load %s\n", argv[2]);
        return 1;
    }
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
    mtmd_bitmap* bitmap = wrapper.bitmap;
    if (bitmap == nullptr)
    {
        std::fprintf(stderr, "cannot read %s\n", argv[3]);
        return 1;
    }
    const std::string prompt = mtmd_default_marker();
    mtmd_input_text text{prompt.c_str(), prompt.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = {bitmap};
    if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0)
    {
        std::fprintf(stderr, "mtmd_tokenize failed\n");
        return 1;
    }
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; ++i)
    {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE)
        {
            size_t n = 0;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
            std::printf("text chunk %zu:", i);
            for (size_t k = 0; k < n; ++k)
            {
                std::printf(" %d", tokens[k]);
            }
            std::printf("\n");
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (mtmd_encode_chunk(ctx, chunk) != 0)
        {
            std::fprintf(stderr, "mtmd_encode_chunk failed\n");
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        const int n_embd = llama_model_n_embd_inp(model);
        const float* embd = mtmd_get_output_embd(ctx);
        FILE* f = std::fopen(argv[4], "wb");
        std::fwrite(embd, sizeof(float), n_tokens * static_cast<size_t>(n_embd), f);
        std::fclose(f);
        std::printf("image chunk %zu: %zu tokens of %d, encode %.2f s\n", i, n_tokens, n_embd,
                    std::chrono::duration<double>(t1 - t0).count());
    }
    if (capture.out != nullptr)
    {
        std::fclose(capture.out);
        std::fprintf(stderr, "llama_vision_dump: %zu tensors written\n", capture.records);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
