#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/model.h"
#include "faisca/thread.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// called for every intermediate vector of every token when a trace is set; layer is -1 after the last block
typedef void (*RunnerTrace)(void* user, const char* name, int layer, const float* data, size_t count);

// forward pass over a KV cache of n_ctx positions. Tokens are processed in
// batches: every weight is read once per batch, so a prompt costs far less than
// one token at a time. Every matrix row group and every attention head is
// computed whole by one thread, so results do not depend on the thread count.
// The cache holds keys and values as f16 unless kv_f16 is false; q and the
// softmax weights stay float32 either way.
class Runner
{
public:
    static constexpr size_t BATCH = 32;

    Runner() = default;

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    bool init(const LlamaModel& model, uint32_t n_ctx, int n_threads, Error& error, bool kv_f16 = true);

    void release();

    void reset()
    {
        m_pos = 0;
    }

    uint32_t position() const
    {
        return m_pos;
    }

    uint32_t context_size() const
    {
        return m_n_ctx;
    }

    int threads() const
    {
        return m_pool.size();
    }

    bool kv_f16() const
    {
        return m_kv_f16;
    }

    size_t cache_bytes() const
    {
        return m_cache_bytes;
    }

    const LlamaModel* model() const
    {
        return m_model;
    }

    void set_trace(RunnerTrace callback, void* user)
    {
        m_trace = callback;
        m_trace_user = user;
    }

    // feeds count tokens after the current position; logits gets n_vocab values for the last one
    bool forward(const int32_t* tokens, size_t count, Array<float>& logits, Error& error);

private:
    struct MatmulJob;
    struct AttentionJob;
    struct TokenJob;

    typedef void (Runner::*TokenStep)(const LlamaLayer* layer, uint32_t pos, size_t token);
    void for_tokens(TokenStep step, const LlamaLayer* layer, uint32_t pos, size_t n_tokens);
    void step_attn_norm(const LlamaLayer* layer, uint32_t pos, size_t token);
    void step_rope_store(const LlamaLayer* layer, uint32_t pos, size_t token);
    void step_ffn_inp(const LlamaLayer* layer, uint32_t pos, size_t token);
    void step_swiglu(const LlamaLayer* layer, uint32_t pos, size_t token);
    void step_residual(const LlamaLayer* layer, uint32_t pos, size_t token);
    void step_output_norm(const LlamaLayer* layer, uint32_t pos, size_t token);
    static void token_task(void* user, int thread_index, int thread_count);
    static void quantize_task(void* user, int thread_index, int thread_count);
    size_t m_layer = 0;
    size_t m_quantize_n = 0;
    Activation m_quantize_kind = Activation::Q8_0;
    const float* m_quantize_x = nullptr;
    size_t m_quantize_stride = 0;
    size_t m_quantize_tokens = 0;

    void forward_batch(const int32_t* tokens, size_t count, uint32_t pos, float* logits);
    void quantize_inputs(const float* x, size_t stride, size_t n_tokens, size_t n, Activation kind);
    void matmul(const Weight& w, const float* x, size_t x_stride, size_t n_tokens, float* out, size_t out_stride);
    void attention(const LlamaLayer& layer, int il, uint32_t pos, size_t n_tokens);
    void attention_unit(int il, uint32_t pos, size_t token, size_t head, float* scores);
    void trace_rows(const char* name, int layer, const float* data, size_t count, size_t n_tokens, size_t stride) const;

    static void matmul_task(void* user, int thread_index, int thread_count);
    static void attention_task(void* user, int thread_index, int thread_count);

    const LlamaModel* m_model = nullptr;
    ThreadPool m_pool;
    uint32_t m_n_ctx = 0;
    uint32_t m_pos = 0;
    bool m_kv_f16 = true;
    size_t m_cache_bytes = 0;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;

    Array<float> m_k; // [layer][kv head][position tile][head dim][16], see key_index in ops.h
    Array<float> m_v; // [layer][position][kv value]
    Array<uint16_t> m_k16; // the same two as f16, used instead when kv_f16 is set
    Array<uint16_t> m_v16;
    Array<float> m_k_batch; // [token][kv value], the keys and values of the batch before they go into the cache
    Array<float> m_v_batch;

    Array<float> m_x; // [token][n_embd]
    Array<float> m_h;
    Array<float> m_q; // [token][n_head * head_dim]
    Array<float> m_attn;
    Array<float> m_gate; // [token][n_ff]
    Array<float> m_up;
    Array<float> m_act;
    Array<float> m_scores; // [thread][n_ctx rounded up to 16]
    Array<float> m_logits; // [token][n_vocab], only while tracing

    Array<uint8_t> m_xq; // [token][Q8_0 or Q8_K row]
    Array<int32_t> m_sums; // [token][block of 32 or 16]
    size_t m_xq_stride = 0;
    size_t m_sums_stride = 0;
};

}
