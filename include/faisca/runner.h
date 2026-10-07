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

// forward pass one token at a time, with a KV cache of n_ctx positions; every
// matrix row and every attention head is computed by one thread, so the result
// does not depend on the thread count
class Runner
{
public:
    Runner() = default;

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    bool init(const LlamaModel& model, uint32_t n_ctx, int n_threads, Error& error);

    int threads() const
    {
        return m_pool.size();
    }

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
    void forward_one(int32_t token, uint32_t pos, float* logits);
    void attention(const LlamaLayer& layer, int il, uint32_t pos);
    void trace(const char* name, int layer, const float* data, size_t count) const;

    // the vector every following matmul multiplies; quantized once if a weight needs it
    void set_input(const float* x, size_t n);
    void matmul(const Weight& w, float* out);
    void matmul_rows(const Weight& w, float* out, size_t first, size_t last) const;
    void attention_heads(const LlamaLayer& layer, int il, uint32_t pos, size_t first, size_t last);

    struct MatmulJob;
    struct AttentionJob;
    static void matmul_task(void* user, int thread_index, int thread_count);
    static void attention_task(void* user, int thread_index, int thread_count);

    const LlamaModel* m_model = nullptr;
    ThreadPool m_pool;
    uint32_t m_n_ctx = 0;
    uint32_t m_pos = 0;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;

    Array<float> m_k; // [layer][position][kv value]
    Array<float> m_v;

    Array<float> m_x;
    Array<float> m_h;
    Array<float> m_q;
    Array<float> m_attn;
    Array<float> m_gate;
    Array<float> m_up;
    Array<float> m_act;
    Array<float> m_scores;

    const float* m_input = nullptr;
    size_t m_input_size = 0;
    bool m_input_quantized = false;
    Array<uint8_t> m_input_q8;
    Array<int32_t> m_input_sums;
};

}
