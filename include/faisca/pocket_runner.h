#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/pocket_model.h"
#include "faisca/runner.h"
#include "faisca/thread.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// the flow LM KV rows of a voice: per layer n rows of dim values, keys then values
struct PocketVoice
{
    Array<float> k;
    Array<float> v;
    uint32_t n = 0;

    // <name>.k and <name>.v of a voices GGUF, F32 [dim, n, n_layers]
    bool load(const char* path, const char* name, const PocketModel& model, Error& error);
};

// torch.manual_seed and normal_ on a CPU float tensor: MT19937, uniforms from the low 24
// bits of one draw, Box-Muller over blocks of 16 (a trailing partial block redrawn)
class TorchRandom
{
public:
    explicit TorchRandom(uint32_t seed);

    uint32_t next();
    float uniform();
    // n at least 16, mean 0
    void normal(float* x, size_t n, float std);

private:
    uint32_t m_state[624];
    size_t m_index = 624;
};

struct PocketStats
{
    size_t chunks = 0;
    size_t frames = 0;
    size_t steps = 0;
};

// the synthesis of pockettts.cpp: the text in chunks of at most max_chunk_tokens tokens,
// each prefilled after the voice rows and run one flow LM step per frame until EOS plus
// frames_after_eos, the latents decoded by Mimi, the first frame of every chunk dropped
class PocketRunner
{
public:
    PocketRunner() = default;

    PocketRunner(const PocketRunner&) = delete;
    PocketRunner& operator=(const PocketRunner&) = delete;

    bool init(const PocketModel& model, ThreadPool& pool, Error& error);

    void set_trace(RunnerTrace callback, void* user)
    {
        m_trace = callback;
        m_trace_user = user;
    }

    void set_seed(uint32_t seed)
    {
        m_seed = seed;
    }

    void set_lsd_steps(int steps)
    {
        m_lsd_steps = steps;
    }

    void set_max_chunk_tokens(size_t count)
    {
        m_max_chunk_tokens = count;
    }

    // PCM at hparams().sample_rate appended to samples
    bool synthesize(const PocketVoice& voice, const char* text, size_t length, Array<float>& samples, Error& error);

    const PocketStats& stats() const
    {
        return m_stats;
    }

private:
    struct RowsJob;
    struct MatmulJob;
    struct AttentionJob;
    struct ConvJob;
    struct ConvTrJob;
    struct UpsampleJob;

    static void rows_task(void* user, int thread_index, int thread_count);
    static void matmul_task(void* user, int thread_index, int thread_count);
    static void quantize_task(void* user, int thread_index, int thread_count);
    static void attention_task(void* user, int thread_index, int thread_count);
    static void conv_task(void* user, int thread_index, int thread_count);
    static void convtr_task(void* user, int thread_index, int thread_count);
    static void upsample_task(void* user, int thread_index, int thread_count);

    void trace(const char* name, int layer, const float* data, size_t count) const;
    bool reserve_rows(size_t rows, size_t width, size_t ffn, Error& error);
    void quantize_inputs(const float* x, size_t n_tokens, size_t n);
    void matmul(const Weight& w, const float* x, size_t n_tokens, float* out);
    void norm_rows(const float* x, const float* w, const float* b, size_t rows, size_t width, float eps, float* out);
    void gelu_rows(float* x, size_t rows, size_t width);
    void attention(const float* q, size_t q_stride, const float* k, const float* v, size_t kv_stride, size_t n_q,
                   size_t position, size_t n_head, size_t head_dim, size_t window, float* out, size_t out_stride);

    // one transformer layer over n_tokens rows of x in place; keys and values of the tokens
    // written to rows position.. of k and v (kv_stride apart) and attended with the window
    void transformer_layer(const PocketTfmLayer& layer, float* x, size_t n_tokens, size_t width, size_t n_head,
                           float max_period, size_t position, float* k, float* v, size_t kv_stride, size_t window);

    bool flow_prefill(const int32_t* ids, size_t count, size_t position, Error& error);
    bool flow_step(const float* latent, const float* noise, size_t position, float& eos, float* out, Error& error);
    void time_embed(const PocketTimeEmbed& te, float t, float* out);
    void flow_head(const float* cond, const float* noise, float* out);

    bool decode(const float* latents, size_t n_frames, Array<float>& audio, Error& error);
    void conv1d(const PocketConv& c, const float* x, size_t length, float* out);
    void convtr(const PocketConv& c, const float* x, size_t length, float* out);
    void elu_rows(float* x, size_t count);

    const PocketModel* m_model = nullptr;
    ThreadPool* m_pool = nullptr;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;
    int m_trace_layer = -1;
    uint32_t m_seed = 0;
    int m_lsd_steps = 1;
    size_t m_max_chunk_tokens = 50;
    PocketStats m_stats;

    // flow LM cache: per layer rows of dim values
    Array<float> m_cache_k;
    Array<float> m_cache_v;
    size_t m_cache_rows = 0;

    // transformer rows
    Array<float> m_x;
    Array<float> m_h;
    Array<float> m_proj;
    Array<float> m_attn;
    Array<float> m_up;
    Array<uint8_t> m_xq;
    Array<int32_t> m_sums;
    size_t m_xq_stride = 0;
    size_t m_sums_stride = 0;
    Array<float> m_scores;
    const float* m_quantize_x = nullptr;
    size_t m_quantize_tokens = 0;
    size_t m_quantize_n = 0;

    // flow head
    Array<float> m_head;

    // Mimi decoder
    Array<float> m_mimi_k;
    Array<float> m_mimi_v;
    Array<float> m_dec_a;
    Array<float> m_dec_b;
    Array<float> m_dec_c;
    Array<float> m_convtr_tmp;
};

}
