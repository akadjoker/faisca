#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/runner.h"
#include "faisca/thread.h"
#include "faisca/whisper_model.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// Whisper in float32: one 30 s window through the encoder, then the decoder with a self
// attention cache; every intermediate tensor can be traced under the names of the reference
class WhisperRunner
{
public:
    WhisperRunner() = default;
    ~WhisperRunner();

    WhisperRunner(const WhisperRunner&) = delete;
    WhisperRunner& operator=(const WhisperRunner&) = delete;

    bool init(const WhisperModel& model, ThreadPool& pool, Error& error);

    void set_trace(RunnerTrace callback, void* user)
    {
        m_trace = callback;
        m_trace_user = user;
    }

    // mel is n_mels rows of 2 * n_audio_ctx frames (the window, zero padded); fills the encoder
    // output and the cross attention keys and values, and empties the self attention cache
    bool encode(const float* mel, Error& error);

    // count tokens at positions position.. appended to the self attention cache; logits gets
    // count rows of n_vocab
    bool decode(const int32_t* tokens, size_t count, Array<float>& logits, Error& error);

    size_t position() const
    {
        return m_position;
    }

    // greedy transcription of the encoded window without timestamps: the text tokens up to
    // and excluding the end of text token
    bool transcribe(int32_t language_id, Array<int32_t>& tokens, Error& error);

    // the whole audio: one window per 30 s, text tokens of every window appended
    bool transcribe_audio(const float* samples, size_t n_samples, int32_t language_id, Array<int32_t>& tokens, Error& error);

private:
    struct RowsJob;
    struct AttentionJob;
    struct ConvJob;
    struct MatmulJob;

    bool allocate(Error& error);
    void trace(const char* name, int layer, const float* data, size_t count) const;

    static void rows_task(void* user, int thread_index, int thread_count);
    static void attention_task(void* user, int thread_index, int thread_count);
    static void conv_task(void* user, int thread_index, int thread_count);
    static void matmul_task(void* user, int thread_index, int thread_count);

    void matmul(const Weight& w, const float* x, size_t n_tokens, float* out);
    void attention(const float* q, const float* k, const float* v, size_t n_q, size_t n_kv, size_t n_head, float scale,
                   bool causal, size_t q_position, float* out);
    void norm_rows(const float* x, const float* w, const float* b, size_t rows, float* out);
    void gelu_rows(float* x, size_t rows, size_t width);
    void attention_block(const WhisperAttention& a, const float* x, size_t n_tokens, size_t n_kv, const float* k_cache,
                         const float* v_cache, bool self, size_t position, float* out, const char* prefix, int layer);
    void mlp_block(const WhisperLayer& layer, float* x, size_t n_tokens, const char* prefix, int layer_index);

    const WhisperModel* m_model = nullptr;
    ThreadPool* m_pool = nullptr;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;
    size_t m_position = 0;
    int32_t m_space_token = -1;

    Array<float> m_conv1;
    Array<float> m_conv2;
    Array<float> m_x;
    Array<float> m_h;
    Array<float> m_q;
    Array<float> m_k;
    Array<float> m_v;
    Array<float> m_attn;
    Array<float> m_up;
    Array<float> m_enc;
    Array<float> m_cross_k;
    Array<float> m_cross_v;
    Array<float> m_self_k;
    Array<float> m_self_v;
    Array<float> m_scores;
    Array<float> m_logits;
};

}
