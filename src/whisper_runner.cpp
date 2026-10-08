#include "faisca/whisper_runner.h"

#include "faisca/audio.h"
#include "faisca/ops.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace faisca
{

struct WhisperRunner::RowsJob
{
    const float* x;
    const float* w;
    const float* b;
    size_t rows;
    size_t width;
    float eps;
    float* out;
};

struct WhisperRunner::AttentionJob
{
    WhisperRunner* runner;
    const float* q;
    const float* k;
    const float* v;
    size_t n_q;
    size_t n_kv;
    size_t n_head;
    size_t head_dim;
    size_t stride;
    float scale;
    bool causal;
    size_t q_position;
    float* out;
};

struct WhisperRunner::ConvJob
{
    const float* x;
    size_t n_in;
    size_t length;
    const float* w;
    const float* b;
    size_t stride;
    size_t n_out;
    float* out;
};

struct WhisperRunner::MatmulJob
{
    const Weight* weight;
    const float* x;
    size_t n_tokens;
    float* out;
};

WhisperRunner::~WhisperRunner() = default;

bool WhisperRunner::init(const WhisperModel& model, ThreadPool& pool, Error& error)
{
    m_model = &model;
    m_pool = &pool;
    m_position = 0;
    m_space_token = -1;
    const WhisperTokens& t = model.tokens();
    for (int32_t id = 0; id < t.eot; ++id)
    {
        size_t length = 0;
        const char* bytes = model.token(id, length);
        if (length == 1 && bytes[0] == ' ')
        {
            m_space_token = id;
            break;
        }
    }
    return allocate(error);
}

bool WhisperRunner::allocate(Error& error)
{
    const WhisperHparams& h = m_model->hparams();
    const size_t n_state = static_cast<size_t>(h.n_audio_state);
    const size_t n_ctx = static_cast<size_t>(h.n_audio_ctx);
    const size_t n_text_ctx = static_cast<size_t>(h.n_text_ctx);
    const size_t rows = n_ctx > n_text_ctx ? n_ctx : n_text_ctx;
    const size_t threads = static_cast<size_t>(m_pool->size());
    const bool ok = m_conv1.resize_uninitialized(n_state * 2 * n_ctx) && m_conv2.resize_uninitialized(n_state * n_ctx) &&
                    m_x.resize_uninitialized(rows * n_state) && m_h.resize_uninitialized(rows * n_state) &&
                    m_q.resize_uninitialized(rows * n_state) && m_k.resize_uninitialized(rows * n_state) &&
                    m_v.resize_uninitialized(rows * n_state) && m_attn.resize_uninitialized(rows * n_state) &&
                    m_up.resize_uninitialized(rows * 4 * n_state) && m_enc.resize_uninitialized(n_ctx * n_state) &&
                    m_cross_k.resize_uninitialized(static_cast<size_t>(h.n_text_layer) * n_ctx * n_state) &&
                    m_cross_v.resize_uninitialized(static_cast<size_t>(h.n_text_layer) * n_ctx * n_state) &&
                    m_self_k.resize_uninitialized(static_cast<size_t>(h.n_text_layer) * n_text_ctx * n_state) &&
                    m_self_v.resize_uninitialized(static_cast<size_t>(h.n_text_layer) * n_text_ctx * n_state) &&
                    m_scores.resize_uninitialized(threads * rows) &&
                    m_logits.resize_uninitialized(n_text_ctx * static_cast<size_t>(h.n_vocab));
    if (!ok)
    {
        error.set("out of memory for the Whisper buffers");
        return false;
    }
    return true;
}

void WhisperRunner::trace(const char* name, int layer, const float* data, size_t count) const
{
    if (m_trace != nullptr)
    {
        m_trace(m_trace_user, name, layer, data, count);
    }
}

void WhisperRunner::rows_task(void* user, int thread_index, int thread_count)
{
    const RowsJob& job = *static_cast<const RowsJob*>(user);
    const size_t first = job.rows * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = job.rows * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t r = first; r < last; ++r)
    {
        if (job.w != nullptr)
        {
            layer_norm(job.x + r * job.width, job.w, job.b, job.width, job.eps, job.out + r * job.width);
        }
        else
        {
            gelu(job.out + r * job.width, job.width);
        }
    }
}

void WhisperRunner::norm_rows(const float* x, const float* w, const float* b, size_t rows, float* out)
{
    RowsJob job{x, w, b, rows, static_cast<size_t>(m_model->hparams().n_audio_state), m_model->hparams().eps, out};
    m_pool->run(rows_task, &job);
}

void WhisperRunner::gelu_rows(float* x, size_t rows, size_t width)
{
    RowsJob job{nullptr, nullptr, nullptr, rows, width, 0.0f, x};
    m_pool->run(rows_task, &job);
}

void WhisperRunner::conv_task(void* user, int thread_index, int thread_count)
{
    const ConvJob& job = *static_cast<const ConvJob*>(user);
    const size_t first = job.n_out * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = job.n_out * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    conv1d_k3(job.x, job.n_in, job.length, job.w, job.b, job.stride, first, last, job.out);
}

void WhisperRunner::matmul_task(void* user, int thread_index, int thread_count)
{
    const MatmulJob& job = *static_cast<const MatmulJob*>(user);
    const Weight& w = *job.weight;
    const size_t group = w.row_group();
    const size_t groups = (w.n_out + group - 1) / group;
    const size_t first = groups * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count) * group;
    size_t last = groups * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count) * group;
    last = last < w.n_out ? last : w.n_out;
    if (first >= last)
    {
        return;
    }
    w.matmul(job.x, w.n_in, nullptr, 0, nullptr, 0, job.n_tokens, first, last, job.out, w.n_out);
}

void WhisperRunner::matmul(const Weight& w, const float* x, size_t n_tokens, float* out)
{
    MatmulJob job{&w, x, n_tokens, out};
    m_pool->run(matmul_task, &job);
}

void WhisperRunner::attention_task(void* user, int thread_index, int thread_count)
{
    const AttentionJob& job = *static_cast<const AttentionJob*>(user);
    const size_t units = job.n_head * job.n_q;
    const size_t first = units * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = units * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    float* scores = job.runner->m_scores.data() + static_cast<size_t>(thread_index) * (job.runner->m_scores.size() / static_cast<size_t>(thread_count));
    for (size_t u = first; u < last; ++u)
    {
        const size_t head = u / job.n_q;
        const size_t i = u % job.n_q;
        const size_t n_kv = job.causal ? job.q_position + i + 1 : job.n_kv;
        attention_rows(job.q + i * job.stride + head * job.head_dim, job.k + head * job.head_dim, job.v + head * job.head_dim,
                       job.stride, job.head_dim, n_kv, job.scale, scores, job.out + i * job.stride + head * job.head_dim);
    }
}

void WhisperRunner::attention(const float* q, const float* k, const float* v, size_t n_q, size_t n_kv, size_t n_head,
                              float scale, bool causal, size_t q_position, float* out)
{
    const size_t n_state = static_cast<size_t>(m_model->hparams().n_audio_state);
    AttentionJob job{this, q, k, v, n_q, n_kv, n_head, n_state / n_head, n_state, scale, causal, q_position, out};
    m_pool->run(attention_task, &job);
}

// x: n_tokens rows; out gets the projected attention output (before the residual)
void WhisperRunner::attention_block(const WhisperAttention& a, const float* x, size_t n_tokens, size_t n_kv,
                                    const float* k_cache, const float* v_cache, bool self, size_t position, float* out,
                                    const char* prefix, int layer)
{
    const WhisperHparams& h = m_model->hparams();
    const size_t n_state = static_cast<size_t>(h.n_audio_state);
    const size_t n_head = static_cast<size_t>(h.n_audio_head);
    const size_t head_dim = n_state / n_head;
    char name[64];
    const auto named = [&](const char* suffix)
    {
        std::snprintf(name, sizeof(name), "%s%s", prefix, suffix);
        return name;
    };
    // the reference names Q, K and V after the block, not after its attention: e_Qcur, d_Qcur, d_cross_Qcur
    char base[64];
    std::snprintf(base, sizeof(base), "%s", prefix);
    const size_t base_length = std::strlen(base);
    if (base_length >= 5 && std::strcmp(base + base_length - 5, "attn_") == 0)
    {
        base[base_length - 5] = '\0';
    }
    const auto qkv = [&](const char* suffix)
    {
        std::snprintf(name, sizeof(name), "%s%s", base, suffix);
        return name;
    };

    norm_rows(x, a.norm_w, a.norm_b, n_tokens, m_h.data());
    trace(named("norm"), layer, m_h.data(), n_tokens * n_state);

    matmul(a.q, m_h.data(), n_tokens, m_q.data());
    for (size_t t = 0; t < n_tokens; ++t)
    {
        add_bias(m_q.data() + t * n_state, a.q_b, n_state);
    }
    const bool decoder = k_cache != nullptr;
    const float quarter = powf(static_cast<float>(head_dim), -0.25f);
    if (self && decoder)
    {
        for (size_t i = 0; i < n_tokens * n_state; ++i)
        {
            m_q[i] *= quarter;
        }
    }
    trace(qkv("Qcur"), layer, m_q.data(), n_tokens * n_state);

    const float* k = k_cache;
    const float* v = v_cache;
    if (self)
    {
        float* k_rows = decoder ? const_cast<float*>(k_cache) + position * n_state : m_k.data();
        float* v_rows = decoder ? const_cast<float*>(v_cache) + position * n_state : m_v.data();
        matmul(a.k, m_h.data(), n_tokens, k_rows);
        if (decoder)
        {
            for (size_t i = 0; i < n_tokens * n_state; ++i)
            {
                k_rows[i] *= quarter;
            }
        }
        trace(qkv("Kcur"), layer, k_rows, n_tokens * n_state);
        matmul(a.v, m_h.data(), n_tokens, v_rows);
        for (size_t t = 0; t < n_tokens; ++t)
        {
            add_bias(v_rows + t * n_state, a.v_b, n_state);
        }
        trace(qkv("Vcur"), layer, v_rows, n_tokens * n_state);
        k = decoder ? k_cache : m_k.data();
        v = decoder ? v_cache : m_v.data();
    }

    float scale = 1.0f;
    if (!decoder)
    {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    else if (!self)
    {
        scale = quarter;
    }
    attention(m_q.data(), k, v, n_tokens, n_kv, n_head, scale, self && decoder, position, m_attn.data());

    matmul(a.out, m_attn.data(), n_tokens, out);
    for (size_t t = 0; t < n_tokens; ++t)
    {
        add_bias(out + t * n_state, a.out_b, n_state);
    }
    trace(named("out"), layer, out, n_tokens * n_state);
}

void WhisperRunner::mlp_block(const WhisperLayer& layer, float* x, size_t n_tokens, const char* prefix, int layer_index)
{
    const WhisperHparams& h = m_model->hparams();
    const size_t n_state = static_cast<size_t>(h.n_audio_state);
    char name[64];
    const auto named = [&](const char* suffix)
    {
        std::snprintf(name, sizeof(name), "%s%s", prefix, suffix);
        return name;
    };
    norm_rows(x, layer.mlp_norm_w, layer.mlp_norm_b, n_tokens, m_h.data());
    trace(named("ffn_norm"), layer_index, m_h.data(), n_tokens * n_state);
    matmul(layer.mlp_0, m_h.data(), n_tokens, m_up.data());
    for (size_t t = 0; t < n_tokens; ++t)
    {
        add_bias(m_up.data() + t * 4 * n_state, layer.mlp_0_b, 4 * n_state);
    }
    trace(named("ffn_up"), layer_index, m_up.data(), n_tokens * 4 * n_state);
    gelu_rows(m_up.data(), n_tokens, 4 * n_state);
    trace(named("ffn_gelu"), layer_index, m_up.data(), n_tokens * 4 * n_state);
    matmul(layer.mlp_1, m_up.data(), n_tokens, m_h.data());
    for (size_t t = 0; t < n_tokens; ++t)
    {
        add_bias(m_h.data() + t * n_state, layer.mlp_1_b, n_state);
    }
    trace(named("ffn_out"), layer_index, m_h.data(), n_tokens * n_state);
    for (size_t t = 0; t < n_tokens; ++t)
    {
        add(x + t * n_state, m_h.data() + t * n_state, n_state);
    }
    trace(named("l_out"), layer_index, x, n_tokens * n_state);
}

bool WhisperRunner::encode(const float* mel, Error& error)
{
    if (m_model == nullptr)
    {
        error.set("WhisperRunner is not initialized");
        return false;
    }
    const WhisperHparams& h = m_model->hparams();
    const size_t n_state = static_cast<size_t>(h.n_audio_state);
    const size_t n_ctx = static_cast<size_t>(h.n_audio_ctx);
    const size_t n_mels = static_cast<size_t>(h.n_mels);

    {
        ConvJob job{mel, n_mels, 2 * n_ctx, m_model->conv1_w(), m_model->conv1_b(), 1, n_state, m_conv1.data()};
        m_pool->run(conv_task, &job);
    }
    gelu_rows(m_conv1.data(), n_state, 2 * n_ctx);
    trace("conv1", -1, m_conv1.data(), n_state * 2 * n_ctx);
    {
        ConvJob job{m_conv1.data(), n_state, 2 * n_ctx, m_model->conv2_w(), m_model->conv2_b(), 2, n_state, m_conv2.data()};
        m_pool->run(conv_task, &job);
    }
    gelu_rows(m_conv2.data(), n_state, n_ctx);
    trace("embd_conv", -1, m_conv2.data(), n_state * n_ctx);

    const float* pe = m_model->positional_embedding_encoder();
    for (size_t t = 0; t < n_ctx; ++t)
    {
        for (size_t c = 0; c < n_state; ++c)
        {
            m_x[t * n_state + c] = pe[t * n_state + c] + m_conv2[c * n_ctx + t];
        }
    }
    trace("e_pos", -1, m_x.data(), n_ctx * n_state);

    for (int32_t il = 0; il < h.n_audio_layer; ++il)
    {
        const WhisperLayer& layer = m_model->encoder_layer(static_cast<size_t>(il));
        attention_block(layer.attn, m_x.data(), n_ctx, n_ctx, nullptr, nullptr, true, 0, m_h.data(), "e_attn_", il);
        for (size_t t = 0; t < n_ctx; ++t)
        {
            add(m_x.data() + t * n_state, m_h.data() + t * n_state, n_state);
        }
        trace("e_ffn_inp", il, m_x.data(), n_ctx * n_state);
        mlp_block(layer, m_x.data(), n_ctx, "e_", il);
    }
    norm_rows(m_x.data(), m_model->encoder_norm_w(), m_model->encoder_norm_b(), n_ctx, m_enc.data());
    trace("e_out", -1, m_enc.data(), n_ctx * n_state);

    const size_t head_dim = n_state / static_cast<size_t>(h.n_audio_head);
    const float quarter = powf(static_cast<float>(head_dim), -0.25f);
    for (int32_t il = 0; il < h.n_text_layer; ++il)
    {
        const WhisperAttention& cross = m_model->decoder_layer(static_cast<size_t>(il)).cross;
        float* k = m_cross_k.data() + static_cast<size_t>(il) * n_ctx * n_state;
        float* v = m_cross_v.data() + static_cast<size_t>(il) * n_ctx * n_state;
        matmul(cross.k, m_enc.data(), n_ctx, k);
        for (size_t i = 0; i < n_ctx * n_state; ++i)
        {
            k[i] *= quarter;
        }
        trace("cross_k", il, k, n_ctx * n_state);
        matmul(cross.v, m_enc.data(), n_ctx, v);
        for (size_t t = 0; t < n_ctx; ++t)
        {
            add_bias(v + t * n_state, cross.v_b, n_state);
        }
        trace("cross_v", il, v, n_ctx * n_state);
    }
    m_position = 0;
    return true;
}

bool WhisperRunner::decode(const int32_t* tokens, size_t count, Array<float>& logits, Error& error)
{
    const WhisperHparams& h = m_model->hparams();
    const size_t n_state = static_cast<size_t>(h.n_audio_state);
    const size_t n_ctx = static_cast<size_t>(h.n_audio_ctx);
    const size_t n_text_ctx = static_cast<size_t>(h.n_text_ctx);
    const size_t n_vocab = static_cast<size_t>(h.n_vocab);
    if (count == 0 || m_position + count > n_text_ctx)
    {
        error.set("decode: %zu tokens at position %zu exceed the text context of %zu", count, m_position, n_text_ctx);
        return false;
    }
    const float* te = m_model->token_embedding().f32();
    const float* pe = m_model->positional_embedding_decoder();
    for (size_t t = 0; t < count; ++t)
    {
        if (tokens[t] < 0 || static_cast<size_t>(tokens[t]) >= n_vocab)
        {
            error.set("decode: token %d out of range", tokens[t]);
            return false;
        }
        const float* embedding = te + static_cast<size_t>(tokens[t]) * n_state;
        const float* positional = pe + (m_position + t) * n_state;
        for (size_t c = 0; c < n_state; ++c)
        {
            m_x[t * n_state + c] = embedding[c] + positional[c];
        }
    }
    for (int32_t il = 0; il < h.n_text_layer; ++il)
    {
        const WhisperLayer& layer = m_model->decoder_layer(static_cast<size_t>(il));
        const float* self_k = m_self_k.data() + static_cast<size_t>(il) * n_text_ctx * n_state;
        const float* self_v = m_self_v.data() + static_cast<size_t>(il) * n_text_ctx * n_state;
        attention_block(layer.attn, m_x.data(), count, m_position + count, self_k, self_v, true, m_position, m_h.data(), "d_attn_", il);
        for (size_t t = 0; t < count; ++t)
        {
            add(m_x.data() + t * n_state, m_h.data() + t * n_state, n_state);
        }
        trace("d_cross_inp", il, m_x.data(), count * n_state);
        const float* cross_k = m_cross_k.data() + static_cast<size_t>(il) * n_ctx * n_state;
        const float* cross_v = m_cross_v.data() + static_cast<size_t>(il) * n_ctx * n_state;
        attention_block(layer.cross, m_x.data(), count, n_ctx, cross_k, cross_v, false, m_position, m_h.data(), "d_cross_", il);
        for (size_t t = 0; t < count; ++t)
        {
            add(m_x.data() + t * n_state, m_h.data() + t * n_state, n_state);
        }
        trace("d_ffn_inp", il, m_x.data(), count * n_state);
        mlp_block(layer, m_x.data(), count, "d_", il);
    }
    norm_rows(m_x.data(), m_model->decoder_norm_w(), m_model->decoder_norm_b(), count, m_h.data());
    trace("d_out", -1, m_h.data(), count * n_state);
    matmul(m_model->token_embedding(), m_h.data(), count, m_logits.data());
    trace("logits", -1, m_logits.data(), count * n_vocab);
    if (!logits.resize_uninitialized(count * n_vocab))
    {
        error.set("out of memory");
        return false;
    }
    std::memcpy(logits.data(), m_logits.data(), count * n_vocab * sizeof(float));
    m_position += count;
    return true;
}

bool WhisperRunner::transcribe(int32_t language_id, Array<int32_t>& tokens, Error& error)
{
    const WhisperHparams& h = m_model->hparams();
    const WhisperTokens& t = m_model->tokens();
    const size_t n_vocab = static_cast<size_t>(h.n_vocab);
    if (language_id < 0 || language_id >= t.n_languages)
    {
        error.set("language id %d is not in this model (%d languages)", language_id, t.n_languages);
        return false;
    }
    m_position = 0;
    const int32_t prompt[4] = {t.sot, t.language(language_id), t.transcribe, t.no_timestamps};
    Array<float> logits;
    if (!decode(prompt, t.multilingual ? 4 : 2, logits, error))
    {
        return false;
    }
    const size_t n_max = static_cast<size_t>(h.n_text_ctx) / 2 - 4;
    for (size_t i = 0; i < n_max; ++i)
    {
        float* row = logits.data() + (logits.size() - n_vocab);
        const float minus_inf = -INFINITY;
        if (i == 0)
        {
            row[t.eot] = minus_inf;
            if (m_space_token >= 0)
            {
                row[m_space_token] = minus_inf;
            }
        }
        row[t.no_timestamps] = minus_inf;
        for (size_t id = static_cast<size_t>(t.beg); id < n_vocab; ++id)
        {
            row[id] = minus_inf;
        }
        row[t.sot] = minus_inf;
        row[t.nosp] = minus_inf;
        row[t.solm] = minus_inf;
        row[t.translate] = minus_inf;
        row[t.transcribe] = minus_inf;
        row[t.prev] = minus_inf;
        for (int32_t l = 0; l < t.n_languages; ++l)
        {
            row[t.language(l)] = minus_inf;
        }
        int32_t best = 0;
        for (size_t id = 1; id < n_vocab; ++id)
        {
            if (row[id] > row[static_cast<size_t>(best)])
            {
                best = static_cast<int32_t>(id);
            }
        }
        if (best == t.eot)
        {
            break;
        }
        if (!tokens.push_back(best))
        {
            error.set("out of memory");
            return false;
        }
        if (!decode(&best, 1, logits, error))
        {
            return false;
        }
    }
    return true;
}

bool WhisperRunner::transcribe_audio(const float* samples, size_t n_samples, int32_t language_id, Array<int32_t>& tokens, Error& error)
{
    const WhisperHparams& h = m_model->hparams();
    const size_t n_mels = static_cast<size_t>(h.n_mels);
    const size_t n_ctx = static_cast<size_t>(h.n_audio_ctx);
    Array<float> mel;
    const size_t n_len = log_mel_spectrogram(samples, n_samples, m_model->filters(), n_mels, mel, error);
    if (n_len == 0)
    {
        return false;
    }
    Array<float> window;
    if (!window.resize_uninitialized(n_mels * 2 * n_ctx))
    {
        error.set("out of memory");
        return false;
    }
    // whisper.cpp stops when less than 100 ms of audio remain; the spectrogram holds 30 s of silence padding
    const size_t n_len_org = 1 + (n_samples + WHISPER_N_FFT / 2 - WHISPER_N_FFT) / WHISPER_HOP;
    for (size_t seek = 0; seek + 10 < n_len_org; seek += 2 * n_ctx)
    {
        std::memset(window.data(), 0, window.size() * sizeof(float));
        const size_t i1 = seek + 2 * n_ctx < n_len ? seek + 2 * n_ctx : n_len;
        for (size_t j = 0; j < n_mels; ++j)
        {
            for (size_t i = seek; i < i1; ++i)
            {
                window[j * 2 * n_ctx + (i - seek)] = mel[j * n_len + i];
            }
        }
        trace("mel", -1, window.data(), window.size());
        if (!encode(window.data(), error) || !transcribe(language_id, tokens, error))
        {
            return false;
        }
    }
    return true;
}

}
