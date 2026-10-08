#include "faisca/runner.h"

#include "faisca/ops.h"

#include <cinttypes>
#include <cmath>
#include <cstring>

namespace faisca
{

struct Runner::MatmulJob
{
    Runner* runner;
    const Weight* weight;
    const float* x;
    size_t x_stride;
    size_t n_tokens;
    float* out;
    size_t out_stride;
};

struct Runner::AttentionJob
{
    Runner* runner;
    int il;
    uint32_t pos;
    size_t n_tokens;
};

struct Runner::TokenJob
{
    Runner* runner;
    TokenStep step;
    const LlamaLayer* layer;
    uint32_t pos;
    size_t n_tokens;
};

void Runner::release()
{
    m_pool.stop();
    m_model = nullptr;
    m_n_ctx = 0;
    m_pos = 0;
    m_k.release();
    m_v.release();
    m_k16.release();
    m_v16.release();
    m_k_batch.release();
    m_v_batch.release();
    m_x.release();
    m_h.release();
    m_q.release();
    m_attn.release();
    m_gate.release();
    m_up.release();
    m_act.release();
    m_scores.release();
    m_logits.release();
    m_xq.release();
    m_sums.release();
    m_xq_stride = 0;
    m_sums_stride = 0;
    m_cache_bytes = 0;
}

bool Runner::init(const LlamaModel& model, uint32_t n_ctx, int n_threads, Error& error, bool kv_f16)
{
    release();
    if (!model.is_loaded())
    {
        error.set("the model is not loaded");
        return false;
    }
    if (n_ctx == 0)
    {
        error.set("the context size must be at least 1");
        return false;
    }
    if (!m_pool.start(n_threads, error))
    {
        return false;
    }

    const LlamaHparams& h = model.hparams();
    const size_t n_kv = static_cast<size_t>(h.n_head_kv) * h.head_dim;
    const size_t n_q = static_cast<size_t>(h.n_head) * h.head_dim;
    const size_t cache = static_cast<size_t>(h.n_layer) * n_ctx * n_kv;
    if (cache / n_kv / n_ctx != h.n_layer || cache > static_cast<size_t>(-1) / (2 * sizeof(float)))
    {
        error.set("a context of %" PRIu32 " positions does not fit in memory", n_ctx);
        return false;
    }

    const size_t widest = h.n_ff > h.n_embd ? h.n_ff : h.n_embd;
    m_xq_stride = (widest / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
    m_sums_stride = widest / Q8_0_BLOCK;
    const size_t threads = static_cast<size_t>(m_pool.size());
    const size_t scores = (static_cast<size_t>(n_ctx) + 15) / 16 * 16;
    const size_t k_cache = static_cast<size_t>(h.n_layer) * h.n_head_kv * (scores * h.head_dim);
    const size_t element = kv_f16 ? sizeof(uint16_t) : sizeof(float);
    m_cache_bytes = (k_cache + cache) * element;
    const bool cache_ok = kv_f16 ? m_k16.resize_uninitialized(k_cache) && m_v16.resize_uninitialized(cache) &&
                                       m_v_batch.resize_uninitialized(BATCH * n_kv)
                                 : m_k.resize_uninitialized(k_cache) && m_v.resize_uninitialized(cache);
    if (!cache_ok || !m_k_batch.resize_uninitialized(BATCH * n_kv) || !m_x.resize_uninitialized(BATCH * h.n_embd) ||
        !m_h.resize_uninitialized(BATCH * h.n_embd) || !m_q.resize_uninitialized(BATCH * n_q) ||
        !m_attn.resize_uninitialized(BATCH * n_q) || !m_gate.resize_uninitialized(BATCH * h.n_ff) ||
        !m_up.resize_uninitialized(BATCH * h.n_ff) || !m_act.resize_uninitialized(BATCH * h.n_ff) ||
        !m_scores.resize_uninitialized(threads * scores) || !m_xq.resize_uninitialized(BATCH * m_xq_stride) ||
        !m_sums.resize_uninitialized(BATCH * m_sums_stride))
    {
        release();
        error.set("out of memory: the KV cache for %" PRIu32 " positions needs %zu MiB", n_ctx,
                  (k_cache + cache) * element >> 20);
        return false;
    }
    // the score kernel reads whole tiles of 16 positions, so unused positions must hold numbers
    if (kv_f16)
    {
        std::memset(m_k16.data(), 0, k_cache * sizeof(uint16_t));
        std::memset(m_v16.data(), 0, cache * sizeof(uint16_t));
    }
    else
    {
        std::memset(m_k.data(), 0, k_cache * sizeof(float));
        std::memset(m_v.data(), 0, cache * sizeof(float));
    }

    m_model = &model;
    m_n_ctx = n_ctx;
    m_pos = 0;
    m_kv_f16 = kv_f16;
    return true;
}

bool Runner::forward(const int32_t* tokens, size_t count, Array<float>& logits, Error& error)
{
    if (m_model == nullptr)
    {
        error.set("the runner is not initialized");
        return false;
    }
    if (count == 0)
    {
        error.set("no tokens to process");
        return false;
    }
    if (count > m_n_ctx - m_pos)
    {
        error.set("context full: %zu tokens at position %" PRIu32 " exceed %" PRIu32, count, m_pos, m_n_ctx);
        return false;
    }

    const LlamaHparams& h = m_model->hparams();
    for (size_t i = 0; i < count; ++i)
    {
        if (tokens[i] < 0 || static_cast<uint32_t>(tokens[i]) >= h.n_vocab)
        {
            error.set("token %" PRId32 " is outside the vocabulary", tokens[i]);
            return false;
        }
    }
    if (!logits.resize_uninitialized(h.n_vocab))
    {
        error.set("out of memory");
        return false;
    }
    if (m_trace != nullptr && !m_logits.resize_uninitialized(BATCH * h.n_vocab))
    {
        error.set("out of memory");
        return false;
    }

    for (size_t done = 0; done < count;)
    {
        const size_t batch = count - done < BATCH ? count - done : BATCH;
        forward_batch(tokens + done, batch, m_pos, logits.data());
        m_pos += static_cast<uint32_t>(batch);
        done += batch;
    }
    return true;
}

void Runner::trace_rows(const char* name, int layer, const float* data, size_t count, size_t n_tokens,
                        size_t stride) const
{
    if (m_trace == nullptr)
    {
        return;
    }
    for (size_t t = 0; t < n_tokens; ++t)
    {
        m_trace(m_trace_user, name, layer, data + t * stride, count);
    }
}

void Runner::quantize_task(void* user, int thread_index, int thread_count)
{
    Runner& runner = *static_cast<Runner*>(user);
    const size_t first = runner.m_quantize_tokens * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = runner.m_quantize_tokens * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t t = first; t < last; ++t)
    {
        quantize_q8_0(runner.m_quantize_x + t * runner.m_quantize_stride, runner.m_quantize_n,
                      runner.m_xq.data() + t * runner.m_xq_stride);
        q8_0_block_sums(runner.m_xq.data() + t * runner.m_xq_stride, runner.m_quantize_n,
                        runner.m_sums.data() + t * runner.m_sums_stride);
    }
}

void Runner::quantize_inputs(const float* x, size_t stride, size_t n_tokens, size_t n)
{
    m_quantize_x = x;
    m_quantize_stride = stride;
    m_quantize_tokens = n_tokens;
    m_quantize_n = n;
    m_pool.run(quantize_task, this);
}

void Runner::token_task(void* user, int thread_index, int thread_count)
{
    const TokenJob& job = *static_cast<const TokenJob*>(user);
    const size_t first = job.n_tokens * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = job.n_tokens * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t t = first; t < last; ++t)
    {
        (job.runner->*job.step)(job.layer, job.pos, t);
    }
}

void Runner::for_tokens(TokenStep step, const LlamaLayer* layer, uint32_t pos, size_t n_tokens)
{
    TokenJob job{this, step, layer, pos, n_tokens};
    m_pool.run(token_task, &job);
}

void Runner::step_attn_norm(const LlamaLayer* layer, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    rms_norm(m_x.data() + token * h.n_embd, layer->attn_norm, h.n_embd, h.rms_eps, m_h.data() + token * h.n_embd);
}

void Runner::step_ffn_inp(const LlamaLayer* layer, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    add(m_x.data() + token * h.n_embd, m_h.data() + token * h.n_embd, h.n_embd);
    rms_norm(m_x.data() + token * h.n_embd, layer->ffn_norm, h.n_embd, h.rms_eps, m_h.data() + token * h.n_embd);
}

void Runner::step_swiglu(const LlamaLayer*, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    swiglu(m_gate.data() + token * h.n_ff, m_up.data() + token * h.n_ff, h.n_ff, m_act.data() + token * h.n_ff);
}

void Runner::step_residual(const LlamaLayer*, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    add(m_x.data() + token * h.n_embd, m_h.data() + token * h.n_embd, h.n_embd);
}

void Runner::step_output_norm(const LlamaLayer*, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    rms_norm(m_x.data() + token * h.n_embd, m_model->output_norm(), h.n_embd, h.rms_eps, m_h.data() + token * h.n_embd);
}

void Runner::matmul_task(void* user, int thread_index, int thread_count)
{
    const MatmulJob& job = *static_cast<const MatmulJob*>(user);
    const Weight& w = *job.weight;
    // whole row groups per thread, so every row is summed the same way whatever the thread count
    const size_t group = w.row_group();
    const size_t groups = (w.n_out + group - 1) / group;
    const size_t first = groups * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count) * group;
    size_t last = groups * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count) * group;
    last = last < w.n_out ? last : w.n_out;
    if (first >= last)
    {
        return;
    }
    const Runner& runner = *job.runner;
    w.matmul(job.x, job.x_stride, runner.m_xq.data(), runner.m_xq_stride, runner.m_sums.data(), runner.m_sums_stride,
             job.n_tokens, first, last, job.out, job.out_stride);
}

void Runner::matmul(const Weight& w, const float* x, size_t x_stride, size_t n_tokens, float* out, size_t out_stride)
{
    if (w.type != GgmlType::F32)
    {
        quantize_inputs(x, x_stride, n_tokens, w.n_in);
    }
    MatmulJob job{this, &w, x, x_stride, n_tokens, out, out_stride};
    m_pool.run(matmul_task, &job);
}

void Runner::attention_task(void* user, int thread_index, int thread_count)
{
    AttentionJob& job = *static_cast<AttentionJob*>(user);
    Runner& runner = *job.runner;
    const size_t n_head = runner.m_model->hparams().n_head;
    const size_t units = job.n_tokens * n_head;
    const size_t first = units * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = units * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    const size_t scores_stride = (static_cast<size_t>(runner.m_n_ctx) + 15) / 16 * 16;
    float* scores = runner.m_scores.data() + static_cast<size_t>(thread_index) * scores_stride;
    for (size_t u = first; u < last; ++u)
    {
        runner.attention_unit(job.il, job.pos, u / n_head, u % n_head, scores);
    }
}

void Runner::attention_unit(int il, uint32_t pos, size_t token, size_t head, float* scores)
{
    const LlamaHparams& h = m_model->hparams();
    const size_t head_dim = h.head_dim;
    const size_t n_kv = static_cast<size_t>(h.n_head_kv) * head_dim;
    const size_t n_q = static_cast<size_t>(h.n_head) * head_dim;
    const size_t kv_head = head / (h.n_head / h.n_head_kv);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const size_t tiles = (static_cast<size_t>(m_n_ctx) + 15) / 16;
    const size_t k_offset = (static_cast<size_t>(il) * h.n_head_kv + kv_head) * tiles * head_dim * KEY_TILE;
    const size_t v_offset = static_cast<size_t>(il) * m_n_ctx * n_kv + kv_head * head_dim;
    const float* q = m_q.data() + token * n_q + head * head_dim;
    float* out = m_attn.data() + token * n_q + head * head_dim;
    const size_t n_past = static_cast<size_t>(pos) + token + 1;
    if (m_kv_f16)
    {
        attention_head_f16(q, m_k16.data() + k_offset, m_v16.data() + v_offset, n_kv, head_dim, n_past, scale, scores,
                           out);
    }
    else
    {
        attention_head(q, m_k.data() + k_offset, m_v.data() + v_offset, n_kv, head_dim, n_past, scale, scores, out);
    }
}

void Runner::step_rope_store(const LlamaLayer*, uint32_t pos, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    const size_t head_dim = h.head_dim;
    const size_t n_kv = static_cast<size_t>(h.n_head_kv) * head_dim;
    const size_t n_q = static_cast<size_t>(h.n_head) * head_dim;
    const uint32_t p = pos + static_cast<uint32_t>(token);
    float* k_cur = m_k_batch.data() + token * n_kv;
    rope_norm(m_q.data() + token * n_q, h.n_head, head_dim, h.n_rot, p, h.rope_freq_base);
    rope_norm(k_cur, h.n_head_kv, head_dim, h.n_rot, p, h.rope_freq_base);

    const size_t tiles = (static_cast<size_t>(m_n_ctx) + 15) / 16;
    const size_t layer_offset = static_cast<size_t>(m_layer) * h.n_head_kv * tiles * head_dim * KEY_TILE;
    if (m_kv_f16)
    {
        uint16_t* k_half = m_k16.data() + layer_offset;
        for (size_t i = 0; i < n_kv; ++i)
        {
            k_half[(i / head_dim) * tiles * head_dim * KEY_TILE + key_index(head_dim, p, i % head_dim)] = f32_to_f16(k_cur[i]);
        }
        f32_to_f16_values(m_v_batch.data() + token * n_kv, n_kv,
                          m_v16.data() + (static_cast<size_t>(m_layer) * m_n_ctx + p) * n_kv);
        return;
    }
    float* k_tiles = m_k.data() + layer_offset;
    for (size_t i = 0; i < n_kv; ++i)
    {
        k_tiles[(i / head_dim) * tiles * head_dim * KEY_TILE + key_index(head_dim, p, i % head_dim)] = k_cur[i];
    }
}

void Runner::attention(const LlamaLayer& layer, int il, uint32_t pos, size_t n_tokens)
{
    const LlamaHparams& h = m_model->hparams();
    const size_t head_dim = h.head_dim;
    const size_t n_kv = static_cast<size_t>(h.n_head_kv) * head_dim;
    const size_t n_q = static_cast<size_t>(h.n_head) * head_dim;
    float* k_cur = m_k_batch.data();
    float* v_cur = m_kv_f16 ? m_v_batch.data() : m_v.data() + (static_cast<size_t>(il) * m_n_ctx + pos) * n_kv;

    matmul(layer.wq, m_h.data(), h.n_embd, n_tokens, m_q.data(), n_q);
    matmul(layer.wk, m_h.data(), h.n_embd, n_tokens, k_cur, n_kv);
    matmul(layer.wv, m_h.data(), h.n_embd, n_tokens, v_cur, n_kv);

    m_layer = static_cast<size_t>(il);
    for_tokens(&Runner::step_rope_store, &layer, pos, n_tokens);
    trace_rows("Qcur", il, m_q.data(), n_q, n_tokens, n_q);
    trace_rows("Kcur", il, k_cur, n_kv, n_tokens, n_kv);
    trace_rows("Vcur", il, v_cur, n_kv, n_tokens, n_kv);

    AttentionJob job{this, il, pos, n_tokens};
    m_pool.run(attention_task, &job);

    matmul(layer.wo, m_attn.data(), n_q, n_tokens, m_h.data(), h.n_embd);
    trace_rows("attn_out", il, m_h.data(), h.n_embd, n_tokens, h.n_embd);
}

void Runner::forward_batch(const int32_t* tokens, size_t count, uint32_t pos, float* logits)
{
    const LlamaHparams& h = m_model->hparams();
    const size_t n_embd = h.n_embd;

    for (size_t t = 0; t < count; ++t)
    {
        m_model->token_embd().row(static_cast<size_t>(tokens[t]), m_x.data() + t * n_embd);
    }
    trace_rows("inp_embd", -1, m_x.data(), n_embd, count, n_embd);

    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        const LlamaLayer& layer = m_model->layer(l);
        const int il = static_cast<int>(l);

        for_tokens(&Runner::step_attn_norm, &layer, pos, count);
        trace_rows("attn_norm", il, m_h.data(), n_embd, count, n_embd);

        attention(layer, il, pos, count);
        for_tokens(&Runner::step_ffn_inp, &layer, pos, count);
        trace_rows("ffn_inp", il, m_x.data(), n_embd, count, n_embd);
        trace_rows("ffn_norm", il, m_h.data(), n_embd, count, n_embd);

        matmul(layer.ffn_gate, m_h.data(), n_embd, count, m_gate.data(), h.n_ff);
        matmul(layer.ffn_up, m_h.data(), n_embd, count, m_up.data(), h.n_ff);
        trace_rows("ffn_gate", il, m_gate.data(), h.n_ff, count, h.n_ff);
        trace_rows("ffn_up", il, m_up.data(), h.n_ff, count, h.n_ff);
        for_tokens(&Runner::step_swiglu, &layer, pos, count);
        trace_rows("ffn_swiglu", il, m_act.data(), h.n_ff, count, h.n_ff);
        matmul(layer.ffn_down, m_act.data(), h.n_ff, count, m_h.data(), n_embd);
        trace_rows("ffn_out", il, m_h.data(), n_embd, count, n_embd);

        for_tokens(&Runner::step_residual, &layer, pos, count);
        trace_rows("l_out", il, m_x.data(), n_embd, count, n_embd);
    }

    if (m_trace != nullptr)
    {
        for_tokens(&Runner::step_output_norm, nullptr, pos, count);
        trace_rows("result_norm", -1, m_h.data(), n_embd, count, n_embd);
        matmul(m_model->output(), m_h.data(), n_embd, count, m_logits.data(), h.n_vocab);
        trace_rows("result_output", -1, m_logits.data(), h.n_vocab, count, h.n_vocab);
        std::memcpy(logits, m_logits.data() + (count - 1) * h.n_vocab, h.n_vocab * sizeof(float));
        return;
    }

    const float* last = m_x.data() + (count - 1) * n_embd;
    rms_norm(last, m_model->output_norm(), n_embd, h.rms_eps, m_h.data());
    matmul(m_model->output(), m_h.data(), n_embd, 1, logits, h.n_vocab);
}

}
