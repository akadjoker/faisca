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
    if (m_pool == &m_own_pool)
    {
        m_own_pool.stop();
    }
    m_pool = &m_own_pool;
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
    if (!m_own_pool.start(n_threads, error))
    {
        return false;
    }
    return init_buffers(model, n_ctx, kv_f16, error);
}

bool Runner::init(const LlamaModel& model, uint32_t n_ctx, ThreadPool& pool, Error& error, bool kv_f16)
{
    release();
    if (pool.size() < 1)
    {
        error.set("the shared thread pool is not started");
        return false;
    }
    m_pool = &pool;
    return init_buffers(model, n_ctx, kv_f16, error);
}

bool Runner::init_buffers(const LlamaModel& model, uint32_t n_ctx, bool kv_f16, Error& error)
{
    if (!model.is_loaded())
    {
        error.set("the model is not loaded");
        release();
        return false;
    }
    if (n_ctx == 0)
    {
        error.set("the context size must be at least 1");
        release();
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
    const size_t q8_0_row = (widest / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
    const size_t q8_k_row = (widest / Q8_K_BLOCK) * Q8_K_BLOCK_BYTES;
    m_xq_stride = q8_0_row > q8_k_row ? q8_0_row : q8_k_row;
    m_sums_stride = widest / 16;
    const size_t threads = static_cast<size_t>(m_pool->size());
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

size_t Runner::state_k_bytes(uint32_t pos) const
{
    const LlamaHparams& h = m_model->hparams();
    const size_t tiles = (static_cast<size_t>(pos) + KEY_TILE - 1) / KEY_TILE;
    const size_t element = m_kv_f16 ? sizeof(uint16_t) : sizeof(float);
    return static_cast<size_t>(h.n_head_kv) * tiles * h.head_dim * KEY_TILE * element;
}

size_t Runner::state_v_bytes(uint32_t pos) const
{
    const LlamaHparams& h = m_model->hparams();
    const size_t element = m_kv_f16 ? sizeof(uint16_t) : sizeof(float);
    return static_cast<size_t>(pos) * h.n_head_kv * h.head_dim * element;
}

size_t Runner::state_bytes(uint32_t pos) const
{
    return m_model == nullptr ? 0 : static_cast<size_t>(m_model->hparams().n_layer) * (state_k_bytes(pos) + state_v_bytes(pos));
}

bool Runner::export_state(uint8_t* out, size_t bytes) const
{
    if (m_model == nullptr || bytes != state_bytes(m_pos))
    {
        return false;
    }
    const LlamaHparams& h = m_model->hparams();
    const size_t element = m_kv_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t n_kv = static_cast<size_t>(h.n_head_kv) * h.head_dim;
    const size_t tiles_all = (static_cast<size_t>(m_n_ctx) + KEY_TILE - 1) / KEY_TILE;
    const size_t head_bytes = tiles_all * h.head_dim * KEY_TILE * element;
    const size_t k_head = state_k_bytes(m_pos) / h.n_head_kv;
    const size_t v_bytes = state_v_bytes(m_pos);
    const uint8_t* k_base = m_kv_f16 ? reinterpret_cast<const uint8_t*>(m_k16.data()) : reinterpret_cast<const uint8_t*>(m_k.data());
    const uint8_t* v_base = m_kv_f16 ? reinterpret_cast<const uint8_t*>(m_v16.data()) : reinterpret_cast<const uint8_t*>(m_v.data());
    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        for (uint32_t head = 0; head < h.n_head_kv; ++head)
        {
            std::memcpy(out, k_base + (static_cast<size_t>(l) * h.n_head_kv + head) * head_bytes, k_head);
            out += k_head;
        }
        std::memcpy(out, v_base + static_cast<size_t>(l) * m_n_ctx * n_kv * element, v_bytes);
        out += v_bytes;
    }
    return true;
}

bool Runner::import_state(const uint8_t* data, size_t bytes, uint32_t pos, Error& error)
{
    if (m_model == nullptr)
    {
        error.set("the runner is not initialized");
        return false;
    }
    if (pos > m_n_ctx)
    {
        error.set("a state of %" PRIu32 " positions does not fit a context of %" PRIu32, pos, m_n_ctx);
        return false;
    }
    if (bytes != state_bytes(pos))
    {
        error.set("a state of %" PRIu32 " positions takes %zu bytes, %zu given", pos, state_bytes(pos), bytes);
        return false;
    }
    const LlamaHparams& h = m_model->hparams();
    const size_t element = m_kv_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t n_kv = static_cast<size_t>(h.n_head_kv) * h.head_dim;
    const size_t tiles_all = (static_cast<size_t>(m_n_ctx) + KEY_TILE - 1) / KEY_TILE;
    const size_t head_bytes = tiles_all * h.head_dim * KEY_TILE * element;
    const size_t k_head = state_k_bytes(pos) / h.n_head_kv;
    const size_t v_bytes = state_v_bytes(pos);
    uint8_t* k_base = m_kv_f16 ? reinterpret_cast<uint8_t*>(m_k16.data()) : reinterpret_cast<uint8_t*>(m_k.data());
    uint8_t* v_base = m_kv_f16 ? reinterpret_cast<uint8_t*>(m_v16.data()) : reinterpret_cast<uint8_t*>(m_v.data());
    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        for (uint32_t head = 0; head < h.n_head_kv; ++head)
        {
            std::memcpy(k_base + (static_cast<size_t>(l) * h.n_head_kv + head) * head_bytes, data, k_head);
            data += k_head;
        }
        std::memcpy(v_base + static_cast<size_t>(l) * m_n_ctx * n_kv * element, data, v_bytes);
        data += v_bytes;
    }
    m_pos = pos;
    return true;
}

bool Runner::forward(const int32_t* tokens, size_t count, Array<float>& logits, Error& error, bool all_logits)
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
    if (!logits.resize_uninitialized(all_logits ? count * h.n_vocab : h.n_vocab))
    {
        error.set("out of memory");
        return false;
    }
    if (m_trace != nullptr && !all_logits && !m_logits.resize_uninitialized(BATCH * h.n_vocab))
    {
        error.set("out of memory");
        return false;
    }

    for (size_t done = 0; done < count;)
    {
        const size_t batch = count - done < BATCH ? count - done : BATCH;
        forward_batch(tokens + done, batch, m_pos, logits.data() + (all_logits ? done * h.n_vocab : 0), all_logits);
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
        const float* x = runner.m_quantize_x + t * runner.m_quantize_stride;
        uint8_t* xq = runner.m_xq.data() + t * runner.m_xq_stride;
        int32_t* sums = runner.m_sums.data() + t * runner.m_sums_stride;
        if (runner.m_quantize_kind == Activation::Q8_K)
        {
            quantize_q8_k(x, runner.m_quantize_n, xq);
            q8_k_block_sums(xq, runner.m_quantize_n, sums);
        }
        else
        {
            quantize_q8_0(x, runner.m_quantize_n, xq);
            q8_0_block_sums(xq, runner.m_quantize_n, sums);
        }
    }
}

void Runner::quantize_inputs(const float* x, size_t stride, size_t n_tokens, size_t n, Activation kind)
{
    m_quantize_x = x;
    m_quantize_stride = stride;
    m_quantize_tokens = n_tokens;
    m_quantize_n = n;
    m_quantize_kind = kind;
    m_pool->run(quantize_task, this);
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
    m_pool->run(token_task, &job);
}

void Runner::step_scale_embd(const LlamaLayer*, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    float* x = m_x.data() + token * h.n_embd;
    for (size_t i = 0; i < h.n_embd; ++i)
    {
        x[i] *= h.embd_scale;
    }
}

void Runner::step_attn_norm(const LlamaLayer* layer, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    rms_norm(m_x.data() + token * h.n_embd, layer->attn_norm, h.n_embd, h.rms_eps, m_h.data() + token * h.n_embd);
}

void Runner::step_qk_norm(const LlamaLayer* layer, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    const size_t head_dim = h.head_dim;
    float* q = m_q.data() + token * h.n_head * head_dim;
    float* k = m_k_batch.data() + token * h.n_head_kv * head_dim;
    for (size_t i = 0; i < h.n_head; ++i)
    {
        rms_norm(q + i * head_dim, layer->attn_q_norm, head_dim, h.rms_eps, q + i * head_dim);
    }
    for (size_t i = 0; i < h.n_head_kv; ++i)
    {
        rms_norm(k + i * head_dim, layer->attn_k_norm, head_dim, h.rms_eps, k + i * head_dim);
    }
}

void Runner::step_attn_post_norm(const LlamaLayer* layer, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    float* row = m_h.data() + token * h.n_embd;
    rms_norm(row, layer->attn_post_norm, h.n_embd, h.rms_eps, row);
}

void Runner::step_ffn_post_norm(const LlamaLayer* layer, uint32_t, size_t token)
{
    const LlamaHparams& h = m_model->hparams();
    float* row = m_h.data() + token * h.n_embd;
    rms_norm(row, layer->ffn_post_norm, h.n_embd, h.rms_eps, row);
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
    if (h.gelu_ffn)
    {
        geglu(m_gate.data() + token * h.n_ff, m_up.data() + token * h.n_ff, h.n_ff, m_act.data() + token * h.n_ff);
        return;
    }
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
    if (w.activation() != Activation::F32)
    {
        quantize_inputs(x, x_stride, n_tokens, w.n_in, w.activation());
    }
    MatmulJob job{this, &w, x, x_stride, n_tokens, out, out_stride};
    m_pool->run(matmul_task, &job);
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
    const float scale = h.attn_scale != 0.0f ? 1.0f : 1.0f / std::sqrt(static_cast<float>(head_dim));

    const size_t tiles = (static_cast<size_t>(m_n_ctx) + 15) / 16;
    const size_t k_offset = (static_cast<size_t>(il) * h.n_head_kv + kv_head) * tiles * head_dim * KEY_TILE;
    const size_t v_offset = static_cast<size_t>(il) * m_n_ctx * n_kv + kv_head * head_dim;
    const float* q = m_q.data() + token * n_q + head * head_dim;
    float* out = m_attn.data() + token * n_q + head * head_dim;
    size_t n_past = static_cast<size_t>(pos) + token + 1;
    // a local layer sees the last n_swa positions: the window starts at a whole tile and
    // the kernel skips the positions before the start inside it
    size_t start_tile = 0;
    size_t skip = 0;
    if (h.local_layer(static_cast<uint32_t>(il)) && n_past > h.n_swa)
    {
        const size_t start = n_past - h.n_swa;
        start_tile = start / KEY_TILE;
        skip = start % KEY_TILE;
        n_past -= start_tile * KEY_TILE;
    }
    const size_t k_start = k_offset + start_tile * head_dim * KEY_TILE;
    const size_t v_start = v_offset + start_tile * KEY_TILE * n_kv;
    if (m_kv_f16)
    {
        attention_head_f16(q, m_k16.data() + k_start, m_v16.data() + v_start, n_kv, head_dim, n_past, scale, scores, out,
                           skip);
    }
    else
    {
        attention_head(q, m_k.data() + k_start, m_v.data() + v_start, n_kv, head_dim, n_past, scale, scores, out, skip);
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
    float* q = m_q.data() + token * n_q;
    const float* factors = m_model->rope_factors();
    const bool local = h.local_layer(static_cast<uint32_t>(m_layer));
    const float freq_base = local ? h.rope_freq_base_swa : h.rope_freq_base;
    const float freq_scale = local ? 1.0f : h.rope_freq_scale;
    if (h.rope_neox)
    {
        rope_neox(q, h.n_head, head_dim, h.n_rot, p, freq_base, freq_scale);
        rope_neox(k_cur, h.n_head_kv, head_dim, h.n_rot, p, freq_base, freq_scale);
    }
    else
    {
        rope_norm(q, h.n_head, head_dim, h.n_rot, p, freq_base, factors, freq_scale);
        rope_norm(k_cur, h.n_head_kv, head_dim, h.n_rot, p, freq_base, factors, freq_scale);
    }

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
    if (layer.bq != nullptr)
    {
        for (size_t t = 0; t < n_tokens; ++t)
        {
            add_bias(m_q.data() + t * n_q, layer.bq, n_q);
            add_bias(k_cur + t * n_kv, layer.bk, n_kv);
            add_bias(v_cur + t * n_kv, layer.bv, n_kv);
        }
    }

    m_layer = static_cast<size_t>(il);
    if (h.qk_norm)
    {
        for_tokens(&Runner::step_qk_norm, &layer, pos, n_tokens);
        trace_rows("Qcur_normed", il, m_q.data(), n_q, n_tokens, n_q);
        trace_rows("Kcur_normed", il, k_cur, n_kv, n_tokens, n_kv);
    }
    for_tokens(&Runner::step_rope_store, &layer, pos, n_tokens);
    trace_rows("Qcur", il, m_q.data(), n_q, n_tokens, n_q);
    trace_rows("Kcur", il, k_cur, n_kv, n_tokens, n_kv);
    trace_rows("Vcur", il, v_cur, n_kv, n_tokens, n_kv);
    if (h.attn_scale != 0.0f)
    {
        float* q = m_q.data();
        for (size_t i = 0; i < n_tokens * n_q; ++i)
        {
            q[i] *= h.attn_scale;
        }
    }

    AttentionJob job{this, il, pos, n_tokens};
    m_pool->run(attention_task, &job);

    matmul(layer.wo, m_attn.data(), n_q, n_tokens, m_h.data(), h.n_embd);
    trace_rows("attn_out", il, m_h.data(), h.n_embd, n_tokens, h.n_embd);
    if (h.post_norms)
    {
        for_tokens(&Runner::step_attn_post_norm, &layer, pos, n_tokens);
        trace_rows("attn_post_norm", il, m_h.data(), h.n_embd, n_tokens, h.n_embd);
    }
}

void Runner::forward_batch(const int32_t* tokens, size_t count, uint32_t pos, float* logits, bool all_logits)
{
    const LlamaHparams& h = m_model->hparams();
    const size_t n_embd = h.n_embd;

    for (size_t t = 0; t < count; ++t)
    {
        m_model->token_embd().row(static_cast<size_t>(tokens[t]), m_x.data() + t * n_embd);
    }
    trace_rows("inp_embd", -1, m_x.data(), n_embd, count, n_embd);
    if (h.embd_scale != 1.0f)
    {
        for_tokens(&Runner::step_scale_embd, nullptr, pos, count);
        trace_rows("inp_scaled", -1, m_x.data(), n_embd, count, n_embd);
    }

    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        const LlamaLayer& layer = m_model->layer(l);
        const int il = static_cast<int>(l);

        for_tokens(&Runner::step_attn_norm, &layer, pos, count);
        trace_rows("attn_norm", il, m_h.data(), n_embd, count, n_embd);

        attention(layer, il, pos, count);
        for_tokens(&Runner::step_ffn_inp, &layer, pos, count);
        trace_rows(h.post_norms ? "sa_out" : "ffn_inp", il, m_x.data(), n_embd, count, n_embd);
        trace_rows("ffn_norm", il, m_h.data(), n_embd, count, n_embd);

        matmul(layer.ffn_gate, m_h.data(), n_embd, count, m_gate.data(), h.n_ff);
        matmul(layer.ffn_up, m_h.data(), n_embd, count, m_up.data(), h.n_ff);
        trace_rows("ffn_gate", il, m_gate.data(), h.n_ff, count, h.n_ff);
        trace_rows("ffn_up", il, m_up.data(), h.n_ff, count, h.n_ff);
        for_tokens(&Runner::step_swiglu, &layer, pos, count);
        trace_rows(h.gelu_ffn ? "ffn_geglu" : "ffn_swiglu", il, m_act.data(), h.n_ff, count, h.n_ff);
        matmul(layer.ffn_down, m_act.data(), h.n_ff, count, m_h.data(), n_embd);
        trace_rows("ffn_out", il, m_h.data(), n_embd, count, n_embd);
        if (h.post_norms)
        {
            for_tokens(&Runner::step_ffn_post_norm, &layer, pos, count);
            trace_rows("ffn_post_norm", il, m_h.data(), n_embd, count, n_embd);
        }

        for_tokens(&Runner::step_residual, &layer, pos, count);
        trace_rows("l_out", il, m_x.data(), n_embd, count, n_embd);
    }

    if (m_trace != nullptr || all_logits)
    {
        float* rows = all_logits ? logits : m_logits.data();
        for_tokens(&Runner::step_output_norm, nullptr, pos, count);
        trace_rows("result_norm", -1, m_h.data(), n_embd, count, n_embd);
        matmul(m_model->output(), m_h.data(), n_embd, count, rows, h.n_vocab);
        softcap(rows, count * h.n_vocab);
        trace_rows("result_output", -1, rows, h.n_vocab, count, h.n_vocab);
        if (!all_logits)
        {
            std::memcpy(logits, rows + (count - 1) * h.n_vocab, h.n_vocab * sizeof(float));
        }
        return;
    }

    const float* last = m_x.data() + (count - 1) * n_embd;
    rms_norm(last, m_model->output_norm(), n_embd, h.rms_eps, m_h.data());
    matmul(m_model->output(), m_h.data(), n_embd, 1, logits, h.n_vocab);
    softcap(logits, h.n_vocab);
}

void Runner::softcap(float* logits, size_t count) const
{
    const float cap = m_model->hparams().final_softcap;
    if (cap == 0.0f)
    {
        return;
    }
    for (size_t i = 0; i < count; ++i)
    {
        logits[i] = cap * std::tanh(logits[i] / cap);
    }
}

}
