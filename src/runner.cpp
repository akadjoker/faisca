#include "faisca/runner.h"

#include "faisca/ops.h"

#include <cinttypes>
#include <cmath>
#include <cstring>

namespace faisca
{

void Runner::release()
{
    m_model = nullptr;
    m_n_ctx = 0;
    m_pos = 0;
    m_k.release();
    m_v.release();
    m_x.release();
    m_h.release();
    m_q.release();
    m_attn.release();
    m_gate.release();
    m_up.release();
    m_act.release();
    m_scores.release();
    m_input = nullptr;
    m_input_size = 0;
    m_input_quantized = false;
    m_input_q8.release();
}

bool Runner::init(const LlamaModel& model, uint32_t n_ctx, Error& error)
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
    if (!m_k.resize_uninitialized(cache) || !m_v.resize_uninitialized(cache) || !m_x.resize_uninitialized(h.n_embd) ||
        !m_h.resize_uninitialized(h.n_embd) || !m_q.resize_uninitialized(n_q) || !m_attn.resize_uninitialized(n_q) ||
        !m_gate.resize_uninitialized(h.n_ff) || !m_up.resize_uninitialized(h.n_ff) ||
        !m_act.resize_uninitialized(h.n_ff) || !m_scores.resize_uninitialized(n_ctx) ||
        !m_input_q8.resize_uninitialized((widest / Q8_0_BLOCK + 1) * Q8_0_BLOCK_BYTES))
    {
        release();
        error.set("out of memory: the KV cache for %" PRIu32 " positions needs %zu MiB", n_ctx,
                  2 * cache * sizeof(float) >> 20);
        return false;
    }

    m_model = &model;
    m_n_ctx = n_ctx;
    m_pos = 0;
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

    for (size_t i = 0; i < count; ++i)
    {
        const bool last = i + 1 == count;
        forward_one(tokens[i], m_pos, last || m_trace != nullptr ? logits.data() : nullptr);
        ++m_pos;
    }
    return true;
}

void Runner::trace(const char* name, int layer, const float* data, size_t count) const
{
    if (m_trace != nullptr)
    {
        m_trace(m_trace_user, name, layer, data, count);
    }
}

void Runner::set_input(const float* x, size_t n)
{
    m_input = x;
    m_input_size = n;
    m_input_quantized = false;
}

void Runner::matmul(const Weight& w, float* out)
{
    if (w.type == GgmlType::Q8_0)
    {
        if (!m_input_quantized)
        {
            quantize_q8_0(m_input, m_input_size, m_input_q8.data());
            m_input_quantized = true;
        }
        matvec_q8_0(w.data, m_input_q8.data(), w.n_in, w.n_out, out);
        return;
    }
    matvec(w.f32(), m_input, w.n_in, w.n_out, out);
}

void Runner::attention(const LlamaLayer& layer, int il, uint32_t pos)
{
    const LlamaHparams& h = m_model->hparams();
    const size_t head_dim = h.head_dim;
    const size_t n_kv = static_cast<size_t>(h.n_head_kv) * head_dim;
    const size_t group = h.n_head / h.n_head_kv;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    float* k_cache = m_k.data() + static_cast<size_t>(il) * m_n_ctx * n_kv;
    float* v_cache = m_v.data() + static_cast<size_t>(il) * m_n_ctx * n_kv;
    float* k_cur = k_cache + static_cast<size_t>(pos) * n_kv;
    float* v_cur = v_cache + static_cast<size_t>(pos) * n_kv;

    set_input(m_h.data(), h.n_embd);
    matmul(layer.wq, m_q.data());
    matmul(layer.wk, k_cur);
    matmul(layer.wv, v_cur);

    rope_norm(m_q.data(), h.n_head, head_dim, h.n_rot, pos, h.rope_freq_base);
    rope_norm(k_cur, h.n_head_kv, head_dim, h.n_rot, pos, h.rope_freq_base);
    trace("Qcur", il, m_q.data(), static_cast<size_t>(h.n_head) * head_dim);
    trace("Kcur", il, k_cur, n_kv);
    trace("Vcur", il, v_cur, n_kv);

    const size_t n_past = static_cast<size_t>(pos) + 1;
    for (size_t head = 0; head < h.n_head; ++head)
    {
        const float* q = m_q.data() + head * head_dim;
        const size_t kv_offset = (head / group) * head_dim;
        float* scores = m_scores.data();
        for (size_t j = 0; j < n_past; ++j)
        {
            scores[j] = dot(q, k_cache + j * n_kv + kv_offset, head_dim) * scale;
        }
        softmax(scores, n_past);

        float* out = m_attn.data() + head * head_dim;
        for (size_t d = 0; d < head_dim; ++d)
        {
            out[d] = 0.0f;
        }
        for (size_t j = 0; j < n_past; ++j)
        {
            const float p = scores[j];
            const float* v = v_cache + j * n_kv + kv_offset;
            for (size_t d = 0; d < head_dim; ++d)
            {
                out[d] += p * v[d];
            }
        }
    }

    set_input(m_attn.data(), static_cast<size_t>(h.n_head) * head_dim);
    matmul(layer.wo, m_h.data());
    trace("attn_out", il, m_h.data(), h.n_embd);
}

void Runner::forward_one(int32_t token, uint32_t pos, float* logits)
{
    const LlamaHparams& h = m_model->hparams();
    const size_t n_embd = h.n_embd;

    m_model->token_embd().row(static_cast<size_t>(token), m_x.data());
    trace("inp_embd", -1, m_x.data(), n_embd);

    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        const LlamaLayer& layer = m_model->layer(l);
        const int il = static_cast<int>(l);

        rms_norm(m_x.data(), layer.attn_norm, n_embd, h.rms_eps, m_h.data());
        trace("attn_norm", il, m_h.data(), n_embd);

        attention(layer, il, pos);
        add(m_x.data(), m_h.data(), n_embd);
        trace("ffn_inp", il, m_x.data(), n_embd);

        rms_norm(m_x.data(), layer.ffn_norm, n_embd, h.rms_eps, m_h.data());
        trace("ffn_norm", il, m_h.data(), n_embd);

        set_input(m_h.data(), n_embd);
        matmul(layer.ffn_gate, m_gate.data());
        matmul(layer.ffn_up, m_up.data());
        trace("ffn_gate", il, m_gate.data(), h.n_ff);
        trace("ffn_up", il, m_up.data(), h.n_ff);
        swiglu(m_gate.data(), m_up.data(), h.n_ff, m_act.data());
        trace("ffn_swiglu", il, m_act.data(), h.n_ff);
        set_input(m_act.data(), h.n_ff);
        matmul(layer.ffn_down, m_h.data());
        trace("ffn_out", il, m_h.data(), n_embd);

        add(m_x.data(), m_h.data(), n_embd);
        trace("l_out", il, m_x.data(), n_embd);
    }

    if (logits == nullptr)
    {
        return;
    }
    rms_norm(m_x.data(), m_model->output_norm(), n_embd, h.rms_eps, m_h.data());
    trace("result_norm", -1, m_h.data(), n_embd);
    set_input(m_h.data(), n_embd);
    matmul(m_model->output(), logits);
    trace("result_output", -1, logits, h.n_vocab);
}

}
