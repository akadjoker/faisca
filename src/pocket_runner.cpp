#include "faisca/pocket_runner.h"

#include "faisca/ops.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

constexpr size_t EOS_MIN_FRAMES = 6;
constexpr float EOS_THRESHOLD = -4.0f;

float silu(float x)
{
    return x / (1.0f + expf(-x));
}

void silu_values(float* x, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = silu(x[i]);
    }
}

// (x - mean) / sqrt(var + eps), the affine part optional
void norm_values(const float* x, const float* w, const float* b, size_t n, float eps, float* out)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += x[i];
    }
    const float mean = static_cast<float>(sum / static_cast<double>(n));
    double squares = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const float centered = x[i] - mean;
        out[i] = centered;
        squares += static_cast<double>(centered) * static_cast<double>(centered);
    }
    const float scale = 1.0f / sqrtf(static_cast<float>(squares / static_cast<double>(n)) + eps);
    for (size_t i = 0; i < n; ++i)
    {
        float v = out[i] * scale;
        if (w != nullptr)
        {
            v *= w[i];
        }
        if (b != nullptr)
        {
            v += b[i];
        }
        out[i] = v;
    }
}

void add_scaled(float* x, const float* y, const float* scale, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        x[i] += y[i] * scale[i];
    }
}

}

bool PocketVoice::load(const char* path, const char* name, const PocketModel& model, Error& error)
{
    GgufFile file;
    if (!file.open(path, error))
    {
        return false;
    }
    char key[256];
    std::snprintf(key, sizeof(key), "%s.k", name);
    const GgufTensor* kt = file.find_tensor(key);
    std::snprintf(key, sizeof(key), "%s.v", name);
    const GgufTensor* vt = file.find_tensor(key);
    const PocketHparams& h = model.hparams();
    if (kt == nullptr || vt == nullptr)
    {
        error.set("voice '%s' is not in '%s'", name, path);
        return false;
    }
    if (kt->type != GgmlType::F32 || vt->type != GgmlType::F32 || kt->n_dims != 3 || vt->n_dims != 3 || kt->dims[0] != h.dim ||
        kt->dims[2] != h.n_layers || vt->dims[0] != kt->dims[0] || vt->dims[1] != kt->dims[1] || vt->dims[2] != kt->dims[2])
    {
        error.set("voice '%s' does not match the model (F32 [%u, n, %u] expected)", name, h.dim, h.n_layers);
        return false;
    }
    n = static_cast<uint32_t>(kt->dims[1]);
    const size_t count = static_cast<size_t>(kt->n_elements);
    if (!k.resize_uninitialized(count) || !v.resize_uninitialized(count))
    {
        error.set("out of memory for the voice");
        return false;
    }
    std::memcpy(k.data(), kt->data, count * sizeof(float));
    std::memcpy(v.data(), vt->data, count * sizeof(float));
    return true;
}

TorchRandom::TorchRandom(uint32_t seed)
{
    m_state[0] = seed;
    for (size_t i = 1; i < 624; ++i)
    {
        m_state[i] = 1812433253u * (m_state[i - 1] ^ (m_state[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
    m_index = 624;
}

uint32_t TorchRandom::next()
{
    if (m_index >= 624)
    {
        for (size_t i = 0; i < 624; ++i)
        {
            const uint32_t y = (m_state[i] & 0x80000000u) | (m_state[(i + 1) % 624] & 0x7fffffffu);
            m_state[i] = m_state[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1u) != 0 ? 0x9908b0dfu : 0u);
        }
        m_index = 0;
    }
    uint32_t y = m_state[m_index++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

float TorchRandom::uniform()
{
    return static_cast<float>(static_cast<double>(next() & 0xffffffu) * (1.0 / 16777216.0));
}

void TorchRandom::normal(float* x, size_t n, float std)
{
    const auto fill16 = [std](float* b)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            const float u1 = 1.0f - b[j];
            const float u2 = b[j + 8];
            const float r = sqrtf(-2.0f * logf(u1));
            const float theta = 2.0f * 3.14159265358979323846f * u2;
            b[j] = r * cosf(theta) * std;
            b[j + 8] = r * sinf(theta) * std;
        }
    };
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = uniform();
    }
    for (size_t i = 0; i + 16 <= n; i += 16)
    {
        fill16(x + i);
    }
    if (n % 16 != 0)
    {
        float* t = x + n - 16;
        for (size_t i = 0; i < 16; ++i)
        {
            t[i] = uniform();
        }
        fill16(t);
    }
}

struct PocketRunner::RowsJob
{
    const float* x;
    const float* w;
    const float* b;
    size_t rows;
    size_t width;
    float eps;
    float* out;
    int mode; // 0 norm, 1 gelu, 2 elu
};

struct PocketRunner::MatmulJob
{
    const Weight* weight;
    const float* x;
    const uint8_t* xq;
    size_t xq_stride;
    const int32_t* sums;
    size_t sums_stride;
    size_t n_tokens;
    float* out;
};

struct PocketRunner::AttentionJob
{
    PocketRunner* runner;
    const float* q;
    size_t q_stride;
    const float* k;
    const float* v;
    size_t kv_stride;
    size_t n_q;
    size_t position;
    size_t n_head;
    size_t head_dim;
    size_t window;
    float scale;
    float* out;
    size_t out_stride;
};

struct PocketRunner::ConvJob
{
    const PocketConv* c;
    const float* x;
    size_t length;
    float* out;
};

struct PocketRunner::ConvTrJob
{
    const PocketConv* c;
    const float* x;
    size_t length;
    float* out;
    float* tmp;
};

struct PocketRunner::UpsampleJob
{
    const float* x; // [n][channels]
    size_t n;
    size_t channels;
    const float* w; // [channels][k]
    size_t k;
    size_t stride;
    float* out; // [n * stride][channels]
};

bool PocketRunner::init(const PocketModel& model, ThreadPool& pool, Error& error)
{
    m_model = &model;
    m_pool = &pool;
    const PocketHparams& h = model.hparams();
    if (h.outer_dim != h.tfm_dim)
    {
        error.set("Mimi outer dim %u differs from its transformer dim %u", h.outer_dim, h.tfm_dim);
        return false;
    }
    const size_t head = static_cast<size_t>(h.head_dim);
    if (!m_head.resize_uninitialized(16 * head + 3 * head + 2 * static_cast<size_t>(h.ldim)))
    {
        error.set("out of memory for the flow head");
        return false;
    }
    return true;
}

void PocketRunner::trace(const char* name, int layer, const float* data, size_t count) const
{
    if (m_trace != nullptr)
    {
        m_trace(m_trace_user, name, layer, data, count);
    }
}

bool PocketRunner::reserve_rows(size_t rows, size_t width, size_t ffn, Error& error)
{
    const size_t widest = ffn > 3 * width ? ffn : 3 * width;
    const size_t threads = static_cast<size_t>(m_pool->size());
    const size_t xq_stride = (widest + Q8_0_BLOCK - 1) / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
    const size_t sums_stride = (widest + Q8_0_BLOCK - 1) / Q8_0_BLOCK;
    const size_t kv = m_cache_rows > static_cast<size_t>(m_model->hparams().tfm_context) ? m_cache_rows
                                                                                          : static_cast<size_t>(m_model->hparams().tfm_context);
    const bool ok = (m_h.size() >= rows * width || m_h.resize_uninitialized(rows * width)) &&
                    (m_proj.size() >= rows * 3 * width || m_proj.resize_uninitialized(rows * 3 * width)) &&
                    (m_attn.size() >= rows * width || m_attn.resize_uninitialized(rows * width)) &&
                    (m_up.size() >= rows * ffn || m_up.resize_uninitialized(rows * ffn)) &&
                    (m_xq.size() >= rows * xq_stride || m_xq.resize_uninitialized(rows * xq_stride)) &&
                    (m_sums.size() >= rows * sums_stride || m_sums.resize_uninitialized(rows * sums_stride)) &&
                    (m_scores.size() >= threads * (kv + KEY_TILE) || m_scores.resize_uninitialized(threads * (kv + KEY_TILE)));
    if (!ok)
    {
        error.set("out of memory for the transformer rows");
        return false;
    }
    m_xq_stride = xq_stride;
    m_sums_stride = sums_stride;
    return true;
}

void PocketRunner::rows_task(void* user, int thread_index, int thread_count)
{
    const RowsJob& job = *static_cast<const RowsJob*>(user);
    const size_t first = job.rows * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = job.rows * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t r = first; r < last; ++r)
    {
        float* out = job.out + r * job.width;
        if (job.mode == 0)
        {
            norm_values(job.x + r * job.width, job.w, job.b, job.width, job.eps, out);
        }
        else if (job.mode == 1)
        {
            gelu(out, job.width);
        }
        else
        {
            for (size_t i = 0; i < job.width; ++i)
            {
                out[i] = out[i] > 0.0f ? out[i] : expm1f(out[i]);
            }
        }
    }
}

void PocketRunner::norm_rows(const float* x, const float* w, const float* b, size_t rows, size_t width, float eps, float* out)
{
    RowsJob job{x, w, b, rows, width, eps, out, 0};
    m_pool->run(rows_task, &job);
}

void PocketRunner::gelu_rows(float* x, size_t rows, size_t width)
{
    RowsJob job{nullptr, nullptr, nullptr, rows, width, 0.0f, x, 1};
    m_pool->run(rows_task, &job);
}

void PocketRunner::elu_rows(float* x, size_t count)
{
    const size_t width = 1024;
    const size_t rows = count / width;
    RowsJob job{nullptr, nullptr, nullptr, rows, width, 0.0f, x, 2};
    m_pool->run(rows_task, &job);
    for (size_t i = rows * width; i < count; ++i)
    {
        x[i] = x[i] > 0.0f ? x[i] : expm1f(x[i]);
    }
}

void PocketRunner::matmul_task(void* user, int thread_index, int thread_count)
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
    w.matmul(job.x, w.n_in, job.xq, job.xq_stride, job.sums, job.sums_stride, job.n_tokens, first, last, job.out, w.n_out);
}

void PocketRunner::quantize_task(void* user, int thread_index, int thread_count)
{
    PocketRunner& runner = *static_cast<PocketRunner*>(user);
    const size_t first = runner.m_quantize_tokens * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = runner.m_quantize_tokens * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t t = first; t < last; ++t)
    {
        uint8_t* xq = runner.m_xq.data() + t * runner.m_xq_stride;
        quantize_q8_0(runner.m_quantize_x + t * runner.m_quantize_n, runner.m_quantize_n, xq);
        q8_0_block_sums(xq, runner.m_quantize_n, runner.m_sums.data() + t * runner.m_sums_stride);
    }
}

void PocketRunner::quantize_inputs(const float* x, size_t n_tokens, size_t n)
{
    m_quantize_x = x;
    m_quantize_tokens = n_tokens;
    m_quantize_n = n;
    m_pool->run(quantize_task, this);
}

void PocketRunner::matmul(const Weight& w, const float* x, size_t n_tokens, float* out)
{
    MatmulJob job{&w, x, nullptr, 0, nullptr, 0, n_tokens, out};
    if (w.activation() != Activation::F32)
    {
        quantize_inputs(x, n_tokens, w.n_in);
        job.xq = m_xq.data();
        job.xq_stride = m_xq_stride;
        job.sums = m_sums.data();
        job.sums_stride = m_sums_stride;
    }
    m_pool->run(matmul_task, &job);
}

void PocketRunner::attention_task(void* user, int thread_index, int thread_count)
{
    const AttentionJob& job = *static_cast<const AttentionJob*>(user);
    const size_t units = job.n_head * job.n_q;
    const size_t first = units * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = units * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    float* scores = job.runner->m_scores.data() +
                    static_cast<size_t>(thread_index) * (job.runner->m_scores.size() / static_cast<size_t>(thread_count));
    for (size_t u = first; u < last; ++u)
    {
        const size_t head = u / job.n_q;
        const size_t i = u % job.n_q;
        const size_t p = job.position + i;
        const size_t start = job.window != 0 && p + 1 > job.window ? p + 1 - job.window : 0;
        attention_rows(job.q + i * job.q_stride + head * job.head_dim, job.k + start * job.kv_stride + head * job.head_dim,
                       job.v + start * job.kv_stride + head * job.head_dim, job.kv_stride, job.head_dim, p - start + 1, job.scale,
                       scores, job.out + i * job.out_stride + head * job.head_dim);
    }
}

void PocketRunner::attention(const float* q, size_t q_stride, const float* k, const float* v, size_t kv_stride, size_t n_q,
                             size_t position, size_t n_head, size_t head_dim, size_t window, float* out, size_t out_stride)
{
    AttentionJob job{this, q, q_stride, k, v, kv_stride, n_q, position, n_head, head_dim, window,
                     1.0f / sqrtf(static_cast<float>(head_dim)), out, out_stride};
    m_pool->run(attention_task, &job);
}

void PocketRunner::transformer_layer(const PocketTfmLayer& layer, float* x, size_t n_tokens, size_t width, size_t n_head,
                                     float max_period, size_t position, float* k, float* v, size_t kv_stride, size_t window)
{
    const size_t head_dim = width / n_head;
    float* h = m_h.data();
    float* proj = m_proj.data();
    norm_rows(x, layer.norm1_w, layer.norm1_b, n_tokens, width, 1e-5f, h);
    matmul(layer.in_proj, h, n_tokens, proj);
    for (size_t t = 0; t < n_tokens; ++t)
    {
        float* row = proj + t * 3 * width;
        const uint32_t pos = static_cast<uint32_t>(position + t);
        rope_norm(row, n_head, head_dim, head_dim, pos, max_period);
        rope_norm(row + width, n_head, head_dim, head_dim, pos, max_period);
        std::memcpy(k + (position + t) * kv_stride, row + width, width * sizeof(float));
        std::memcpy(v + (position + t) * kv_stride, row + 2 * width, width * sizeof(float));
    }
    attention(proj, 3 * width, k, v, kv_stride, n_tokens, position, n_head, head_dim, window, m_attn.data(), width);
    trace("attn", m_trace_layer, m_attn.data(), n_tokens * width);
    matmul(layer.out_proj, m_attn.data(), n_tokens, h);
    trace("attn_out", m_trace_layer, h, n_tokens * width);
    for (size_t t = 0; t < n_tokens; ++t)
    {
        if (layer.ls1 != nullptr)
        {
            add_scaled(x + t * width, h + t * width, layer.ls1, width);
        }
        else
        {
            add(x + t * width, h + t * width, width);
        }
    }
    trace("res_attn", m_trace_layer, x, n_tokens * width);
    norm_rows(x, layer.norm2_w, layer.norm2_b, n_tokens, width, 1e-5f, h);
    trace("norm2", m_trace_layer, h, n_tokens * width);
    matmul(layer.linear1, h, n_tokens, m_up.data());
    trace("up", m_trace_layer, m_up.data(), n_tokens * layer.linear1.n_out);
    gelu_rows(m_up.data(), n_tokens, layer.linear1.n_out);
    matmul(layer.linear2, m_up.data(), n_tokens, h);
    trace("down", m_trace_layer, h, n_tokens * width);
    for (size_t t = 0; t < n_tokens; ++t)
    {
        if (layer.ls2 != nullptr)
        {
            add_scaled(x + t * width, h + t * width, layer.ls2, width);
        }
        else
        {
            add(x + t * width, h + t * width, width);
        }
    }
    trace("res_ffn", m_trace_layer, x, n_tokens * width);
}

bool PocketRunner::flow_prefill(const int32_t* ids, size_t count, size_t position, Error& error)
{
    const PocketHparams& h = m_model->hparams();
    const size_t dim = h.dim;
    if (!reserve_rows(count, dim, h.ffn_dim, error) || (m_x.size() < count * dim && !m_x.resize_uninitialized(count * dim)))
    {
        error.set("out of memory for the prefill");
        return false;
    }
    for (size_t t = 0; t < count; ++t)
    {
        std::memcpy(m_x.data() + t * dim, m_model->embed_rows() + static_cast<size_t>(ids[t]) * dim, dim * sizeof(float));
    }
    trace("text-embed", -1, m_x.data(), count * dim);
    const size_t layer_floats = m_cache_rows * dim;
    for (size_t l = 0; l < h.n_layers; ++l)
    {
        m_trace_layer = 100 + static_cast<int>(l);
        transformer_layer(m_model->layer(l), m_x.data(), count, dim, h.n_heads, h.max_period, position,
                          m_cache_k.data() + l * layer_floats, m_cache_v.data() + l * layer_floats, dim, 0);
    }
    m_trace_layer = -1;
    return true;
}

bool PocketRunner::flow_step(const float* latent, const float* noise, size_t position, float& eos, float* out, Error& error)
{
    const PocketHparams& h = m_model->hparams();
    const size_t dim = h.dim;
    if (!reserve_rows(1, dim, h.ffn_dim, error) || (m_x.size() < dim && !m_x.resize_uninitialized(dim)))
    {
        return false;
    }
    matmul(m_model->input_linear(), latent, 1, m_x.data());
    const size_t layer_floats = m_cache_rows * dim;
    for (size_t l = 0; l < h.n_layers; ++l)
    {
        m_trace_layer = static_cast<int>(l);
        transformer_layer(m_model->layer(l), m_x.data(), 1, dim, h.n_heads, h.max_period, position,
                          m_cache_k.data() + l * layer_floats, m_cache_v.data() + l * layer_floats, dim, 0);
    }
    m_trace_layer = -1;
    float* c = m_h.data();
    norm_values(m_x.data(), m_model->out_norm_w(), m_model->out_norm_b(), dim, 1e-5f, c);
    trace("hidden", -1, c, dim);
    eos = dot(m_model->out_eos_w(), c, dim) + m_model->out_eos_b();
    flow_head(c, noise, out);
    return true;
}

void PocketRunner::time_embed(const PocketTimeEmbed& te, float t, float* out)
{
    const size_t n = te.l0.n_out;
    const size_t n_freqs = te.l0.n_in / 2;
    float* args = m_head.data() + 12 * n;
    float* e = args + 2 * n_freqs;
    for (size_t i = 0; i < n_freqs; ++i)
    {
        const float a = te.freqs[i] * t;
        args[i] = cosf(a);
        args[n_freqs + i] = sinf(a);
    }
    matmul(te.l0, args, 1, e);
    add_bias(e, te.l0_b, n);
    silu_values(e, n);
    matmul(te.l2, e, 1, out);
    add_bias(out, te.l2_b, n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += out[i];
    }
    const float mean = static_cast<float>(sum) / static_cast<float>(n);
    double squares = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const float d = out[i] - mean;
        squares += static_cast<double>(d * d);
    }
    const float var = static_cast<float>(squares) * (1.0f / static_cast<float>(n - 1)) + 1e-5f;
    const float inv = sqrtf(var);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = out[i] * te.alpha[i] / inv;
    }
}

void PocketRunner::flow_head(const float* cond, const float* noise, float* out)
{
    const PocketHparams& hp = m_model->hparams();
    const size_t n = hp.head_dim;
    const size_t ldim = hp.ldim;
    float* yc = m_head.data();
    float* y = yc + n;
    float* ys = y + n;
    float* h = ys + n;
    float* u = h + n;
    float* te0 = u + n;
    float* te1 = te0 + n;
    float* ada = te1 + n; // 3 n
    float* x = ada + 3 * n;
    float* step = x + ldim;
    matmul(m_model->cond(), cond, 1, yc);
    add_bias(yc, m_model->cond_b(), n);
    std::memcpy(x, noise, ldim * sizeof(float));
    const int n_steps = m_lsd_steps;
    for (int i = 0; i < n_steps; ++i)
    {
        const float s = static_cast<float>(i) / static_cast<float>(n_steps);
        const float t = static_cast<float>(i + 1) / static_cast<float>(n_steps);
        std::memcpy(y, yc, n * sizeof(float));
        if (hp.n_time_conds == 2)
        {
            time_embed(m_model->time_embed(0), s, te0);
            time_embed(m_model->time_embed(1), t, te1);
            for (size_t j = 0; j < n; ++j)
            {
                y[j] += (te0[j] + te1[j]) * 0.5f;
            }
        }
        else if (hp.n_time_conds == 1)
        {
            time_embed(m_model->time_embed(0), s, te0);
            add(y, te0, n);
        }
        std::memcpy(ys, y, n * sizeof(float));
        silu_values(ys, n);

        matmul(m_model->head_in(), x, 1, h);
        add_bias(h, m_model->head_in_b(), n);
        for (size_t b = 0; b < hp.head_depth; ++b)
        {
            const PocketResBlock& blk = m_model->res_block(b);
            matmul(blk.ada, ys, 1, ada);
            add_bias(ada, blk.ada_b, 3 * n);
            const float* shift = ada;
            const float* scale = ada + n;
            const float* gate = ada + 2 * n;
            norm_values(h, blk.ln_w, blk.ln_b, n, 1e-6f, u);
            for (size_t j = 0; j < n; ++j)
            {
                u[j] = (u[j] + u[j] * scale[j]) + shift[j];
            }
            matmul(blk.mlp0, u, 1, te0);
            add_bias(te0, blk.mlp0_b, n);
            silu_values(te0, n);
            matmul(blk.mlp2, te0, 1, u);
            add_bias(u, blk.mlp2_b, n);
            add_scaled(h, u, gate, n);
        }
        matmul(m_model->final_ada(), ys, 1, ada);
        add_bias(ada, m_model->final_ada_b(), 2 * n);
        norm_values(h, nullptr, nullptr, n, 1e-6f, u);
        for (size_t j = 0; j < n; ++j)
        {
            u[j] = (u[j] + u[j] * ada[n + j]) + ada[j];
        }
        matmul(m_model->final_linear(), u, 1, step);
        add_bias(step, m_model->final_b(), ldim);
        const float inv = 1.0f / static_cast<float>(n_steps);
        for (size_t j = 0; j < ldim; ++j)
        {
            x[j] += step[j] * inv;
        }
    }
    std::memcpy(out, x, ldim * sizeof(float));
}

void PocketRunner::upsample_task(void* user, int thread_index, int thread_count)
{
    const UpsampleJob& job = *static_cast<const UpsampleJob*>(user);
    const size_t rows = job.n * job.stride;
    const size_t first = rows * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = rows * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t r = first; r < last; ++r)
    {
        float* out = job.out + r * job.channels;
        std::memset(out, 0, job.channels * sizeof(float));
        for (size_t t = r / job.stride + 1; t-- > 0;)
        {
            const size_t k = r - t * job.stride;
            if (k >= job.k)
            {
                break;
            }
            const float* x = job.x + t * job.channels;
            for (size_t c = 0; c < job.channels; ++c)
            {
                out[c] += x[c] * job.w[c * job.k + k];
            }
        }
    }
}

void PocketRunner::conv_task(void* user, int thread_index, int thread_count)
{
    const ConvJob& job = *static_cast<const ConvJob*>(user);
    const PocketConv& c = *job.c;
    const size_t pad = (c.k - 1) * c.dilation + 1 - c.stride;
    const size_t threads = static_cast<size_t>(thread_count);
    const size_t index = static_cast<size_t>(thread_index);
    if (c.oc >= 4 * threads)
    {
        const size_t blocks = (c.oc + 3) / 4;
        const size_t first = blocks * index / threads * 4;
        size_t last = blocks * (index + 1) / threads * 4;
        last = last < c.oc ? last : c.oc;
        if (first < last)
        {
            conv1d_rows(job.x, job.length, job.length, c.ic, c.w, c.k, c.dilation, pad, c.b, first, last, 0, job.length, job.out,
                        job.length);
        }
    }
    else
    {
        const size_t t_first = job.length * index / threads;
        const size_t t_last = job.length * (index + 1) / threads;
        if (t_first < t_last)
        {
            conv1d_rows(job.x, job.length, job.length, c.ic, c.w, c.k, c.dilation, pad, c.b, 0, c.oc, t_first, t_last, job.out,
                        job.length);
        }
    }
}

// x: ic channels of length values; out: oc channels of length values (stride 1)
void PocketRunner::conv1d(const PocketConv& c, const float* x, size_t length, float* out)
{
    ConvJob job{&c, x, length, out};
    m_pool->run(conv_task, &job);
}

void PocketRunner::convtr_task(void* user, int thread_index, int thread_count)
{
    const ConvTrJob& job = *static_cast<const ConvTrJob*>(user);
    const PocketConv& c = *job.c;
    const size_t threads = static_cast<size_t>(thread_count);
    const size_t index = static_cast<size_t>(thread_index);
    const size_t blocks = (c.oc + 3) / 4;
    const size_t first = blocks * index / threads * 4;
    size_t last = blocks * (index + 1) / threads * 4;
    last = last < c.oc ? last : c.oc;
    const size_t out_length = job.length * c.stride;
    float* tmp = job.tmp + index * 4 * job.length;
    for (size_t o = first; o < last; o += 4)
    {
        const size_t rows = last - o < 4 ? last - o : 4;
        for (size_t p = 0; p < c.stride; ++p)
        {
            conv1d_rows(job.x, job.length, job.length, c.ic, c.w + p * c.oc * c.ic * 2, 2, 1, 1, nullptr, o, o + rows, 0,
                        job.length, tmp - o * job.length, job.length);
            for (size_t r = 0; r < rows; ++r)
            {
                const float bias = c.b != nullptr ? c.b[o + r] : 0.0f;
                const float* src = tmp + r * job.length;
                float* dst = job.out + (o + r) * out_length + p;
                for (size_t t = 0; t < job.length; ++t)
                {
                    dst[t * c.stride] = src[t] + bias;
                }
            }
        }
    }
}

// x: ic channels of length values; out: oc channels of length * stride values
void PocketRunner::convtr(const PocketConv& c, const float* x, size_t length, float* out)
{
    ConvTrJob job{&c, x, length, out, m_convtr_tmp.data()};
    m_pool->run(convtr_task, &job);
}

bool PocketRunner::decode(const float* latents, size_t n_frames, Array<float>& audio, Error& error)
{
    const PocketHparams& h = m_model->hparams();
    const size_t ldim = h.ldim;
    const size_t C = h.outer_dim;
    const size_t T = n_frames * h.upsample_stride;
    size_t widest = C * T;
    size_t longest_in = T;
    {
        size_t channels = m_model->decoder_in().oc;
        size_t length = T;
        for (size_t s = 0; s < h.n_ratios; ++s)
        {
            channels = m_model->decoder_stage(s).resample.oc;
            longest_in = length > longest_in ? length : longest_in;
            length *= h.ratios[s];
            widest = channels * length > widest ? channels * length : widest;
        }
    }
    const size_t tmp_floats = static_cast<size_t>(m_pool->size()) * 4 * longest_in;
    const bool ok = reserve_rows(T, C, h.tfm_ffn_dim, error) && (m_x.size() >= T * C || m_x.resize_uninitialized(T * C)) &&
                    (m_mimi_k.size() >= T * C || m_mimi_k.resize_uninitialized(T * C)) &&
                    (m_mimi_v.size() >= T * C || m_mimi_v.resize_uninitialized(T * C)) &&
                    (m_dec_a.size() >= widest || m_dec_a.resize_uninitialized(widest)) &&
                    (m_dec_b.size() >= widest || m_dec_b.resize_uninitialized(widest)) &&
                    (m_dec_c.size() >= widest || m_dec_c.resize_uninitialized(widest)) &&
                    (m_convtr_tmp.size() >= tmp_floats || m_convtr_tmp.resize_uninitialized(tmp_floats));
    if (!ok)
    {
        error.set("out of memory for the Mimi decoder");
        return false;
    }
    float* a = m_dec_a.data();
    float* b = m_dec_b.data();
    float* c = m_dec_c.data();
    for (size_t t = 0; t < n_frames; ++t)
    {
        for (size_t i = 0; i < ldim; ++i)
        {
            a[t * ldim + i] = latents[t * ldim + i] * m_model->emb_std()[i] + m_model->emb_mean()[i];
        }
    }
    matmul(m_model->quant_proj(), a, n_frames, b);
    UpsampleJob up{b, n_frames, C, m_model->upsample_w(), m_model->upsample_k(), h.upsample_stride, m_x.data()};
    m_pool->run(upsample_task, &up);
    for (size_t l = 0; l < h.tfm_n_layers; ++l)
    {
        transformer_layer(m_model->decoder_layer(l), m_x.data(), T, C, h.tfm_n_heads, h.tfm_max_period, 0, m_mimi_k.data(),
                          m_mimi_v.data(), C, h.tfm_context);
    }
    for (size_t t = 0; t < T; ++t)
    {
        for (size_t ch = 0; ch < C; ++ch)
        {
            a[ch * T + t] = m_x[t * C + ch];
        }
    }
    size_t length = T;
    conv1d(m_model->decoder_in(), a, length, b);
    float* x = b;
    float* t1 = a;
    float* t2 = c;
    for (size_t s = 0; s < h.n_ratios; ++s)
    {
        const PocketStage& st = m_model->decoder_stage(s);
        elu_rows(x, st.resample.ic * length);
        convtr(st.resample, x, length, t1);
        length *= st.resample.stride;
        float* swap = x;
        x = t1;
        t1 = swap;
        for (size_t r = 0; r < st.n_res; ++r)
        {
            const PocketResUnit& ru = st.res[r];
            std::memcpy(t1, x, ru.c1.ic * length * sizeof(float));
            elu_rows(t1, ru.c1.ic * length);
            conv1d(ru.c1, t1, length, t2);
            elu_rows(t2, ru.c1.oc * length);
            conv1d(ru.c2, t2, length, t1);
            add(x, t1, ru.c2.oc * length);
        }
    }
    const PocketConv& out = m_model->decoder_out();
    elu_rows(x, out.ic * length);
    const size_t n_samples = out.oc * length;
    if (!audio.resize_uninitialized(n_samples))
    {
        error.set("out of memory for the audio");
        return false;
    }
    conv1d(out, x, length, audio.data());
    return true;
}

bool PocketRunner::synthesize(const PocketVoice& voice, const char* text, size_t length, Array<float>& samples, Error& error)
{
    const PocketHparams& h = m_model->hparams();
    const size_t dim = h.dim;
    const size_t ldim = h.ldim;
    m_stats = PocketStats();
    if (voice.n == 0)
    {
        error.set("the voice has no rows");
        return false;
    }
    Array<char> chunks;
    Array<uint32_t> offsets;
    if (!pocket_chunk_text(m_model->text_rules(), m_model->tokenizer(), text, length, m_max_chunk_tokens, chunks, offsets, error))
    {
        return false;
    }
    const size_t n_chunks = offsets.size() - 1;
    if (n_chunks == 0)
    {
        error.set("empty text");
        return false;
    }
    TorchRandom rng(m_seed);
    const float stdv = sqrtf(h.temperature);
    Array<char> prepared;
    Array<int32_t> ids;
    Array<float> noise;
    Array<float> latent;
    Array<float> next;
    Array<float> latents;
    Array<float> eos_values;
    Array<float> audio;
    if (!noise.resize_uninitialized(ldim) || !latent.resize_uninitialized(ldim) || !next.resize_uninitialized(ldim))
    {
        error.set("out of memory");
        return false;
    }
    for (size_t ci = 0; ci < n_chunks; ++ci)
    {
        int guess = 1;
        prepared.clear();
        ids.clear();
        if (!pocket_prepare_text(m_model->text_rules(), chunks.data() + offsets[ci], offsets[ci + 1] - offsets[ci], prepared, guess,
                                 error) ||
            !m_model->tokenizer().encode(prepared.data(), prepared.size(), ids))
        {
            error.set("the text of chunk %zu could not be prepared", ci);
            return false;
        }
        const size_t ntok = ids.size();
        const size_t fae = h.frames_after_eos >= 0 ? static_cast<size_t>(h.frames_after_eos) : static_cast<size_t>(guess) + 2;
        const size_t max_gen = static_cast<size_t>(ceil((static_cast<double>(ntok) / 3.0 + 2.0) * static_cast<double>(h.frame_rate)));
        const size_t rows = voice.n + ntok + max_gen + 1;
        if (rows > m_cache_rows)
        {
            if (!m_cache_k.resize_uninitialized(rows * dim * h.n_layers) || !m_cache_v.resize_uninitialized(rows * dim * h.n_layers))
            {
                error.set("out of memory for the KV cache");
                return false;
            }
            m_cache_rows = rows;
        }
        const size_t voice_floats = static_cast<size_t>(voice.n) * dim;
        for (size_t l = 0; l < h.n_layers; ++l)
        {
            std::memcpy(m_cache_k.data() + l * m_cache_rows * dim, voice.k.data() + l * voice_floats, voice_floats * sizeof(float));
            std::memcpy(m_cache_v.data() + l * m_cache_rows * dim, voice.v.data() + l * voice_floats, voice_floats * sizeof(float));
        }
        if (m_trace != nullptr)
        {
            Array<float> as_float;
            if (as_float.resize_uninitialized(ntok))
            {
                for (size_t i = 0; i < ntok; ++i)
                {
                    as_float[i] = static_cast<float>(ids[i]);
                }
                trace("prompt-ids", static_cast<int>(ci), as_float.data(), ntok);
            }
        }
        if (!flow_prefill(ids.data(), ntok, voice.n, error))
        {
            return false;
        }
        rng.normal(noise.data(), ldim, stdv);

        std::memcpy(latent.data(), m_model->bos_emb(), ldim * sizeof(float));
        latents.clear();
        size_t position = voice.n + ntok;
        ptrdiff_t eos_step = -1;
        size_t step = 0;
        for (; step < max_gen; ++step)
        {
            rng.normal(noise.data(), ldim, stdv);
            float eos = 0.0f;
            if (!flow_step(latent.data(), noise.data(), position, eos, next.data(), error))
            {
                return false;
            }
            if (ci == 0 && step == 0)
            {
                trace("step0-latent", -1, next.data(), ldim);
            }
            if (!eos_values.push_back(eos))
            {
                error.set("out of memory");
                return false;
            }
            ++position;
            ++m_stats.steps;
            if (eos > EOS_THRESHOLD && eos_step < 0 && step >= EOS_MIN_FRAMES)
            {
                eos_step = static_cast<ptrdiff_t>(step);
            }
            if (eos_step >= 0 && step >= static_cast<size_t>(eos_step) + fae)
            {
                break;
            }
            if (!latents.append(next.data(), ldim))
            {
                error.set("out of memory");
                return false;
            }
            std::memcpy(latent.data(), next.data(), ldim * sizeof(float));
        }
        const size_t frames = latents.size() / ldim;
        if (ci == 0)
        {
            trace("latents", -1, latents.data(), latents.size());
        }
        m_stats.frames += frames;
        m_stats.chunks++;
        if (frames == 0)
        {
            continue;
        }
        if (!decode(latents.data(), frames, audio, error))
        {
            return false;
        }
        const size_t skip = audio.size() < h.frame_size ? audio.size() : h.frame_size;
        if (!samples.append(audio.data() + skip, audio.size() - skip))
        {
            error.set("out of memory for the samples");
            return false;
        }
    }
    trace("eos", -1, eos_values.data(), eos_values.size());
    return true;
}

}
