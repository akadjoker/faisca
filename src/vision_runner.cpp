#include "faisca/vision_runner.h"

#include "faisca/ops.h"

#include <cmath>
#include <cstring>

namespace faisca
{

bool preprocess_image(const Image& image, const VisionHparams& h, Array<float>& planes, Error& error)
{
    const uint32_t size = h.image_size;
    if (image.width == 0 || image.height == 0)
    {
        error.set("the image is empty");
        return false;
    }
    const float scale_w = static_cast<float>(size) / static_cast<float>(image.width);
    const float scale_h = static_cast<float>(size) / static_cast<float>(image.height);
    const float scale = scale_w < scale_h ? scale_w : scale_h;
    uint32_t new_w = static_cast<uint32_t>(std::ceil(static_cast<float>(image.width) * scale));
    uint32_t new_h = static_cast<uint32_t>(std::ceil(static_cast<float>(image.height) * scale));
    new_w = new_w < size ? new_w : size;
    new_h = new_h < size ? new_h : size;
    Image resized;
    if (!resize_bilinear(image, new_w, new_h, resized, error))
    {
        return false;
    }
    const uint32_t offset_x = (size - new_w) / 2;
    const uint32_t offset_y = (size - new_h) / 2;
    const size_t n = static_cast<size_t>(size) * size;
    if (!planes.resize_uninitialized(3 * n))
    {
        error.set("out of memory for the image planes");
        return false;
    }
    for (uint32_t c = 0; c < 3; ++c)
    {
        const float black = (0.0f - h.image_mean[c]) / h.image_std[c];
        float* plane = planes.data() + c * n;
        for (size_t i = 0; i < n; ++i)
        {
            plane[i] = black;
        }
        for (uint32_t y = 0; y < new_h; ++y)
        {
            const uint8_t* row = resized.rgb.data() + static_cast<size_t>(y) * new_w * 3;
            float* dst = plane + static_cast<size_t>(offset_y + y) * size + offset_x;
            for (uint32_t x = 0; x < new_w; ++x)
            {
                const float value = static_cast<float>(row[x * 3 + c]) / 255.0f;
                dst[x] = (value - h.image_mean[c]) / h.image_std[c];
            }
        }
    }
    return true;
}

struct VisionRunner::RowsJob
{
    const float* x;
    const float* w;
    const float* b;
    size_t rows;
    size_t width;
    float eps;
    float* out;
    int mode; // 0 layer norm, 1 gelu, 2 bias, 3 add
};

struct VisionRunner::MatmulJob
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

struct VisionRunner::AttentionJob
{
    VisionRunner* runner;
    const float* q;
    const float* k_tiles;
    const float* v;
    size_t n_q;
    size_t n_kv;
    size_t n_head;
    size_t head_dim;
    size_t stride;
    float scale;
    float* out;
};

struct VisionRunner::TileJob
{
    const float* k;
    size_t n_kv;
    size_t n_head;
    size_t head_dim;
    size_t stride;
    size_t per_head;
    float* tiles;
};

struct VisionRunner::PatchJob
{
    const float* planes;
    size_t size;
    size_t patch;
    size_t per_side;
    float* out;
};

bool VisionRunner::init(const VisionModel& model, ThreadPool& pool, Error& error)
{
    m_model = &model;
    m_pool = &pool;
    const VisionHparams& h = model.hparams();
    const size_t n = h.n_patches();
    const size_t dim = h.n_embd;
    const size_t threads = static_cast<size_t>(pool.size());
    const size_t widest = h.n_ff_padded > dim ? h.n_ff_padded : dim;
    const size_t taps = 3 * static_cast<size_t>(h.patch_size) * h.patch_size;
    const size_t taps_padded = (taps + Q8_0_BLOCK - 1) / Q8_0_BLOCK * Q8_0_BLOCK;
    m_xq_stride = widest / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
    m_sums_stride = widest / Q8_0_BLOCK;
    const bool ok = m_patches.resize_uninitialized(n * taps_padded) && m_x.resize_uninitialized(n * dim) &&
                    m_h.resize_uninitialized(n * dim) && m_q.resize_uninitialized(n * dim) && m_k.resize_uninitialized(n * dim) &&
                    m_v.resize_uninitialized(n * dim) && m_attn.resize_uninitialized(n * dim) &&
                    m_up.resize_uninitialized(n * h.n_ff_padded) && m_k_tiles.resize_uninitialized(h.n_head * tiles_per_head()) &&
                    m_scores.resize_uninitialized(threads * (n + KEY_TILE)) && m_pooled.resize_uninitialized(h.n_tokens() * dim) &&
                    m_xq.resize_uninitialized(n * m_xq_stride) && m_sums.resize_uninitialized(n * m_sums_stride);
    if (!ok)
    {
        error.set("out of memory for the vision encoder buffers");
        return false;
    }
    std::memset(m_patches.data(), 0, n * taps_padded * sizeof(float));
    std::memset(m_up.data(), 0, n * h.n_ff_padded * sizeof(float));
    return true;
}

void VisionRunner::trace(const char* name, int layer, const float* data, size_t count) const
{
    if (m_trace != nullptr)
    {
        m_trace(m_trace_user, name, layer, data, count);
    }
}

size_t VisionRunner::tiles_per_head() const
{
    const VisionHparams& h = m_model->hparams();
    const size_t head_dim = h.n_embd / h.n_head;
    return (h.n_patches() + KEY_TILE - 1) / KEY_TILE * KEY_TILE * head_dim;
}

void VisionRunner::rows_task(void* user, int thread_index, int thread_count)
{
    const RowsJob& job = *static_cast<const RowsJob*>(user);
    const size_t first = job.rows * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = job.rows * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t r = first; r < last; ++r)
    {
        float* out = job.out + r * job.width;
        switch (job.mode)
        {
        case 0:
            layer_norm(job.x + r * job.width, job.w, job.b, job.width, job.eps, out);
            break;
        case 1:
            gelu(out, job.width);
            break;
        case 2:
            add_bias(out, job.b, job.width);
            break;
        default:
            add(out, job.x + r * job.width, job.width);
            break;
        }
    }
}

void VisionRunner::norm_rows(const float* x, const float* w, const float* b, size_t rows, size_t width, float* out)
{
    RowsJob job{x, w, b, rows, width, m_model->hparams().eps, out, 0};
    m_pool->run(rows_task, &job);
}

void VisionRunner::gelu_rows(float* x, size_t rows, size_t width)
{
    RowsJob job{nullptr, nullptr, nullptr, rows, width, 0.0f, x, 1};
    m_pool->run(rows_task, &job);
}

void VisionRunner::bias_rows(float* x, const float* b, size_t rows, size_t width)
{
    RowsJob job{nullptr, nullptr, b, rows, width, 0.0f, x, 2};
    m_pool->run(rows_task, &job);
}

void VisionRunner::add_rows(float* x, const float* y, size_t rows, size_t width)
{
    RowsJob job{y, nullptr, nullptr, rows, width, 0.0f, x, 3};
    m_pool->run(rows_task, &job);
}

void VisionRunner::matmul_task(void* user, int thread_index, int thread_count)
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

void VisionRunner::quantize_task(void* user, int thread_index, int thread_count)
{
    VisionRunner& runner = *static_cast<VisionRunner*>(user);
    const size_t first = runner.m_quantize_tokens * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = runner.m_quantize_tokens * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t t = first; t < last; ++t)
    {
        uint8_t* xq = runner.m_xq.data() + t * runner.m_xq_stride;
        quantize_q8_0(runner.m_quantize_x + t * runner.m_quantize_n, runner.m_quantize_n, xq);
        q8_0_block_sums(xq, runner.m_quantize_n, runner.m_sums.data() + t * runner.m_sums_stride);
    }
}

void VisionRunner::quantize_inputs(const float* x, size_t n_tokens, size_t n)
{
    m_quantize_x = x;
    m_quantize_tokens = n_tokens;
    m_quantize_n = n;
    m_pool->run(quantize_task, this);
}

// x rows are w.n_in apart, out rows w.n_out apart
void VisionRunner::matmul(const Weight& w, const float* x, size_t n_tokens, float* out)
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

void VisionRunner::tile_task(void* user, int thread_index, int thread_count)
{
    const TileJob& job = *static_cast<const TileJob*>(user);
    const size_t first = job.n_head * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = job.n_head * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    for (size_t head = first; head < last; ++head)
    {
        float* tiles = job.tiles + head * job.per_head;
        for (size_t p = 0; p < job.n_kv; ++p)
        {
            const float* row = job.k + p * job.stride + head * job.head_dim;
            for (size_t d = 0; d < job.head_dim; ++d)
            {
                tiles[key_index(job.head_dim, p, d)] = row[d];
            }
        }
        for (size_t p = job.n_kv; p < (job.n_kv + KEY_TILE - 1) / KEY_TILE * KEY_TILE; ++p)
        {
            for (size_t d = 0; d < job.head_dim; ++d)
            {
                tiles[key_index(job.head_dim, p, d)] = 0.0f;
            }
        }
    }
}

void VisionRunner::tile_keys(const float* k, size_t n_kv, float* tiles)
{
    const VisionHparams& h = m_model->hparams();
    TileJob job{k, n_kv, h.n_head, h.n_embd / h.n_head, h.n_embd, tiles_per_head(), tiles};
    m_pool->run(tile_task, &job);
}

void VisionRunner::attention_task(void* user, int thread_index, int thread_count)
{
    const AttentionJob& job = *static_cast<const AttentionJob*>(user);
    const size_t units = job.n_head * job.n_q;
    const size_t first = units * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = units * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    float* scores = job.runner->m_scores.data() +
                    static_cast<size_t>(thread_index) * (job.runner->m_scores.size() / static_cast<size_t>(thread_count));
    const size_t per_head = job.runner->tiles_per_head();
    for (size_t u = first; u < last; ++u)
    {
        const size_t head = u / job.n_q;
        const size_t i = u % job.n_q;
        attention_head(job.q + i * job.stride + head * job.head_dim, job.k_tiles + head * per_head, job.v + head * job.head_dim,
                       job.stride, job.head_dim, job.n_kv, job.scale, scores, job.out + i * job.stride + head * job.head_dim);
    }
}

void VisionRunner::patch_task(void* user, int thread_index, int thread_count)
{
    const PatchJob& job = *static_cast<const PatchJob*>(user);
    const size_t n = job.per_side * job.per_side;
    const size_t first = n * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
    const size_t last = n * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
    const size_t taps = 3 * job.patch * job.patch;
    const size_t taps_padded = (taps + Q8_0_BLOCK - 1) / Q8_0_BLOCK * Q8_0_BLOCK;
    for (size_t p = first; p < last; ++p)
    {
        const size_t py = p / job.per_side;
        const size_t px = p % job.per_side;
        float* out = job.out + p * taps_padded;
        for (size_t c = 0; c < 3; ++c)
        {
            const float* plane = job.planes + c * job.size * job.size;
            for (size_t ky = 0; ky < job.patch; ++ky)
            {
                const float* row = plane + (py * job.patch + ky) * job.size + px * job.patch;
                std::memcpy(out + (c * job.patch + ky) * job.patch, row, job.patch * sizeof(float));
            }
        }
    }
}

bool VisionRunner::encode(const Image& image, Array<float>& embeddings, Error& error)
{
    if (!preprocess_image(image, m_model->hparams(), m_planes, error))
    {
        return false;
    }
    trace("inp_raw", -1, m_planes.data(), m_planes.size());
    return encode_planes(m_planes.data(), embeddings, error);
}

bool VisionRunner::encode_planes(const float* planes, Array<float>& embeddings, Error& error)
{
    const VisionHparams& h = m_model->hparams();
    const size_t n = h.n_patches();
    const size_t dim = h.n_embd;
    const size_t head_dim = dim / h.n_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    PatchJob patches{planes, h.image_size, h.patch_size, h.patches_per_side(), m_patches.data()};
    m_pool->run(patch_task, &patches);
    matmul(m_model->patch_embd(), m_patches.data(), n, m_x.data());
    bias_rows(m_x.data(), m_model->patch_bias(), n, dim);
    trace("patch_bias", -1, m_x.data(), n * dim);
    add_rows(m_x.data(), m_model->position_embd(), n, dim);
    trace("pos_embed", -1, m_x.data(), n * dim);

    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        const VisionLayer& layer = m_model->layer(l);
        const int il = static_cast<int>(l);
        norm_rows(m_x.data(), layer.ln1_w, layer.ln1_b, n, dim, m_h.data());
        trace("layer_inp_normed", il, m_h.data(), n * dim);
        matmul(layer.q, m_h.data(), n, m_q.data());
        bias_rows(m_q.data(), layer.q_b, n, dim);
        matmul(layer.k, m_h.data(), n, m_k.data());
        bias_rows(m_k.data(), layer.k_b, n, dim);
        matmul(layer.v, m_h.data(), n, m_v.data());
        bias_rows(m_v.data(), layer.v_b, n, dim);
        trace("Qcur", il, m_q.data(), n * dim);
        trace("Kcur", il, m_k.data(), n * dim);
        trace("Vcur", il, m_v.data(), n * dim);
        tile_keys(m_k.data(), n, m_k_tiles.data());
        AttentionJob job{this, m_q.data(), m_k_tiles.data(), m_v.data(), n, n, h.n_head, head_dim, dim, scale, m_attn.data()};
        m_pool->run(attention_task, &job);
        trace("kqv_out", il, m_attn.data(), n * dim);
        matmul(layer.out, m_attn.data(), n, m_h.data());
        bias_rows(m_h.data(), layer.out_b, n, dim);
        trace("attn_out", il, m_h.data(), n * dim);
        add_rows(m_x.data(), m_h.data(), n, dim);
        trace("ffn_inp", il, m_x.data(), n * dim);
        norm_rows(m_x.data(), layer.ln2_w, layer.ln2_b, n, dim, m_h.data());
        trace("ffn_inp_normed", il, m_h.data(), n * dim);
        matmul(layer.up, m_h.data(), n, m_up.data());
        // the padded tail of every row stays zero: the bias and the GELU cover n_ff values
        for (size_t t = 0; t < n; ++t)
        {
            add_bias(m_up.data() + t * h.n_ff_padded, layer.up_b, h.n_ff);
            gelu(m_up.data() + t * h.n_ff_padded, h.n_ff);
        }
        trace("ffn_gelu", il, m_up.data(), n * h.n_ff_padded);
        matmul(layer.down, m_up.data(), n, m_h.data());
        bias_rows(m_h.data(), layer.down_b, n, dim);
        trace("ffn_out", il, m_h.data(), n * dim);
        add_rows(m_x.data(), m_h.data(), n, dim);
        trace("layer_out", il, m_x.data(), n * dim);
    }
    norm_rows(m_x.data(), m_model->post_ln_w(), m_model->post_ln_b(), n, dim, m_h.data());
    trace("post_ln", -1, m_h.data(), n * dim);

    // average pool of n_merge x n_merge patches, then RMS norm and the projection
    const size_t side = h.patches_per_side();
    const size_t merge = h.n_merge;
    const size_t out_side = side / merge;
    const float inverse = 1.0f / static_cast<float>(merge * merge);
    for (size_t oy = 0; oy < out_side; ++oy)
    {
        for (size_t ox = 0; ox < out_side; ++ox)
        {
            float* pooled = m_pooled.data() + (oy * out_side + ox) * dim;
            for (size_t d = 0; d < dim; ++d)
            {
                pooled[d] = 0.0f;
            }
            for (size_t ky = 0; ky < merge; ++ky)
            {
                for (size_t kx = 0; kx < merge; ++kx)
                {
                    const float* src = m_h.data() + ((oy * merge + ky) * side + ox * merge + kx) * dim;
                    for (size_t d = 0; d < dim; ++d)
                    {
                        pooled[d] += src[d];
                    }
                }
            }
            for (size_t d = 0; d < dim; ++d)
            {
                pooled[d] *= inverse;
            }
            rms_norm(pooled, m_model->soft_emb_norm(), dim, h.eps, pooled);
        }
    }
    const size_t n_tokens = h.n_tokens();
    trace("pooled_normed", -1, m_pooled.data(), n_tokens * dim);
    if (!embeddings.resize_uninitialized(n_tokens * h.n_proj))
    {
        error.set("out of memory for the image embeddings");
        return false;
    }
    matmul(m_model->projection(), m_pooled.data(), n_tokens, embeddings.data());
    trace("image_embd", -1, embeddings.data(), embeddings.size());
    return true;
}

}
