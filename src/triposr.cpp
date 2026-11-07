#include "faisca/triposr.h"

#include "faisca/ops.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

template <typename F>
void parallel_for(ThreadPool& pool, size_t count, size_t group, const F& body)
{
    struct Job
    {
        const F* body;
        size_t count;
        size_t group;
    };
    Job job = {&body, count, group};
    pool.run(
        [](void* user, int thread_index, int thread_count)
        {
            const Job& j = *static_cast<const Job*>(user);
            const size_t units = (j.count + j.group - 1) / j.group;
            const size_t first = units * static_cast<size_t>(thread_index) / static_cast<size_t>(thread_count);
            const size_t last = units * static_cast<size_t>(thread_index + 1) / static_cast<size_t>(thread_count);
            const size_t begin = first * j.group;
            const size_t end = last * j.group < j.count ? last * j.group : j.count;
            if (begin < end)
            {
                (*j.body)(begin, end, thread_index);
            }
        },
        &job);
}

// units handed out one chunk at a time from a shared counter, so the faster cores of a
// hybrid CPU take more of them; body(first, last, thread) must not depend on the thread
// for its results
template <typename F>
void parallel_dynamic(ThreadPool& pool, size_t count, size_t chunk, const F& body)
{
    struct Job
    {
        const F* body;
        size_t count;
        size_t chunk;
        std::atomic<size_t> next;
    };
    Job job;
    job.body = &body;
    job.count = count;
    job.chunk = chunk;
    job.next.store(0, std::memory_order_relaxed);
    pool.run(
        [](void* user, int thread_index, int)
        {
            Job& j = *static_cast<Job*>(user);
            for (;;)
            {
                const size_t first = j.next.fetch_add(j.chunk, std::memory_order_relaxed);
                if (first >= j.count)
                {
                    return;
                }
                const size_t last = first + j.chunk < j.count ? first + j.chunk : j.count;
                (*j.body)(first, last, thread_index);
            }
        },
        &job);
}

float gelu_erf(float x)
{
    return x * 0.5f * (1.0f + std::erf(x * 0.70710678118654752440f));
}

}

const float* TripoModel::f32(const char* name, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3, Error& error)
{
    const CkptTensor* t = m_file.find(name);
    if (t == nullptr)
    {
        error.set("TripoSR checkpoint has no %s", name);
        return nullptr;
    }
    const uint64_t want[4] = {d0, d1, d2, d3};
    uint32_t n_dims = 0;
    while (n_dims < 4 && want[n_dims] != 0)
    {
        n_dims += 1;
    }
    bool same = t->dtype == CkptDtype::F32 && t->n_dims == n_dims;
    for (uint32_t d = 0; same && d < n_dims; ++d)
    {
        same = t->shape[d] == want[d];
    }
    if (!same)
    {
        error.set("TripoSR %s: unexpected %s tensor of %u dims", name, ckpt_dtype_name(t->dtype), t->n_dims);
        return nullptr;
    }
    return reinterpret_cast<const float*>(t->data);
}

void TripoModel::add_linear(const float* rows, size_t n_out, size_t n_in, TripoLinear& w)
{
    w.panels = nullptr;
    w.n_in = n_in;
    w.n_out = n_out;
    if (m_n_pending < MAX_LINEARS)
    {
        m_pending[m_n_pending] = Pending{&w, rows};
    }
    m_n_pending += 1;
}

bool TripoModel::pack_linears(Error& error)
{
    if (m_n_pending > MAX_LINEARS)
    {
        error.set("TripoSR: more than %zu linear layers", MAX_LINEARS);
        return false;
    }
    size_t total = 0;
    for (size_t i = 0; i < m_n_pending; ++i)
    {
        const TripoLinear& w = *m_pending[i].linear;
        if (w.n_out % F32_PANEL != 0)
        {
            error.set("TripoSR: a linear layer of %zu outputs is not a multiple of %zu", w.n_out, F32_PANEL);
            return false;
        }
        total += w.n_in * w.n_out;
    }
    if (!m_panels.resize_uninitialized(total))
    {
        error.set("out of memory packing %zu TripoSR weights", total);
        return false;
    }
    size_t offset = 0;
    for (size_t i = 0; i < m_n_pending; ++i)
    {
        TripoLinear& w = *m_pending[i].linear;
        pack_f32_panels(m_pending[i].rows, w.n_in, w.n_out, m_panels.data() + offset);
        w.panels = m_panels.data() + offset;
        offset += w.n_in * w.n_out;
    }
    m_n_pending = 0;
    return true;
}

bool TripoModel::linear(const char* name, uint64_t n_out, uint64_t n_in, TripoLinear& w, const float** bias, Error& error)
{
    char key[192];
    std::snprintf(key, sizeof(key), "%s.weight", name);
    const float* data = f32(key, n_out, n_in, 0, 0, error);
    if (data == nullptr)
    {
        return false;
    }
    add_linear(data, n_out, n_in, w);
    if (bias != nullptr)
    {
        std::snprintf(key, sizeof(key), "%s.bias", name);
        *bias = f32(key, n_out, 0, 0, 0, error);
        return *bias != nullptr;
    }
    return true;
}

bool TripoModel::open(const char* path, Error& error)
{
    if (!m_file.open(path, error))
    {
        return false;
    }
    const char* vit = "image_tokenizer.model";
    char key[160];
    std::snprintf(key, sizeof(key), "%s.embeddings.cls_token", vit);
    m_vit_cls = f32(key, 1, 1, VIT_DIM, 0, error);
    std::snprintf(key, sizeof(key), "%s.embeddings.position_embeddings", vit);
    m_vit_pos = f32(key, 1, VIT_POS_SIDE * VIT_POS_SIDE + 1, VIT_DIM, 0, error);
    std::snprintf(key, sizeof(key), "%s.embeddings.patch_embeddings.projection.weight", vit);
    const float* patch = f32(key, VIT_DIM, 3, VIT_PATCH, VIT_PATCH, error);
    std::snprintf(key, sizeof(key), "%s.embeddings.patch_embeddings.projection.bias", vit);
    m_vit_patch_b = f32(key, VIT_DIM, 0, 0, 0, error);
    std::snprintf(key, sizeof(key), "%s.layernorm.weight", vit);
    m_vit_norm_w = f32(key, VIT_DIM, 0, 0, 0, error);
    std::snprintf(key, sizeof(key), "%s.layernorm.bias", vit);
    m_vit_norm_b = f32(key, VIT_DIM, 0, 0, 0, error);
    if (!m_vit_cls || !m_vit_pos || !patch || !m_vit_patch_b || !m_vit_norm_w || !m_vit_norm_b)
    {
        return false;
    }
    add_linear(patch, VIT_DIM, 3 * VIT_PATCH * VIT_PATCH, m_vit_patch);

    for (uint32_t il = 0; il < VIT_LAYERS; ++il)
    {
        TripoVitLayer& l = m_vit_layers[il];
        char base[96];
        std::snprintf(base, sizeof(base), "%s.encoder.layer.%u", vit, il);
        auto vec = [&](const char* suffix) -> const float*
        {
            std::snprintf(key, sizeof(key), "%s.%s", base, suffix);
            return f32(key, VIT_DIM, 0, 0, 0, error);
        };
        auto lin = [&](const char* suffix, uint64_t n_out, uint64_t n_in, TripoLinear& w, const float** b)
        {
            std::snprintf(key, sizeof(key), "%s.%s", base, suffix);
            return linear(key, n_out, n_in, w, b, error);
        };
        l.ln1_w = vec("layernorm_before.weight");
        l.ln1_b = vec("layernorm_before.bias");
        l.ln2_w = vec("layernorm_after.weight");
        l.ln2_b = vec("layernorm_after.bias");
        if (!l.ln1_w || !l.ln1_b || !l.ln2_w || !l.ln2_b ||
            !lin("attention.attention.query", VIT_DIM, VIT_DIM, l.q, &l.q_b) ||
            !lin("attention.attention.key", VIT_DIM, VIT_DIM, l.k, &l.k_b) ||
            !lin("attention.attention.value", VIT_DIM, VIT_DIM, l.v, &l.v_b) ||
            !lin("attention.output.dense", VIT_DIM, VIT_DIM, l.out, &l.out_b) ||
            !lin("intermediate.dense", VIT_FF, VIT_DIM, l.up, &l.up_b) ||
            !lin("output.dense", VIT_DIM, VIT_FF, l.down, &l.down_b))
        {
            return false;
        }
    }

    m_planes = f32("tokenizer.embeddings", 3, DIM, PLANE_SIZE, PLANE_SIZE, error);
    m_group_w = f32("backbone.norm.weight", DIM, 0, 0, 0, error);
    m_group_b = f32("backbone.norm.bias", DIM, 0, 0, 0, error);
    if (!m_planes || !m_group_w || !m_group_b || !linear("backbone.proj_in", DIM, DIM, m_proj_in, &m_proj_in_b, error) ||
        !linear("backbone.proj_out", DIM, DIM, m_proj_out, &m_proj_out_b, error))
    {
        return false;
    }
    for (uint32_t ib = 0; ib < BLOCKS; ++ib)
    {
        TripoBlock& b = m_blocks[ib];
        char base[96];
        std::snprintf(base, sizeof(base), "backbone.transformer_blocks.%u", ib);
        auto vec = [&](const char* suffix) -> const float*
        {
            std::snprintf(key, sizeof(key), "%s.%s", base, suffix);
            return f32(key, DIM, 0, 0, 0, error);
        };
        auto lin = [&](const char* suffix, uint64_t n_out, uint64_t n_in, TripoLinear& w, const float** bias)
        {
            std::snprintf(key, sizeof(key), "%s.%s", base, suffix);
            return linear(key, n_out, n_in, w, bias, error);
        };
        b.ln1_w = vec("norm1.weight");
        b.ln1_b = vec("norm1.bias");
        b.ln2_w = vec("norm2.weight");
        b.ln2_b = vec("norm2.bias");
        b.ln3_w = vec("norm3.weight");
        b.ln3_b = vec("norm3.bias");
        if (!b.ln1_w || !b.ln1_b || !b.ln2_w || !b.ln2_b || !b.ln3_w || !b.ln3_b ||
            !lin("attn1.to_q", DIM, DIM, b.self_q, nullptr) || !lin("attn1.to_k", DIM, DIM, b.self_k, nullptr) ||
            !lin("attn1.to_v", DIM, DIM, b.self_v, nullptr) ||
            !lin("attn1.to_out.0", DIM, DIM, b.self_out, &b.self_out_b) ||
            !lin("attn2.to_q", DIM, DIM, b.cross_q, nullptr) ||
            !lin("attn2.to_k", DIM, VIT_DIM, b.cross_k, nullptr) ||
            !lin("attn2.to_v", DIM, VIT_DIM, b.cross_v, nullptr) ||
            !lin("attn2.to_out.0", DIM, DIM, b.cross_out, &b.cross_out_b) ||
            !lin("ff.net.0.proj", 2 * FF, DIM, b.ff_in, &b.ff_in_b) ||
            !lin("ff.net.2", DIM, FF, b.ff_out, &b.ff_out_b))
        {
            return false;
        }
    }

    const float* up = f32("post_processor.upsample.weight", DIM, CODE_CHANNELS, 2, 2, error);
    m_upsample_b = f32("post_processor.upsample.bias", CODE_CHANNELS, 0, 0, 0, error);
    if (!up || !m_upsample_b)
    {
        return false;
    }
    const size_t rows = CODE_CHANNELS * 4;
    if (!m_upsample_rows.resize_uninitialized(rows * DIM))
    {
        error.set("out of memory for the TripoSR upsampler");
        return false;
    }
    for (size_t ci = 0; ci < DIM; ++ci)
    {
        for (size_t r = 0; r < rows; ++r)
        {
            m_upsample_rows[r * DIM + ci] = up[ci * rows + r];
        }
    }
    add_linear(m_upsample_rows.data(), rows, DIM, m_upsample);

    for (uint32_t i = 0; i + 1 < MLP_LAYERS; ++i)
    {
        std::snprintf(key, sizeof(key), "decoder.layers.%u", 2 * i);
        if (!linear(key, MLP_WIDTH, i == 0 ? MLP_IN : MLP_WIDTH, m_mlp[i], &m_mlp_b[i], error))
        {
            return false;
        }
    }
    std::snprintf(key, sizeof(key), "decoder.layers.%u.weight", 2 * (MLP_LAYERS - 1));
    const float* last_w = f32(key, 4, MLP_WIDTH, 0, 0, error);
    std::snprintf(key, sizeof(key), "decoder.layers.%u.bias", 2 * (MLP_LAYERS - 1));
    const float* last_b = f32(key, 4, 0, 0, 0, error);
    if (!last_w || !last_b)
    {
        return false;
    }
    if (!m_mlp_last.resize_uninitialized(F32_PANEL * MLP_WIDTH + F32_PANEL))
    {
        error.set("out of memory for the TripoSR decoder");
        return false;
    }
    std::memset(m_mlp_last.data(), 0, m_mlp_last.size() * sizeof(float));
    std::memcpy(m_mlp_last.data(), last_w, 4 * MLP_WIDTH * sizeof(float));
    std::memcpy(m_mlp_last.data() + F32_PANEL * MLP_WIDTH, last_b, 4 * sizeof(float));
    add_linear(m_mlp_last.data(), F32_PANEL, MLP_WIDTH, m_mlp[MLP_LAYERS - 1]);
    m_mlp_b[MLP_LAYERS - 1] = m_mlp_last.data() + F32_PANEL * MLP_WIDTH;

    if (!pack_linears(error))
    {
        return false;
    }
    m_upsample_rows.release();
    return true;
}

void tripo_bicubic_rows(const float* pos, uint32_t side, uint32_t dim, double scale, uint32_t out_side, float* out)
{
    // UpSample.h compute_scales_value, area_pixel_compute_source_index (cubic),
    // guard_index_and_lambda, get_cubic_upsample_coefficients (A = -0.75)
    const float inverse = static_cast<float>(1.0 / scale);
    const int64_t last = static_cast<int64_t>(side) - 1;
    Array<int64_t> index;
    Array<float> weight;
    if (!index.resize_uninitialized(4 * size_t(out_side)) || !weight.resize_uninitialized(4 * size_t(out_side)))
    {
        return;
    }
    for (uint32_t i = 0; i < out_side; ++i)
    {
        const float real = inverse * (static_cast<float>(i) + 0.5f) - 0.5f;
        int64_t input = static_cast<int64_t>(std::floor(real));
        input = input < last ? input : last;
        float t = real - static_cast<float>(input);
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        const float A = -0.75f;
        const float x1 = t;
        const float x1p = static_cast<float>(static_cast<double>(x1) + 1.0);
        const float x2 = static_cast<float>(1.0 - static_cast<double>(t));
        const float x2p = static_cast<float>(static_cast<double>(x2) + 1.0);
        weight[4 * i + 0] = ((A * x1p - 5 * A) * x1p + 8 * A) * x1p - 4 * A;
        weight[4 * i + 1] = ((A + 2) * x1 - (A + 3)) * x1 * x1 + 1;
        weight[4 * i + 2] = ((A + 2) * x2 - (A + 3)) * x2 * x2 + 1;
        weight[4 * i + 3] = ((A * x2p - 5 * A) * x2p + 8 * A) * x2p - 4 * A;
        for (int64_t j = 0; j < 4; ++j)
        {
            int64_t s = input + j - 1;
            s = s < last ? s : last;
            s = s > 0 ? s : 0;
            index[4 * i + size_t(j)] = s;
        }
    }
    for (uint32_t oy = 0; oy < out_side; ++oy)
    {
        for (uint32_t ox = 0; ox < out_side; ++ox)
        {
            float* o = out + (size_t(oy) * out_side + ox) * dim;
            for (uint32_t c = 0; c < dim; ++c)
            {
                float acc = 0.0f;
                for (uint32_t jy = 0; jy < 4; ++jy)
                {
                    const float* row = pos + size_t(index[4 * oy + jy]) * side * dim;
                    float t = 0.0f;
                    for (uint32_t jx = 0; jx < 4; ++jx)
                    {
                        const float v = row[size_t(index[4 * ox + jx]) * dim + c] * weight[4 * ox + jx];
                        t = jx == 0 ? v : t + v;
                    }
                    const float v = t * weight[4 * oy + jy];
                    acc = jy == 0 ? v : acc + v;
                }
                o[c] = acc;
            }
        }
    }
}

bool TripoRunner::init(const TripoModel& model, ThreadPool& pool, Error& error)
{
    m_model = &model;
    m_pool = &pool;
    const size_t side = TripoModel::IMAGE_SIZE / TripoModel::VIT_PATCH;
    const size_t vit_dim = TripoModel::VIT_DIM;
    m_n_image = side * side + 1;
    const size_t n_plane = TripoModel::PLANE_TOKENS;
    const size_t taps = 3 * TripoModel::VIT_PATCH * TripoModel::VIT_PATCH;
    const size_t threads = static_cast<size_t>(pool.size());
    const size_t rows = n_plane > m_n_image ? n_plane : m_n_image;
    const size_t wide = rows * TripoModel::DIM;
    const size_t ff = n_plane * 2 * TripoModel::FF > m_n_image * TripoModel::VIT_FF ? n_plane * 2 * TripoModel::FF
                                                                                     : m_n_image * TripoModel::VIT_FF;
    const size_t code = 3 * TripoModel::CODE_CHANNELS * TripoModel::CODE_SIZE * TripoModel::CODE_SIZE;
    const bool ok = m_pos.resize_uninitialized(m_n_image * vit_dim) &&
                    m_patches.resize_uninitialized((m_n_image - 1) * taps) &&
                    m_image.resize_uninitialized(m_n_image * vit_dim) && m_residual.resize_uninitialized(wide) &&
                    m_x.resize_uninitialized(wide) && m_h.resize_uninitialized(wide) &&
                    m_q.resize_uninitialized(wide) && m_k.resize_uninitialized(wide) &&
                    m_v.resize_uninitialized(wide) && m_attn.resize_uninitialized(wide) &&
                    m_up.resize_uninitialized(ff) &&
                    m_scores.resize_uninitialized(threads * ATTENTION_QUERIES * (rows + KEY_TILE)) &&
                    m_k_tiles.resize_uninitialized(TripoModel::HEADS * (rows + KEY_TILE) * (TripoModel::DIM / TripoModel::HEADS)) &&
                    m_v_heads.resize_uninitialized(wide) && m_code.resize_uninitialized(code) &&
                    m_gemm_scratch.resize_uninitialized(threads * gemm_f32_scratch(TripoModel::FF > rows ? TripoModel::FF : rows)) &&
                    m_planes_last.resize_uninitialized(code) &&
                    m_query.resize_uninitialized(threads * QUERY_CHUNK *
                                                 (TripoModel::MLP_IN + 2 * TripoModel::MLP_WIDTH + F32_PANEL));
    if (!ok)
    {
        error.set("out of memory for the TripoSR buffers");
        return false;
    }
    // ViTEmbeddings.interpolate_pos_encoding of transformers 4.35.0: scale_factor (side + 0.1) / 14
    std::memcpy(m_pos.data(), model.vit_pos(), vit_dim * sizeof(float));
    const double scale =
        (static_cast<double>(side) + 0.1) / std::sqrt(double(TripoModel::VIT_POS_SIDE * TripoModel::VIT_POS_SIDE));
    tripo_bicubic_rows(model.vit_pos() + vit_dim, TripoModel::VIT_POS_SIDE, TripoModel::VIT_DIM, scale,
                       static_cast<uint32_t>(side), m_pos.data() + vit_dim);
    return true;
}

void TripoRunner::trace(const char* name, int layer, const float* data, size_t count) const
{
    if (m_trace != nullptr)
    {
        m_trace(m_trace_user, name, layer, data, count);
    }
}

// one unit is GEMM_PANELS panels of outputs for GEMM_TOKENS tokens
void TripoRunner::matmul(const TripoLinear& w, const float* bias, const float* x, size_t n_tokens, float* out,
                         size_t x_stride)
{
    const size_t GEMM_PANELS = 4;
    const size_t GEMM_TOKENS = 192;
    const size_t stride = x_stride != 0 ? x_stride : w.n_in;
    const size_t groups = (w.n_out / F32_PANEL + GEMM_PANELS - 1) / GEMM_PANELS;
    const size_t chunks = (n_tokens + GEMM_TOKENS - 1) / GEMM_TOKENS;
    const size_t scratch = m_gemm_scratch.size() / static_cast<size_t>(m_pool->size());
    parallel_dynamic(*m_pool, groups * chunks, 1,
                     [&](size_t first, size_t last, int thread)
                     {
                         for (size_t u = first; u < last; ++u)
                         {
                             const size_t t0 = (u % chunks) * GEMM_TOKENS;
                             const size_t nt = n_tokens - t0 < GEMM_TOKENS ? n_tokens - t0 : GEMM_TOKENS;
                             const size_t o0 = (u / chunks) * GEMM_PANELS * F32_PANEL;
                             const size_t o1 = o0 + GEMM_PANELS * F32_PANEL < w.n_out ? o0 + GEMM_PANELS * F32_PANEL : w.n_out;
                             gemm_f32_packed(w.panels, w.n_in, x + t0 * stride, stride, nt, o0, o1, out + t0 * w.n_out,
                                             w.n_out, m_gemm_scratch.data() + size_t(thread) * scratch);
                             if (bias != nullptr)
                             {
                                 for (size_t t = t0; t < t0 + nt; ++t)
                                 {
                                     float* o = out + t * w.n_out;
                                     for (size_t r = o0; r < o1; ++r)
                                     {
                                         o[r] += bias[r];
                                     }
                                 }
                             }
                         }
                     });
}

void TripoRunner::norm_rows(const float* x, const float* w, const float* b, float eps, size_t rows, size_t width,
                            float* out)
{
    parallel_for(*m_pool, rows, 1,
                 [&](size_t first, size_t last, int)
                 {
                     for (size_t r = first; r < last; ++r)
                     {
                         layer_norm(x + r * width, w, b, width, eps, out + r * width);
                     }
                 });
}

// per head, the keys as gemm panels of 16 positions (the KEY_TILE layout) and the values
// as head_dim / 16 panels over the positions; then per head and block of ATTENTION_QUERIES
// queries: scores = q k^T with gemm_f32_packed, softmax over the n_kv real positions,
// out = p v with gemm_f32_packed, divided by the sum. Units do not depend on the thread
void TripoRunner::attention(const float* q, const float* k, const float* v, size_t n_q, size_t n_kv, size_t heads,
                            size_t head_dim, float* out)
{
    const size_t stride = heads * head_dim;
    const size_t padded = (n_kv + KEY_TILE - 1) / KEY_TILE * KEY_TILE;
    const size_t per_head = padded * head_dim;
    parallel_for(*m_pool, heads, 1,
                 [&](size_t first, size_t last, int)
                 {
                     for (size_t head = first; head < last; ++head)
                     {
                         float* tiles = m_k_tiles.data() + head * per_head;
                         for (size_t p = 0; p < n_kv; ++p)
                         {
                             const float* row = k + p * stride + head * head_dim;
                             for (size_t d = 0; d < head_dim; ++d)
                             {
                                 tiles[key_index(head_dim, p, d)] = row[d];
                             }
                         }
                         for (size_t p = n_kv; p < padded; ++p)
                         {
                             for (size_t d = 0; d < head_dim; ++d)
                             {
                                 tiles[key_index(head_dim, p, d)] = 0.0f;
                             }
                         }
                         float* panels = m_v_heads.data() + head * n_kv * head_dim;
                         for (size_t j = 0; j < head_dim / F32_PANEL; ++j)
                         {
                             float* panel = panels + j * n_kv * F32_PANEL;
                             for (size_t p = 0; p < n_kv; ++p)
                             {
                                 std::memcpy(panel + p * F32_PANEL, v + p * stride + head * head_dim + j * F32_PANEL,
                                             F32_PANEL * sizeof(float));
                             }
                         }
                     }
                 });
    const size_t block = ATTENTION_QUERIES;
    const size_t blocks = (n_q + block - 1) / block;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t scores_per_thread = m_scores.size() / static_cast<size_t>(m_pool->size());
    const size_t gemm_per_thread = m_gemm_scratch.size() / static_cast<size_t>(m_pool->size());
    parallel_dynamic(*m_pool, heads * blocks, 1,
                     [&](size_t first, size_t last, int thread)
                     {
                         float* scores = m_scores.data() + size_t(thread) * scores_per_thread;
                         float* scratch = m_gemm_scratch.data() + size_t(thread) * gemm_per_thread;
                         for (size_t u = first; u < last; ++u)
                         {
                             const size_t head = u / blocks;
                             const size_t i = (u % blocks) * block;
                             const size_t count = n_q - i < block ? n_q - i : block;
                             gemm_f32_packed(m_k_tiles.data() + head * per_head, head_dim, q + i * stride + head * head_dim,
                                             stride, count, 0, padded, scores, padded, scratch);
                             float sums[ATTENTION_QUERIES];
                             for (size_t r = 0; r < count; ++r)
                             {
                                 float* row = scores + r * padded;
                                 float max = row[0] * scale;
                                 for (size_t p = 0; p < n_kv; ++p)
                                 {
                                     row[p] *= scale;
                                     max = row[p] > max ? row[p] : max;
                                 }
                                 for (size_t p = 0; p < n_kv; ++p)
                                 {
                                     row[p] -= max;
                                 }
                                 exp_values(row, n_kv);
                                 float lanes[8] = {};
                                 for (size_t p = 0; p < n_kv; ++p)
                                 {
                                     lanes[p & 7] += row[p];
                                 }
                                 sums[r] = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                                           ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
                             }
                             float* o = out + i * stride + head * head_dim;
                             gemm_f32_packed(m_v_heads.data() + head * n_kv * head_dim, n_kv, scores, padded, count, 0,
                                             head_dim, o, stride, scratch);
                             for (size_t r = 0; r < count; ++r)
                             {
                                 const float inverse = 1.0f / sums[r];
                                 for (size_t d = 0; d < head_dim; ++d)
                                 {
                                     o[r * stride + d] *= inverse;
                                 }
                             }
                         }
                     });
}

bool TripoRunner::encode_image(const float* rgb, Error& error)
{
    if (m_model == nullptr)
    {
        error.set("TripoRunner used before init");
        return false;
    }
    const TripoModel& m = *m_model;
    const size_t size = TripoModel::IMAGE_SIZE;
    const size_t patch = TripoModel::VIT_PATCH;
    const size_t side = size / patch;
    const size_t n = m_n_image;
    const size_t dim = TripoModel::VIT_DIM;
    const size_t taps = 3 * patch * patch;
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdev[3] = {0.229f, 0.224f, 0.225f};

    parallel_for(*m_pool, side * side, 1,
                 [&](size_t first, size_t last, int)
                 {
                     for (size_t p = first; p < last; ++p)
                     {
                         const size_t py = p / side;
                         const size_t px = p % side;
                         float* out = m_patches.data() + p * taps;
                         for (size_t c = 0; c < 3; ++c)
                         {
                             for (size_t ky = 0; ky < patch; ++ky)
                             {
                                 const float* row = rgb + ((py * patch + ky) * size + px * patch) * 3;
                                 for (size_t kx = 0; kx < patch; ++kx)
                                 {
                                     out[(c * patch + ky) * patch + kx] = (row[kx * 3 + c] - mean[c]) / stdev[c];
                                 }
                             }
                         }
                     }
                 });
    float* x = m_image.data();
    matmul(m.vit_patch(), m.vit_patch_b(), m_patches.data(), side * side, x + dim);
    std::memcpy(x, m.vit_cls(), dim * sizeof(float));
    add(x, m_pos.data(), n * dim);
    trace("vit_embed", -1, x, n * dim);

    for (uint32_t il = 0; il < TripoModel::VIT_LAYERS; ++il)
    {
        const TripoVitLayer& l = m.vit_layer(il);
        norm_rows(x, l.ln1_w, l.ln1_b, TripoModel::VIT_EPS, n, dim, m_h.data());
        matmul(l.q, l.q_b, m_h.data(), n, m_q.data());
        matmul(l.k, l.k_b, m_h.data(), n, m_k.data());
        matmul(l.v, l.v_b, m_h.data(), n, m_v.data());
        attention(m_q.data(), m_k.data(), m_v.data(), n, n, TripoModel::VIT_HEADS, dim / TripoModel::VIT_HEADS,
                  m_attn.data());
        matmul(l.out, l.out_b, m_attn.data(), n, m_h.data());
        add(x, m_h.data(), n * dim);
        norm_rows(x, l.ln2_w, l.ln2_b, TripoModel::VIT_EPS, n, dim, m_h.data());
        matmul(l.up, l.up_b, m_h.data(), n, m_up.data());
        parallel_for(*m_pool, n * TripoModel::VIT_FF, 4096,
                     [&](size_t first, size_t last, int)
                     {
                         for (size_t i = first; i < last; ++i)
                         {
                             m_up[i] = gelu_erf(m_up[i]);
                         }
                     });
        matmul(l.down, l.down_b, m_up.data(), n, m_h.data());
        add(x, m_h.data(), n * dim);
        trace("vit_layer", static_cast<int>(il), x, n * dim);
    }
    norm_rows(x, m.vit_norm_w(), m.vit_norm_b(), TripoModel::VIT_EPS, n, dim, x);
    trace("vit_tokens", -1, x, n * dim);
    return true;
}

bool TripoRunner::build_triplane(Error& error)
{
    if (m_model == nullptr)
    {
        error.set("TripoRunner used before init");
        return false;
    }
    const TripoModel& m = *m_model;
    const size_t n = TripoModel::PLANE_TOKENS;
    const size_t dim = TripoModel::DIM;
    const size_t plane = TripoModel::PLANE_SIZE * TripoModel::PLANE_SIZE;
    const size_t heads = TripoModel::HEADS;
    const size_t head_dim = dim / heads;
    const size_t ff = TripoModel::FF;

    // Triplane1DTokenizer: token (np, hp, wp) of channel c is embeddings[np][c][hp][wp]
    const float* planes = m.plane_embeddings();
    float* residual = m_residual.data();
    for (size_t t = 0; t < n; ++t)
    {
        const size_t np = t / plane;
        const size_t pixel = t % plane;
        for (size_t c = 0; c < dim; ++c)
        {
            residual[t * dim + c] = planes[(np * dim + c) * plane + pixel];
        }
    }
    // GroupNorm over (channels of the group, all tokens), then proj_in on the token rows
    const size_t group = dim / TripoModel::GROUPS;
    parallel_for(*m_pool, TripoModel::GROUPS, 1,
                 [&](size_t first, size_t last, int)
                 {
                     for (size_t g = first; g < last; ++g)
                     {
                         double sum = 0.0;
                         for (size_t t = 0; t < n; ++t)
                         {
                             for (size_t c = g * group; c < (g + 1) * group; ++c)
                             {
                                 sum += residual[t * dim + c];
                             }
                         }
                         const double mean = sum / double(n * group);
                         double squares = 0.0;
                         for (size_t t = 0; t < n; ++t)
                         {
                             for (size_t c = g * group; c < (g + 1) * group; ++c)
                             {
                                 const double d = residual[t * dim + c] - mean;
                                 squares += d * d;
                             }
                         }
                         const float rstd =
                             static_cast<float>(1.0 / std::sqrt(squares / double(n * group) + TripoModel::GROUP_EPS));
                         const float mean_f = static_cast<float>(mean);
                         for (size_t t = 0; t < n; ++t)
                         {
                             for (size_t c = g * group; c < (g + 1) * group; ++c)
                             {
                                 m_h[t * dim + c] =
                                     (residual[t * dim + c] - mean_f) * rstd * m.group_norm_w()[c] + m.group_norm_b()[c];
                             }
                         }
                     }
                 });
    float* x = m_x.data();
    matmul(m.proj_in(), m.proj_in_b(), m_h.data(), n, x);
    trace("bb_proj_in", -1, x, n * dim);

    const size_t n_image = m_n_image;
    for (uint32_t ib = 0; ib < TripoModel::BLOCKS; ++ib)
    {
        const TripoBlock& b = m.block(ib);
        norm_rows(x, b.ln1_w, b.ln1_b, TripoModel::LN_EPS, n, dim, m_h.data());
        matmul(b.self_q, nullptr, m_h.data(), n, m_q.data());
        matmul(b.self_k, nullptr, m_h.data(), n, m_k.data());
        matmul(b.self_v, nullptr, m_h.data(), n, m_v.data());
        attention(m_q.data(), m_k.data(), m_v.data(), n, n, heads, head_dim, m_attn.data());
        matmul(b.self_out, b.self_out_b, m_attn.data(), n, m_h.data());
        add(x, m_h.data(), n * dim);

        norm_rows(x, b.ln2_w, b.ln2_b, TripoModel::LN_EPS, n, dim, m_h.data());
        matmul(b.cross_q, nullptr, m_h.data(), n, m_q.data());
        matmul(b.cross_k, nullptr, m_image.data(), n_image, m_k.data());
        matmul(b.cross_v, nullptr, m_image.data(), n_image, m_v.data());
        attention(m_q.data(), m_k.data(), m_v.data(), n, n_image, heads, head_dim, m_attn.data());
        matmul(b.cross_out, b.cross_out_b, m_attn.data(), n, m_h.data());
        add(x, m_h.data(), n * dim);

        // GEGLU: the first half of the projection times the erf GELU of the second, in place
        // in the first half of each row
        norm_rows(x, b.ln3_w, b.ln3_b, TripoModel::LN_EPS, n, dim, m_h.data());
        matmul(b.ff_in, b.ff_in_b, m_h.data(), n, m_up.data());
        parallel_for(*m_pool, n, 1,
                     [&](size_t first, size_t last, int)
                     {
                         for (size_t t = first; t < last; ++t)
                         {
                             float* row = m_up.data() + t * 2 * ff;
                             for (size_t i = 0; i < ff; ++i)
                             {
                                 row[i] = row[i] * gelu_erf(row[ff + i]);
                             }
                         }
                     });
        matmul(b.ff_out, b.ff_out_b, m_up.data(), n, m_h.data(), 2 * ff);
        add(x, m_h.data(), n * dim);
        trace("bb_block", static_cast<int>(ib), x, n * dim);
    }
    matmul(m.proj_out(), m.proj_out_b(), x, n, m_h.data());
    add(m_h.data(), residual, n * dim);
    trace("bb_out", -1, m_h.data(), n * dim);

    // ConvTranspose2d(DIM, CODE_CHANNELS, 2, stride 2) per plane
    const TripoLinear& up = m.upsample();
    matmul(up, nullptr, m_h.data(), n, m_attn.data());
    const size_t code_side = TripoModel::CODE_SIZE;
    const size_t channels = TripoModel::CODE_CHANNELS;
    for (size_t t = 0; t < n; ++t)
    {
        const size_t np = t / plane;
        const size_t hp = (t % plane) / TripoModel::PLANE_SIZE;
        const size_t wp = t % TripoModel::PLANE_SIZE;
        const float* row = m_attn.data() + t * up.n_out;
        for (size_t co = 0; co < channels; ++co)
        {
            for (size_t ky = 0; ky < 2; ++ky)
            {
                for (size_t kx = 0; kx < 2; ++kx)
                {
                    m_code[((np * channels + co) * code_side + 2 * hp + ky) * code_side + 2 * wp + kx] =
                        row[(co * 2 + ky) * 2 + kx] + m.upsample_b()[co];
                }
            }
        }
    }
    trace("scene_codes", -1, m_code.data(), m_code.size());
    for (size_t np = 0; np < 3; ++np)
    {
        for (size_t pixel = 0; pixel < code_side * code_side; ++pixel)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                m_planes_last[(np * code_side * code_side + pixel) * channels + c] =
                    m_code[(np * channels + c) * code_side * code_side + pixel];
            }
        }
    }
    return true;
}

namespace
{

// F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False) of one plane
// at (gx, gy) in [-1, 1], as PyTorch v2.11.0 GridSamplerKernel.cpp ApplyGridSample computes
// it; plane is [size][size][channels]
void sample_plane(const float* plane, int size, size_t channels, float gx, float gy, float* out)
{
    const float half = static_cast<float>(size) / 2;
    const float x = (gx + 1.0f) * half - 0.5f;
    const float y = (gy + 1.0f) * half - 0.5f;
    const float x_w = std::floor(x);
    const float y_n = std::floor(y);
    const float w = x - x_w;
    const float e = 1.0f - w;
    const float n = y - y_n;
    const float s = 1.0f - n;
    const float nw = s * e;
    const float ne = s * w;
    const float sw = n * e;
    const float se = n * w;
    const int ix = static_cast<int>(x_w);
    const int iy = static_cast<int>(y_n);
    const bool west = ix > -1 && ix < size;
    const bool east = ix + 1 > -1 && ix + 1 < size;
    const bool north = iy > -1 && iy < size;
    const bool south = iy + 1 > -1 && iy + 1 < size;
    const float* p_nw = plane + (size_t(iy) * size_t(size) + size_t(ix)) * channels;
    const float* p_ne = p_nw + channels;
    const float* p_sw = p_nw + size_t(size) * channels;
    const float* p_se = p_sw + channels;
    for (size_t c = 0; c < channels; ++c)
    {
        const float v_nw = west && north ? p_nw[c] : 0.0f;
        const float v_ne = east && north ? p_ne[c] : 0.0f;
        const float v_sw = west && south ? p_sw[c] : 0.0f;
        const float v_se = east && south ? p_se[c] : 0.0f;
        out[c] = v_nw * nw + v_ne * ne + v_sw * sw + v_se * se;
    }
}

void add_bias_rows(float* x, const float* bias, size_t rows, size_t width)
{
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t i = 0; i < width; ++i)
        {
            x[r * width + i] += bias[i];
        }
    }
}

// x / (1 + exp(-x)), the form of PyTorch's vectorized silu; tmp holds n floats
void silu_values(float* x, float* tmp, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        tmp[i] = -x[i];
    }
    exp_values(tmp, n);
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = x[i] / (1.0f + tmp[i]);
    }
}

}

bool TripoRunner::query_points(PointSource source, const void* user, size_t n, float* density, float* color,
                               Error& error)
{
    if (m_model == nullptr)
    {
        error.set("TripoRunner used before init");
        return false;
    }
    const TripoModel& m = *m_model;
    const size_t in = TripoModel::MLP_IN;
    const size_t width = TripoModel::MLP_WIDTH;
    const size_t channels = TripoModel::CODE_CHANNELS;
    const int side = static_cast<int>(TripoModel::CODE_SIZE);
    const size_t plane = size_t(side) * size_t(side) * channels;
    const size_t per_thread = m_query.size() / static_cast<size_t>(m_pool->size());
    const size_t scratch_per_thread = m_gemm_scratch.size() / static_cast<size_t>(m_pool->size());
    const size_t chunks = (n + QUERY_CHUNK - 1) / QUERY_CHUNK;
    const float low = -TripoModel::RADIUS;
    const float span = TripoModel::RADIUS - low;
    parallel_dynamic(*m_pool, chunks, 1,
                     [&](size_t first, size_t last, int thread)
                     {
                         float* features = m_query.data() + size_t(thread) * per_thread;
                         float* a = features + QUERY_CHUNK * in;
                         float* b = a + QUERY_CHUNK * width;
                         float* out = b + QUERY_CHUNK * width;
                         float* scratch = m_gemm_scratch.data() + size_t(thread) * scratch_per_thread;
                         for (size_t u = first; u < last; ++u)
                         {
                             const size_t p0 = u * QUERY_CHUNK;
                             const size_t count = n - p0 < QUERY_CHUNK ? n - p0 : QUERY_CHUNK;
                             for (size_t i = 0; i < count; ++i)
                             {
                                 float p[3];
                                 source(user, p0 + i, p);
                                 float q[3];
                                 for (int d = 0; d < 3; ++d)
                                 {
                                     q[d] = (p[d] - low) / span * 2.0f + -1.0f;
                                 }
                                 float* f = features + i * in;
                                 sample_plane(m_planes_last.data(), side, channels, q[0], q[1], f);
                                 sample_plane(m_planes_last.data() + plane, side, channels, q[0], q[2], f + channels);
                                 sample_plane(m_planes_last.data() + 2 * plane, side, channels, q[1], q[2],
                                              f + 2 * channels);
                             }
                             const float* x = features;
                             size_t x_width = in;
                             for (uint32_t l = 0; l + 1 < TripoModel::MLP_LAYERS; ++l)
                             {
                                 const TripoLinear& w = m.mlp(l);
                                 gemm_f32_packed(w.panels, w.n_in, x, x_width, count, 0, width, a, width, scratch);
                                 add_bias_rows(a, m.mlp_b(l), count, width);
                                 silu_values(a, b, count * width);
                                 float* t = a;
                                 a = b;
                                 b = t;
                                 x = b;
                                 x_width = width;
                             }
                             const TripoLinear& w = m.mlp(TripoModel::MLP_LAYERS - 1);
                             gemm_f32_packed(w.panels, w.n_in, x, x_width, count, 0, F32_PANEL, out, F32_PANEL, scratch);
                             add_bias_rows(out, m.mlp_b(TripoModel::MLP_LAYERS - 1), count, F32_PANEL);
                             float* d = density + p0;
                             for (size_t i = 0; i < count; ++i)
                             {
                                 d[i] = out[i * F32_PANEL] + TripoModel::DENSITY_BIAS;
                             }
                             exp_values(d, count);
                             if (color != nullptr)
                             {
                                 float* c = color + 3 * p0;
                                 for (size_t i = 0; i < count; ++i)
                                 {
                                     for (int k = 0; k < 3; ++k)
                                     {
                                         c[3 * i + size_t(k)] = -out[i * F32_PANEL + 1 + size_t(k)];
                                     }
                                 }
                                 exp_values(c, 3 * count);
                                 for (size_t i = 0; i < 3 * count; ++i)
                                 {
                                     c[i] = 1.0f / (1.0f + c[i]);
                                 }
                             }
                         }
                     });
    return true;
}

bool TripoRunner::query(const float* points, size_t n, float* density, float* color, Error& error)
{
    return query_points([](const void* user, size_t index, float out[3])
                        { std::memcpy(out, static_cast<const float*>(user) + 3 * index, 3 * sizeof(float)); },
                        points, n, density, color, error);
}

namespace
{

#include "triposr_mc_tables.inc"

struct McPoint
{
    float x;
    float y;
    float z;
};

// torchmcubes VertexInterp
McPoint vertex_interp(float isolevel, McPoint p1, McPoint p2, float v1, float v2)
{
    if (std::fabs(isolevel - v1) < 1.0e-4f)
    {
        return p1;
    }
    if (std::fabs(isolevel - v2) < 1.0e-4f)
    {
        return p2;
    }
    if (std::fabs(v1 - v2) < 1.0e-4f)
    {
        return p1;
    }
    const float mu = (isolevel - v1) / (v2 - v1);
    return McPoint{p1.x + mu * (p2.x - p1.x), p1.y + mu * (p2.y - p1.y), p1.z + mu * (p2.z - p1.z)};
}

// torchmcubes Polygonise and the cell setup of mcubes_method for the cell at (x, y, z),
// vol[z][y][x] with vol of side size; returns the triangle count, writing 3 points each
// when out is not null
int polygonise(const float* vol, size_t size, int x, int y, int z, McPoint* out)
{
    static const int INDEX[8] = {0, 1, 4, 5, 3, 2, 7, 6};
    McPoint p[8];
    float val[8];
    for (int i = 0; i < 8; ++i)
    {
        const int dx = i % 2;
        const int dy = (i / 2) % 2;
        const int dz = i / 4;
        p[INDEX[i]] = McPoint{float(x + dx), float(y + dy), float(z + dz)};
        val[INDEX[i]] = vol[(size_t(z + dz) * size + size_t(y + dy)) * size + size_t(x + dx)];
    }
    const float iso = 0.0f;
    int cube = 0;
    for (int i = 0; i < 8; ++i)
    {
        if (val[i] < iso)
        {
            cube |= 1 << i;
        }
    }
    const int edges = MC_EDGES[cube];
    if (edges == 0)
    {
        return 0;
    }
    if (out == nullptr)
    {
        int count = 0;
        for (int i = 0; MC_TRIANGLES[cube][i] != -1; i += 3)
        {
            count += 1;
        }
        return count;
    }
    static const int FROM[12] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3};
    static const int TO[12] = {1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7};
    McPoint list[12] = {};
    for (int e = 0; e < 12; ++e)
    {
        if (edges & (1 << e))
        {
            list[e] = vertex_interp(iso, p[FROM[e]], p[TO[e]], val[FROM[e]], val[TO[e]]);
        }
    }
    int count = 0;
    for (int i = 0; MC_TRIANGLES[cube][i] != -1; i += 3)
    {
        out[3 * count + 0] = list[MC_TRIANGLES[cube][i]];
        out[3 * count + 1] = list[MC_TRIANGLES[cube][i + 1]];
        out[3 * count + 2] = list[MC_TRIANGLES[cube][i + 2]];
        count += 1;
    }
    return count;
}

uint64_t point_hash(const McPoint& p)
{
    uint32_t bits[3];
    std::memcpy(bits, &p, sizeof(bits));
    uint64_t h = 0x9e3779b97f4a7c15ull;
    for (uint32_t b : bits)
    {
        h = (h ^ b) * 0xff51afd7ed558ccdull;
        h ^= h >> 32;
    }
    return h;
}

struct GridSource
{
    const float* lin;
    size_t size;
};

}

namespace
{

// PyTorch v2.11.0 UpSampleKernel.cpp _compute_indices_min_size_weights_aa with the
// HelperInterpLinear filter, scale input / output in float
struct AaTaps
{
    Array<int64_t> first;
    Array<int64_t> count;
    Array<float> weights;
    int64_t width = 0;
};

bool aa_taps(int64_t input, int64_t output, AaTaps& taps)
{
    const float scale = static_cast<float>(input) / static_cast<float>(output);
    const float support = static_cast<float>((2 * 0.5) * (scale >= 1.0f ? static_cast<double>(scale) : 1.0));
    taps.width = static_cast<int64_t>(std::ceil(support)) * 2 + 1;
    if (!taps.first.resize_uninitialized(size_t(output)) || !taps.count.resize_uninitialized(size_t(output)) ||
        !taps.weights.resize_uninitialized(size_t(output * taps.width)))
    {
        return false;
    }
    const float invscale = scale >= 1.0f ? static_cast<float>(1.0 / static_cast<double>(scale)) : 1.0f;
    for (int64_t i = 0; i < output; ++i)
    {
        const float center = static_cast<float>(static_cast<double>(scale) * (static_cast<double>(i) + 0.5));
        int64_t xmin = static_cast<int64_t>(static_cast<double>(center - support) + 0.5);
        xmin = xmin > 0 ? xmin : 0;
        int64_t xmax = static_cast<int64_t>(static_cast<double>(center + support) + 0.5);
        xmax = xmax < input ? xmax : input;
        int64_t xsize = xmax - xmin;
        xsize = xsize < 0 ? 0 : (xsize > taps.width ? taps.width : xsize);
        float* w = taps.weights.data() + i * taps.width;
        float total = 0.0f;
        for (int64_t j = 0; j < xsize; ++j)
        {
            const float x = static_cast<float>(
                (static_cast<double>(static_cast<float>(j + xmin) - center) + 0.5) * static_cast<double>(invscale));
            const float a = std::fabs(x);
            w[j] = a < 1.0f ? static_cast<float>(1.0 - static_cast<double>(a)) : 0.0f;
            total += w[j];
        }
        if (total != 0.0f)
        {
            for (int64_t j = 0; j < xsize; ++j)
            {
                w[j] /= total;
            }
        }
        taps.first[size_t(i)] = xmin;
        taps.count[size_t(i)] = xsize;
    }
    return true;
}

// one output value: t0 w0, then += tj wj (interpolate_aa_single_dim)
float aa_sum(const float* src, size_t stride, const AaTaps& taps, size_t i)
{
    const float* w = taps.weights.data() + int64_t(i) * taps.width;
    const float* s = src + size_t(taps.first[i]) * stride;
    float out = s[0] * w[0];
    for (int64_t j = 1; j < taps.count[i]; ++j)
    {
        out = std::fma(s[size_t(j) * stride], w[j], out);
    }
    return out;
}

}

bool tripo_prepare_image(const Image& image, float foreground_ratio, Array<float>& rgb, Error& error)
{
    size_t width = image.width;
    size_t height = image.height;
    Array<float> src;
    if (!image.alpha.empty())
    {
        size_t y1 = height, y2 = 0, x1 = width, x2 = 0;
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                if (image.alpha[y * width + x] > 0)
                {
                    y1 = y < y1 ? y : y1;
                    y2 = y > y2 ? y : y2;
                    x1 = x < x1 ? x : x1;
                    x2 = x > x2 ? x : x2;
                }
            }
        }
        if (y1 > y2)
        {
            error.set("the image is fully transparent");
            return false;
        }
        // fg = image[y1:y2, x1:x2], padded to a square, then to int(size / ratio)
        const size_t h = y2 - y1;
        const size_t w = x2 - x1;
        const size_t size = h > w ? h : w;
        const size_t full = static_cast<size_t>(static_cast<double>(size) / static_cast<double>(foreground_ratio));
        if (size == 0 || full < size)
        {
            error.set("foreground of %zu x %zu pixels with ratio %g", w, h, double(foreground_ratio));
            return false;
        }
        const size_t top = (size - h) / 2 + (full - size) / 2;
        const size_t left = (size - w) / 2 + (full - size) / 2;
        if (!src.resize_uninitialized(full * full * 3))
        {
            error.set("out of memory preparing the image");
            return false;
        }
        for (size_t y = 0; y < full; ++y)
        {
            for (size_t x = 0; x < full; ++x)
            {
                const bool inside = y >= top && y < top + h && x >= left && x < left + w;
                const size_t sy = y1 + y - (inside ? top : 0);
                const size_t sx = x1 + x - (inside ? left : 0);
                const float a = inside ? static_cast<float>(image.alpha[sy * width + sx]) / 255.0f : 0.0f;
                for (size_t c = 0; c < 3; ++c)
                {
                    const float v = inside ? static_cast<float>(image.rgb[(sy * width + sx) * 3 + c]) / 255.0f : 0.0f;
                    const float blended = v * a + (1.0f - a) * 0.5f;
                    const uint8_t byte = static_cast<uint8_t>(blended * 255.0f);
                    src[(y * full + x) * 3 + c] = static_cast<float>(byte) / 255.0f;
                }
            }
        }
        width = height = full;
    }
    else
    {
        if (!src.resize_uninitialized(width * height * 3))
        {
            error.set("out of memory preparing the image");
            return false;
        }
        for (size_t i = 0; i < width * height * 3; ++i)
        {
            src[i] = static_cast<float>(image.rgb[i]) / 255.0f;
        }
    }

    // separable_upsample_generic_Nd_kernel_impl: the width first, then the height; a
    // dimension already at the output size is left alone
    const size_t out = TripoModel::IMAGE_SIZE;
    if (!rgb.resize_uninitialized(out * out * 3))
    {
        error.set("out of memory preparing the image");
        return false;
    }
    Array<float> mid;
    const float* rows = src.data();
    if (width != out)
    {
        AaTaps taps;
        if (!aa_taps(int64_t(width), int64_t(out), taps) || !mid.resize_uninitialized(height * out * 3))
        {
            error.set("out of memory preparing the image");
            return false;
        }
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < out; ++x)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    mid[(y * out + x) * 3 + c] = aa_sum(src.data() + y * width * 3 + c, 3, taps, x);
                }
            }
        }
        rows = mid.data();
    }
    if (height != out)
    {
        AaTaps taps;
        if (!aa_taps(int64_t(height), int64_t(out), taps))
        {
            error.set("out of memory preparing the image");
            return false;
        }
        for (size_t y = 0; y < out; ++y)
        {
            for (size_t x = 0; x < out * 3; ++x)
            {
                rgb[y * out * 3 + x] = aa_sum(rows + x, out * 3, taps, y);
            }
        }
    }
    else
    {
        std::memcpy(rgb.data(), rows, out * out * 3 * sizeof(float));
    }
    return true;
}

bool tripo_marching_cubes(ThreadPool& pool, const float* vol, uint32_t resolution, Array<float>& points,
                          Array<uint32_t>& faces, Error& error)
{
    const size_t size = resolution;
    const size_t cells = size - 1;
    // triangles per z slab counted, then written at their offsets, so the order is the
    // serial order of mcubes_method whatever the thread count
    Array<size_t> offsets;
    if (!offsets.resize_uninitialized(cells + 1))
    {
        error.set("out of memory for the marching cubes");
        return false;
    }
    parallel_dynamic(pool, cells, 1,
                     [&](size_t first, size_t last, int)
                     {
                         for (size_t z = first; z < last; ++z)
                         {
                             size_t count = 0;
                             for (size_t y = 0; y < cells; ++y)
                             {
                                 for (size_t x = 0; x < cells; ++x)
                                 {
                                     count += size_t(polygonise(vol, size, int(x), int(y), int(z), nullptr));
                                 }
                             }
                             offsets[z + 1] = count;
                         }
                     });
    offsets[0] = 0;
    for (size_t z = 0; z < cells; ++z)
    {
        offsets[z + 1] += offsets[z];
    }
    const size_t n_triangles = offsets[cells];
    Array<McPoint> corners;
    if (!corners.resize_uninitialized(3 * n_triangles + 15))
    {
        error.set("out of memory for %zu triangles", n_triangles);
        return false;
    }
    parallel_dynamic(pool, cells, 1,
                     [&](size_t first, size_t last, int)
                     {
                         for (size_t z = first; z < last; ++z)
                         {
                             McPoint* out = corners.data() + 3 * offsets[z];
                             for (size_t y = 0; y < cells; ++y)
                             {
                                 for (size_t x = 0; x < cells; ++x)
                                 {
                                     out += 3 * polygonise(vol, size, int(x), int(y), int(z), out);
                                 }
                             }
                         }
                     });

    // unique vertices in first seen order, by exact position as the unordered_map of XYZ
    size_t slots = 16;
    while (slots < 6 * n_triangles)
    {
        slots *= 2;
    }
    Array<uint32_t> table;
    Array<McPoint> unique;
    if (!table.resize_uninitialized(slots) || !unique.reserve(3 * n_triangles) ||
        !faces.resize_uninitialized(3 * n_triangles))
    {
        error.set("out of memory for %zu triangles", n_triangles);
        return false;
    }
    std::memset(table.data(), 0xff, slots * sizeof(uint32_t));
    for (size_t i = 0; i < 3 * n_triangles; ++i)
    {
        const McPoint& p = corners[i];
        size_t slot = static_cast<size_t>(point_hash(p)) & (slots - 1);
        for (;;)
        {
            const uint32_t index = table[slot];
            if (index == 0xffffffffu)
            {
                table[slot] = static_cast<uint32_t>(unique.size());
                unique.push_back_unchecked(p);
                faces[i] = table[slot];
                break;
            }
            const McPoint& q = unique[index];
            if (q.x == p.x && q.y == p.y && q.z == p.z)
            {
                faces[i] = index;
                break;
            }
            slot = (slot + 1) & (slots - 1);
        }
    }

    if (!points.resize_uninitialized(3 * unique.size()))
    {
        error.set("out of memory for %zu vertices", unique.size());
        return false;
    }
    std::memcpy(points.data(), unique.data(), unique.size() * sizeof(McPoint));
    return true;
}

bool TripoRunner::extract_mesh(uint32_t resolution, float threshold, TripoMesh& mesh, Error& error)
{
    if (m_model == nullptr || resolution < 2)
    {
        error.set("TripoRunner::extract_mesh: no model or resolution below 2");
        return false;
    }
    const size_t size = resolution;
    Array<float> lin;
    Array<float> vol;
    if (!lin.resize_uninitialized(size) || !vol.resize_uninitialized(size * size * size))
    {
        error.set("out of memory for a %u^3 density grid", resolution);
        return false;
    }
    // torch.linspace(0, 1, size) (RangeFactoriesKernel.cpp linspace_kernel), then
    // scale_tensor to (-RADIUS, RADIUS)
    const float step = (1.0f - 0.0f) / static_cast<float>(size - 1);
    for (size_t i = 0; i < size; ++i)
    {
        const float v = i < size / 2 ? 0.0f + step * static_cast<float>(i)
                                     : 1.0f - step * static_cast<float>(size - i - 1);
        lin[i] = v * (TripoModel::RADIUS - -TripoModel::RADIUS) + -TripoModel::RADIUS;
    }
    GridSource grid = {lin.data(), size};
    if (!query_points(
            [](const void* user, size_t index, float out[3])
            {
                const GridSource& g = *static_cast<const GridSource*>(user);
                out[0] = g.lin[index / (g.size * g.size)];
                out[1] = g.lin[(index / g.size) % g.size];
                out[2] = g.lin[index % g.size];
            },
            &grid, size * size * size, vol.data(), nullptr, error))
    {
        return false;
    }
    trace("density_act", -1, vol.data(), vol.size());
    for (size_t i = 0; i < vol.size(); ++i)
    {
        vol[i] = vol[i] - threshold;
    }

    Array<float> grid_points;
    if (!tripo_marching_cubes(*m_pool, vol.data(), resolution, grid_points, mesh.faces, error))
    {
        return false;
    }

    // v_pos[..., [2, 1, 0]] / (resolution - 1), scale_tensor to (-RADIUS, RADIUS)
    const size_t n_vertices = grid_points.size() / 3;
    if (!mesh.positions.resize_uninitialized(3 * n_vertices) || !mesh.colors.resize_uninitialized(3 * n_vertices))
    {
        error.set("out of memory for %zu vertices", n_vertices);
        return false;
    }
    const float scale = static_cast<float>(resolution) - 1.0f;
    for (size_t i = 0; i < n_vertices; ++i)
    {
        const float* g = grid_points.data() + 3 * i;
        const float v[3] = {g[2] / scale, g[1] / scale, g[0] / scale};
        for (int d = 0; d < 3; ++d)
        {
            mesh.positions[3 * i + size_t(d)] =
                v[d] * (TripoModel::RADIUS - -TripoModel::RADIUS) + -TripoModel::RADIUS;
        }
    }
    Array<float> density;
    if (!density.resize_uninitialized(n_vertices))
    {
        error.set("out of memory for %zu vertices", n_vertices);
        return false;
    }
    return query(mesh.positions.data(), n_vertices, density.data(), mesh.colors.data(), error);
}

}
