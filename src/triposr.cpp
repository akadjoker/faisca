#include "faisca/triposr.h"

#include "faisca/ops.h"

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

void gelu_erf(float* x, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = x[i] * 0.5f * (1.0f + std::erf(x[i] * 0.70710678118654752440f));
    }
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

bool TripoModel::linear(const char* name, uint64_t n_out, uint64_t n_in, Weight& w, const float** bias, Error& error)
{
    char key[192];
    std::snprintf(key, sizeof(key), "%s.weight", name);
    const float* data = f32(key, n_out, n_in, 0, 0, error);
    if (data == nullptr)
    {
        return false;
    }
    w.type = GgmlType::F32;
    w.packed = false;
    w.data = reinterpret_cast<const uint8_t*>(data);
    w.n_in = n_in;
    w.n_out = n_out;
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
    m_vit_patch.type = GgmlType::F32;
    m_vit_patch.data = reinterpret_cast<const uint8_t*>(patch);
    m_vit_patch.n_in = 3 * VIT_PATCH * VIT_PATCH;
    m_vit_patch.n_out = VIT_DIM;

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
        auto lin = [&](const char* suffix, uint64_t n_out, uint64_t n_in, Weight& w, const float** b)
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
    const size_t dim = TripoModel::VIT_DIM;
    m_n_tokens = side * side + 1;
    const size_t n = m_n_tokens;
    const size_t taps = 3 * TripoModel::VIT_PATCH * TripoModel::VIT_PATCH;
    const size_t threads = static_cast<size_t>(pool.size());
    const bool ok = m_pos.resize_uninitialized(n * dim) && m_patches.resize_uninitialized((n - 1) * taps) &&
                    m_x.resize_uninitialized(n * dim) && m_h.resize_uninitialized(n * dim) &&
                    m_q.resize_uninitialized(n * dim) && m_k.resize_uninitialized(n * dim) &&
                    m_v.resize_uninitialized(n * dim) && m_attn.resize_uninitialized(n * dim) &&
                    m_up.resize_uninitialized(n * TripoModel::VIT_FF) && m_scores.resize_uninitialized(threads * n);
    if (!ok)
    {
        error.set("out of memory for the TripoSR buffers");
        return false;
    }
    // ViTEmbeddings.interpolate_pos_encoding of transformers 4.35.0: scale_factor (side + 0.1) / 14
    std::memcpy(m_pos.data(), model.vit_pos(), dim * sizeof(float));
    const double scale = (static_cast<double>(side) + 0.1) / std::sqrt(double(TripoModel::VIT_POS_SIDE * TripoModel::VIT_POS_SIDE));
    tripo_bicubic_rows(model.vit_pos() + dim, TripoModel::VIT_POS_SIDE, TripoModel::VIT_DIM, scale,
                       static_cast<uint32_t>(side), m_pos.data() + dim);
    return true;
}

void TripoRunner::trace(const char* name, int layer, const float* data, size_t count) const
{
    if (m_trace != nullptr)
    {
        m_trace(m_trace_user, name, layer, data, count);
    }
}

void TripoRunner::matmul(const Weight& w, const float* bias, const float* x, size_t n_tokens, float* out)
{
    parallel_for(*m_pool, w.n_out, w.row_group(),
                 [&](size_t first, size_t last, int)
                 {
                     w.matmul(x, w.n_in, nullptr, 0, nullptr, 0, n_tokens, first, last, out, w.n_out);
                     if (bias != nullptr)
                     {
                         for (size_t t = 0; t < n_tokens; ++t)
                         {
                             float* o = out + t * w.n_out;
                             for (size_t r = first; r < last; ++r)
                             {
                                 o[r] += bias[r];
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

void TripoRunner::vit_attention(const TripoVitLayer&)
{
    const size_t n = m_n_tokens;
    const size_t dim = TripoModel::VIT_DIM;
    const size_t head_dim = dim / TripoModel::VIT_HEADS;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    parallel_for(*m_pool, TripoModel::VIT_HEADS * n, 1,
                 [&](size_t first, size_t last, int thread)
                 {
                     float* scores = m_scores.data() + size_t(thread) * n;
                     for (size_t u = first; u < last; ++u)
                     {
                         const size_t head = u / n;
                         const size_t query = u % n;
                         const float* q = m_q.data() + query * dim + head * head_dim;
                         for (size_t j = 0; j < n; ++j)
                         {
                             scores[j] = dot(q, m_k.data() + j * dim + head * head_dim, head_dim) * scale;
                         }
                         softmax(scores, n);
                         float* o = m_attn.data() + query * dim + head * head_dim;
                         for (size_t d = 0; d < head_dim; ++d)
                         {
                             o[d] = 0.0f;
                         }
                         for (size_t j = 0; j < n; ++j)
                         {
                             const float p = scores[j];
                             const float* v = m_v.data() + j * dim + head * head_dim;
                             for (size_t d = 0; d < head_dim; ++d)
                             {
                                 o[d] += p * v[d];
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
    const size_t n = m_n_tokens;
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
    matmul(m.vit_patch(), m.vit_patch_b(), m_patches.data(), side * side, m_x.data() + dim);
    std::memcpy(m_x.data(), m.vit_cls(), dim * sizeof(float));
    add(m_x.data(), m_pos.data(), n * dim);
    trace("vit_embed", -1, m_x.data(), n * dim);

    for (uint32_t il = 0; il < TripoModel::VIT_LAYERS; ++il)
    {
        const TripoVitLayer& l = m.vit_layer(il);
        norm_rows(m_x.data(), l.ln1_w, l.ln1_b, TripoModel::VIT_EPS, n, dim, m_h.data());
        matmul(l.q, l.q_b, m_h.data(), n, m_q.data());
        matmul(l.k, l.k_b, m_h.data(), n, m_k.data());
        matmul(l.v, l.v_b, m_h.data(), n, m_v.data());
        vit_attention(l);
        matmul(l.out, l.out_b, m_attn.data(), n, m_h.data());
        add(m_x.data(), m_h.data(), n * dim);
        norm_rows(m_x.data(), l.ln2_w, l.ln2_b, TripoModel::VIT_EPS, n, dim, m_h.data());
        matmul(l.up, l.up_b, m_h.data(), n, m_up.data());
        parallel_for(*m_pool, n, 1,
                     [&](size_t first, size_t last, int)
                     { gelu_erf(m_up.data() + first * TripoModel::VIT_FF, (last - first) * TripoModel::VIT_FF); });
        matmul(l.down, l.down_b, m_up.data(), n, m_h.data());
        add(m_x.data(), m_h.data(), n * dim);
        trace("vit_layer", static_cast<int>(il), m_x.data(), n * dim);
    }
    norm_rows(m_x.data(), m.vit_norm_w(), m.vit_norm_b(), TripoModel::VIT_EPS, n, dim, m_x.data());
    trace("vit_tokens", -1, m_x.data(), n * dim);
    return true;
}

}
