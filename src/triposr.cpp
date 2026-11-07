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
        auto lin = [&](const char* suffix, uint64_t n_out, uint64_t n_in, Weight& w, const float** bias)
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
    m_upsample.type = GgmlType::F32;
    m_upsample.data = reinterpret_cast<const uint8_t*>(m_upsample_rows.data());
    m_upsample.n_in = DIM;
    m_upsample.n_out = rows;
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
                    m_up.resize_uninitialized(ff) && m_scores.resize_uninitialized(threads * rows) &&
                    m_code.resize_uninitialized(code);
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

void TripoRunner::matmul(const Weight& w, const float* bias, const float* x, size_t n_tokens, float* out,
                         size_t x_stride)
{
    const size_t stride = x_stride != 0 ? x_stride : w.n_in;
    parallel_for(*m_pool, w.n_out, w.row_group(),
                 [&](size_t first, size_t last, int)
                 {
                     w.matmul(x, stride, nullptr, 0, nullptr, 0, n_tokens, first, last, out, w.n_out);
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

void TripoRunner::attention(const float* q, const float* k, const float* v, size_t n_q, size_t n_kv, size_t heads,
                            size_t head_dim, float* out)
{
    const size_t stride = heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    parallel_for(*m_pool, heads * n_q, 1,
                 [&](size_t first, size_t last, int thread)
                 {
                     float* scores = m_scores.data() + size_t(thread) * (m_scores.size() / size_t(m_pool->size()));
                     for (size_t u = first; u < last; ++u)
                     {
                         const size_t head = u / n_q;
                         const size_t query = u % n_q;
                         const float* qh = q + query * stride + head * head_dim;
                         for (size_t j = 0; j < n_kv; ++j)
                         {
                             scores[j] = dot(qh, k + j * stride + head * head_dim, head_dim) * scale;
                         }
                         softmax(scores, n_kv);
                         float* o = out + query * stride + head * head_dim;
                         for (size_t d = 0; d < head_dim; ++d)
                         {
                             o[d] = 0.0f;
                         }
                         for (size_t j = 0; j < n_kv; ++j)
                         {
                             const float p = scores[j];
                             const float* vh = v + j * stride + head * head_dim;
                             for (size_t d = 0; d < head_dim; ++d)
                             {
                                 o[d] += p * vh[d];
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
    const Weight& up = m.upsample();
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
    return true;
}

}
