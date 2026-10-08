#include "faisca/vision_model.h"

#include "faisca/ops.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

bool key_u32(const GgufFile& file, const char* key, uint32_t& out, Error& error, bool required = true)
{
    uint64_t value = 0;
    if (!file.get_uint(key, value))
    {
        if (required)
        {
            error.set("%s is missing", key);
        }
        return !required;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

}

float* VisionModel::reserve(size_t count)
{
    if (m_used + count > m_storage.size())
    {
        return nullptr;
    }
    float* out = m_storage.data() + m_used;
    m_used += count;
    return out;
}

bool VisionModel::vector(const GgufFile& file, const char* name, size_t count, const float*& out, Error& error)
{
    const GgufTensor* t = file.find_tensor(name);
    if (t == nullptr)
    {
        error.set("tensor '%s' is missing", name);
        return false;
    }
    if (t->n_elements != count)
    {
        error.set("tensor '%s' has %" PRIu64 " values, expected %zu", name, t->n_elements, count);
        return false;
    }
    float* dst = reserve(count);
    if (dst == nullptr)
    {
        error.set("out of memory for '%s'", name);
        return false;
    }
    if (t->type == GgmlType::F32)
    {
        std::memcpy(dst, t->data, count * sizeof(float));
    }
    else if (t->type == GgmlType::F16)
    {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(t->data);
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = f16_to_f32(h[i]);
        }
    }
    else
    {
        error.set("tensor '%s' must be F32 or F16", name);
        return false;
    }
    out = dst;
    return true;
}

// the file holds n_out rows of n_in values (or, transposed, n_in rows of n_out); the matrix
// gets rows of n_in rounded up to the block, zero padded, and is packed when quantizing
bool VisionModel::matrix(const GgufFile& file, const char* name, size_t n_in, size_t n_out, bool transpose, Weight& out,
                         Error& error)
{
    const GgufTensor* t = file.find_tensor(name);
    if (t == nullptr)
    {
        error.set("tensor '%s' is missing", name);
        return false;
    }
    if (t->n_elements != n_in * n_out || (t->type != GgmlType::F32 && t->type != GgmlType::F16))
    {
        error.set("tensor '%s' is not an F32 or F16 matrix of %zu x %zu", name, n_in, n_out);
        return false;
    }
    const size_t padded = (n_in + Q8_0_BLOCK - 1) / Q8_0_BLOCK * Q8_0_BLOCK;
    float* dst = reserve(padded * n_out);
    if (dst == nullptr)
    {
        error.set("out of memory for '%s'", name);
        return false;
    }
    const float* f32 = reinterpret_cast<const float*>(t->data);
    const uint16_t* f16 = reinterpret_cast<const uint16_t*>(t->data);
    for (size_t o = 0; o < n_out; ++o)
    {
        for (size_t i = 0; i < n_in; ++i)
        {
            const size_t src = transpose ? i * n_out + o : o * n_in + i;
            dst[o * padded + i] = t->type == GgmlType::F32 ? f32[src] : f16_to_f32(f16[src]);
        }
        for (size_t i = n_in; i < padded; ++i)
        {
            dst[o * padded + i] = 0.0f;
        }
    }
    out = Weight();
    out.n_in = padded;
    out.n_out = n_out;
    out.type = GgmlType::F32;
    out.packed = false;
    out.data = reinterpret_cast<const uint8_t*>(dst);
    if (!m_quantize || !q8_0_packed_supported() || t->type == GgmlType::F32)
    {
        return true;
    }
    const size_t row_bytes = padded / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
    const size_t bytes = q8_0_packed_bytes(padded, n_out);
    Array<uint8_t> rows;
    if (m_packed_used + bytes > m_packed.size() || !rows.resize_uninitialized(row_bytes * n_out))
    {
        error.set("out of memory for the packed '%s'", name);
        return false;
    }
    for (size_t o = 0; o < n_out; ++o)
    {
        quantize_q8_0(dst + o * padded, padded, rows.data() + o * row_bytes);
    }
    pack_q8_0(rows.data(), padded, n_out, m_packed.data() + m_packed_used);
    out.type = GgmlType::Q8_0;
    out.packed = true;
    out.data = m_packed.data() + m_packed_used;
    m_packed_used += bytes;
    m_quantized = true;
    // the float32 copy is no longer needed
    m_used -= padded * n_out;
    return true;
}

bool VisionModel::load(const char* path, Error& error, bool quantize)
{
    GgufFile file;
    if (!file.open(path, error))
    {
        return false;
    }
    std::string_view projector;
    if (!file.get_string("clip.projector_type", projector) || projector != "gemma3")
    {
        error.set("'%s' is not a gemma3 vision projector (clip.projector_type)", path);
        return false;
    }
    VisionHparams& h = m_hparams;
    if (!key_u32(file, "clip.vision.image_size", h.image_size, error) ||
        !key_u32(file, "clip.vision.patch_size", h.patch_size, error) ||
        !key_u32(file, "clip.vision.embedding_length", h.n_embd, error) ||
        !key_u32(file, "clip.vision.feed_forward_length", h.n_ff, error) ||
        !key_u32(file, "clip.vision.block_count", h.n_layer, error) ||
        !key_u32(file, "clip.vision.attention.head_count", h.n_head, error) ||
        !key_u32(file, "clip.vision.projection_dim", h.n_proj, error) ||
        !key_u32(file, "clip.vision.projector.scale_factor", h.n_merge, error, false))
    {
        return false;
    }
    double value = 0.0;
    if (file.get_float("clip.vision.attention.layer_norm_epsilon", value))
    {
        h.eps = static_cast<float>(value);
    }
    const GgufArray* mean = file.get_array("clip.vision.image_mean");
    const GgufArray* std = file.get_array("clip.vision.image_std");
    if (mean != nullptr && std != nullptr && mean->count >= 3 && std->count >= 3 && mean->elem_type == GgufType::Float32 &&
        std->elem_type == GgufType::Float32)
    {
        Array<float> values;
        if (mean->copy_to(values))
        {
            std::memcpy(h.image_mean, values.data(), 3 * sizeof(float));
        }
        if (std->copy_to(values))
        {
            std::memcpy(h.image_std, values.data(), 3 * sizeof(float));
        }
    }
    if (h.image_size == 0 || h.patch_size == 0 || h.image_size % h.patch_size != 0 || h.n_embd == 0 || h.n_head == 0 ||
        h.n_embd % h.n_head != 0 || h.n_layer == 0 || h.n_merge == 0 || h.patches_per_side() % h.n_merge != 0 || h.n_proj == 0)
    {
        error.set("the vision hyperparameters of '%s' are not consistent", path);
        return false;
    }
    h.n_ff_padded = static_cast<uint32_t>((h.n_ff + Q8_0_BLOCK - 1) / Q8_0_BLOCK * Q8_0_BLOCK);
    m_quantize = quantize;
    m_quantized = false;

    size_t floats = 0;
    size_t packed = 0;
    for (size_t i = 0; i < file.tensors().size(); ++i)
    {
        const GgufTensor& t = file.tensors()[i];
        const size_t n_in = t.n_dims >= 2 ? static_cast<size_t>(t.dims[0]) : 1;
        const size_t rows = static_cast<size_t>(t.n_elements) / n_in;
        const size_t padded = (n_in + Q8_0_BLOCK - 1) / Q8_0_BLOCK * Q8_0_BLOCK;
        floats += padded * rows + padded * rows;
        if (t.n_dims == 2)
        {
            packed += q8_0_packed_bytes(padded, rows);
        }
    }
    if (!m_storage.resize_uninitialized(floats) || !m_packed.resize_uninitialized(packed))
    {
        error.set("out of memory for the vision model");
        return false;
    }
    m_used = 0;
    m_packed_used = 0;

    const size_t dim = h.n_embd;
    const size_t taps = 3 * static_cast<size_t>(h.patch_size) * h.patch_size;
    bool ok = matrix(file, "v.patch_embd.weight", taps, dim, false, m_patch, error) &&
              vector(file, "v.patch_embd.bias", dim, m_patch_b, error) &&
              vector(file, "v.position_embd.weight", static_cast<size_t>(h.n_patches()) * dim, m_position, error) &&
              vector(file, "v.post_ln.weight", dim, m_post_ln_w, error) && vector(file, "v.post_ln.bias", dim, m_post_ln_b, error) &&
              vector(file, "mm.soft_emb_norm.weight", dim, m_soft_emb_norm, error) &&
              matrix(file, "mm.input_projection.weight", dim, h.n_proj, true, m_projection, error);
    if (!ok)
    {
        return false;
    }
    if (!m_layers.resize_uninitialized(h.n_layer))
    {
        error.set("out of memory");
        return false;
    }
    char name[96];
    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        VisionLayer& layer = m_layers[l];
        const auto named = [&](const char* suffix)
        {
            std::snprintf(name, sizeof(name), "v.blk.%u.%s", l, suffix);
            return name;
        };
        ok = vector(file, named("ln1.weight"), dim, layer.ln1_w, error) && vector(file, named("ln1.bias"), dim, layer.ln1_b, error) &&
             matrix(file, named("attn_q.weight"), dim, dim, false, layer.q, error) &&
             vector(file, named("attn_q.bias"), dim, layer.q_b, error) &&
             matrix(file, named("attn_k.weight"), dim, dim, false, layer.k, error) &&
             vector(file, named("attn_k.bias"), dim, layer.k_b, error) &&
             matrix(file, named("attn_v.weight"), dim, dim, false, layer.v, error) &&
             vector(file, named("attn_v.bias"), dim, layer.v_b, error) &&
             matrix(file, named("attn_out.weight"), dim, dim, false, layer.out, error) &&
             vector(file, named("attn_out.bias"), dim, layer.out_b, error) &&
             vector(file, named("ln2.weight"), dim, layer.ln2_w, error) && vector(file, named("ln2.bias"), dim, layer.ln2_b, error) &&
             matrix(file, named("ffn_up.weight"), dim, h.n_ff, false, layer.up, error) &&
             vector(file, named("ffn_up.bias"), h.n_ff, layer.up_b, error) &&
             matrix(file, named("ffn_down.weight"), h.n_ff, dim, false, layer.down, error) &&
             vector(file, named("ffn_down.bias"), dim, layer.down_b, error);
        if (!ok)
        {
            return false;
        }
    }
    return true;
}

}
