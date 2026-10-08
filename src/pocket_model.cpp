#include "faisca/pocket_model.h"

#include "faisca/ops.h"
#include "faisca/unicode.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

bool key_u32(const GgufFile& file, const char* key, uint32_t& out, Error& error)
{
    uint64_t value = 0;
    if (!file.get_uint(key, value))
    {
        error.set("%s is missing", key);
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

bool key_f32(const GgufFile& file, const char* key, float& out, Error& error)
{
    double value = 0.0;
    if (!file.get_float(key, value))
    {
        error.set("%s is missing", key);
        return false;
    }
    out = static_cast<float>(value);
    return true;
}

bool key_bool(const GgufFile& file, const char* key, bool& out, Error& error)
{
    if (!file.get_bool(key, out))
    {
        error.set("%s is missing", key);
        return false;
    }
    return true;
}

}

PocketModel::~PocketModel() = default;

float* PocketModel::reserve(size_t count)
{
    if (m_used + count > m_storage.size())
    {
        return nullptr;
    }
    float* out = m_storage.data() + m_used;
    m_used += count;
    return out;
}

bool PocketModel::read_hparams(const GgufFile& file, Error& error)
{
    PocketHparams& h = m_hparams;
    bool ok = key_u32(file, "pocket.flow.dim", h.dim, error) && key_u32(file, "pocket.flow.n_layers", h.n_layers, error) &&
              key_u32(file, "pocket.flow.n_heads", h.n_heads, error) && key_u32(file, "pocket.flow.ffn_dim", h.ffn_dim, error) &&
              key_f32(file, "pocket.flow.max_period", h.max_period, error) && key_u32(file, "pocket.flow.ldim", h.ldim, error) &&
              key_u32(file, "pocket.flow.n_bins", h.n_bins, error) &&
              key_bool(file, "pocket.flow.bos_before_voice", h.bos_before_voice, error) &&
              key_u32(file, "pocket.head.dim", h.head_dim, error) && key_u32(file, "pocket.head.depth", h.head_depth, error) &&
              key_u32(file, "pocket.head.n_time_conds", h.n_time_conds, error) &&
              key_u32(file, "pocket.mimi.sample_rate", h.sample_rate, error) &&
              key_f32(file, "pocket.mimi.frame_rate", h.frame_rate, error) &&
              key_u32(file, "pocket.mimi.n_residual_layers", h.n_residual_layers, error) &&
              key_u32(file, "pocket.mimi.dilation_base", h.dilation_base, error) &&
              key_u32(file, "pocket.mimi.inner_dim", h.inner_dim, error) && key_u32(file, "pocket.mimi.outer_dim", h.outer_dim, error) &&
              key_u32(file, "pocket.mimi.tfm.dim", h.tfm_dim, error) && key_u32(file, "pocket.mimi.tfm.n_layers", h.tfm_n_layers, error) &&
              key_u32(file, "pocket.mimi.tfm.n_heads", h.tfm_n_heads, error) &&
              key_u32(file, "pocket.mimi.tfm.ffn_dim", h.tfm_ffn_dim, error) &&
              key_u32(file, "pocket.mimi.tfm.context", h.tfm_context, error) &&
              key_f32(file, "pocket.mimi.tfm.max_period", h.tfm_max_period, error) &&
              key_f32(file, "pocket.gen.temperature", h.temperature, error);
    if (!ok)
    {
        return false;
    }
    int64_t fae = -1;
    if (file.get_int("pocket.gen.frames_after_eos", fae))
    {
        h.frames_after_eos = static_cast<int32_t>(fae);
    }
    const GgufArray* ratios = file.get_array("pocket.mimi.ratios");
    if (ratios == nullptr || ratios->count == 0 || ratios->count > 4)
    {
        error.set("pocket.mimi.ratios is missing");
        return false;
    }
    h.n_ratios = static_cast<uint32_t>(ratios->count);
    Array<int32_t> r;
    Array<uint32_t> ru;
    if (ratios->elem_type == GgufType::Int32 && ratios->copy_to(r))
    {
        for (size_t i = 0; i < r.size(); ++i)
        {
            h.ratios[i] = static_cast<uint32_t>(r[i]);
        }
    }
    else if (ratios->elem_type == GgufType::UInt32 && ratios->copy_to(ru))
    {
        for (size_t i = 0; i < ru.size(); ++i)
        {
            h.ratios[i] = ru[i];
        }
    }
    else
    {
        error.set("pocket.mimi.ratios has an unexpected type");
        return false;
    }
    if (h.dim == 0 || h.n_layers == 0 || h.n_heads == 0 || h.dim % h.n_heads != 0 || h.ldim == 0 || h.head_depth == 0 ||
        h.n_time_conds > 2 || h.tfm_dim == 0 || h.tfm_n_heads == 0 || h.tfm_dim % h.tfm_n_heads != 0 ||
        h.n_residual_layers > POCKET_MAX_RES || h.frame_rate <= 0.0f || h.tfm_n_layers == 0)
    {
        error.set("unsupported Pocket TTS hparams");
        return false;
    }
    h.hop_length = 1;
    for (uint32_t i = 0; i < h.n_ratios; ++i)
    {
        h.hop_length *= h.ratios[i];
    }
    h.upsample_stride = static_cast<uint32_t>(static_cast<float>(h.sample_rate) / static_cast<float>(h.hop_length) / h.frame_rate + 0.5f);
    h.frame_size = h.hop_length * h.upsample_stride;
    return true;
}

bool PocketModel::read_rules(const GgufFile& file, Error& error)
{
    PocketTextRules& r = m_rules;
    if (!key_bool(file, "pocket.text.pad_with_spaces", r.pad_with_spaces, error) ||
        !key_bool(file, "pocket.text.remove_semicolons", r.remove_semicolons, error) ||
        !key_bool(file, "pocket.text.append_terminal_punctuation", r.append_terminal_punctuation, error) ||
        !key_bool(file, "pocket.text.capitalize_first_letter", r.capitalize_first_letter, error))
    {
        return false;
    }
    const GgufArray* from = file.get_array("pocket.text.replace_from");
    const GgufArray* to = file.get_array("pocket.text.replace_to");
    r.replace_from.clear();
    r.replace_to.clear();
    r.replace_to_offsets.clear();
    if (!r.replace_to_offsets.push_back(0))
    {
        error.set("out of memory");
        return false;
    }
    if (from == nullptr || to == nullptr)
    {
        return true;
    }
    Array<std::string_view> f;
    Array<std::string_view> t;
    if (!from->strings_to(f) || !to->strings_to(t) || f.size() != t.size())
    {
        error.set("pocket.text.replace_from and replace_to differ");
        return false;
    }
    for (size_t i = 0; i < f.size(); ++i)
    {
        size_t offset = 0;
        const uint32_t cp = utf8_decode(f[i].data(), f[i].size(), offset);
        if (offset != f[i].size())
        {
            error.set("pocket.text.replace_from entry %zu is not one character", i);
            return false;
        }
        if (!r.replace_from.push_back(cp))
        {
            error.set("out of memory");
            return false;
        }
        offset = 0;
        while (offset < t[i].size())
        {
            if (!r.replace_to.push_back(utf8_decode(t[i].data(), t[i].size(), offset)))
            {
                error.set("out of memory");
                return false;
            }
        }
        if (!r.replace_to_offsets.push_back(static_cast<uint32_t>(r.replace_to.size())))
        {
            error.set("out of memory");
            return false;
        }
    }
    return true;
}

bool PocketModel::vector(const GgufFile& file, const char* name, size_t count, const float*& out, Error& error, bool required)
{
    const GgufTensor* t = file.find_tensor(name);
    if (t == nullptr)
    {
        if (!required)
        {
            out = nullptr;
            return true;
        }
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
        for (size_t i = 0; i < count; ++i)
        {
            uint16_t bits = 0;
            std::memcpy(&bits, t->data + 2 * i, 2);
            dst[i] = f16_to_f32(bits);
        }
    }
    else if (t->type == GgmlType::Q8_0 && count % Q8_0_BLOCK == 0)
    {
        dequantize_q8_0(t->data, count, dst);
    }
    else
    {
        error.set("tensor '%s' has type %u", name, static_cast<unsigned>(t->type));
        return false;
    }
    out = dst;
    return true;
}

bool PocketModel::matrix(const GgufFile& file, const char* name, size_t n_in, size_t n_out, Weight& out, Error& error,
                         const float** rows)
{
    const GgufTensor* t = file.find_tensor(name);
    if (t == nullptr)
    {
        error.set("tensor '%s' is missing", name);
        return false;
    }
    if (t->n_dims != 2 || t->dims[0] != n_in || t->dims[1] != n_out)
    {
        error.set("tensor '%s' is %" PRIu64 " x %" PRIu64 ", expected %zu x %zu", name, t->dims[0], t->dims[1], n_in, n_out);
        return false;
    }
    out = Weight();
    out.n_in = n_in;
    out.n_out = n_out;
    const bool pack = q8_0_packed_supported() && n_in % Q8_0_BLOCK == 0;
    if (t->type == GgmlType::Q8_0 && pack && m_quantize)
    {
        const size_t bytes = q8_0_packed_bytes(n_in, n_out);
        if (m_packed_used + bytes > m_packed.size())
        {
            error.set("out of memory for '%s'", name);
            return false;
        }
        pack_q8_0(t->data, n_in, n_out, m_packed.data() + m_packed_used);
        out.type = GgmlType::Q8_0;
        out.packed = true;
        out.data = m_packed.data() + m_packed_used;
        m_packed_used += bytes;
        if (rows != nullptr)
        {
            const float* f = nullptr;
            if (!vector(file, name, n_in * n_out, f, error))
            {
                return false;
            }
            *rows = f;
        }
        return true;
    }
    const float* f = nullptr;
    if (!vector(file, name, n_in * n_out, f, error))
    {
        return false;
    }
    if (rows != nullptr)
    {
        *rows = f;
    }
    if (m_quantize && pack)
    {
        const size_t row_bytes = n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
        Array<uint8_t> q;
        if (!q.resize_uninitialized(n_out * row_bytes))
        {
            error.set("out of memory for '%s'", name);
            return false;
        }
        for (size_t r = 0; r < n_out; ++r)
        {
            quantize_q8_0(f + r * n_in, n_in, q.data() + r * row_bytes);
        }
        const size_t bytes = q8_0_packed_bytes(n_in, n_out);
        if (m_packed_used + bytes > m_packed.size())
        {
            error.set("out of memory for '%s'", name);
            return false;
        }
        pack_q8_0(q.data(), n_in, n_out, m_packed.data() + m_packed_used);
        out.type = GgmlType::Q8_0;
        out.packed = true;
        out.data = m_packed.data() + m_packed_used;
        m_packed_used += bytes;
        return true;
    }
    out.type = GgmlType::F32;
    out.packed = false;
    out.data = reinterpret_cast<const uint8_t*>(f);
    return true;
}

bool PocketModel::conv(const GgufFile& file, const char* prefix, uint32_t stride, uint32_t dilation, bool bias,
                       PocketConv& out, Error& error)
{
    char name[256];
    std::snprintf(name, sizeof(name), "%s.weight", prefix);
    const GgufTensor* t = file.find_tensor(name);
    if (t == nullptr)
    {
        error.set("tensor '%s' is missing", name);
        return false;
    }
    out = PocketConv();
    out.k = static_cast<uint32_t>(t->dims[0]);
    out.ic = static_cast<uint32_t>(t->n_dims > 1 ? t->dims[1] : 1);
    out.oc = static_cast<uint32_t>(t->n_dims > 2 ? t->dims[2] : 1);
    out.stride = stride;
    out.dilation = dilation;
    if (!vector(file, name, static_cast<size_t>(out.k) * out.ic * out.oc, out.w, error))
    {
        return false;
    }
    if (bias)
    {
        std::snprintf(name, sizeof(name), "%s.bias", prefix);
        if (!vector(file, name, out.oc, out.b, error))
        {
            return false;
        }
    }
    return true;
}

bool PocketModel::convtr(const GgufFile& file, const char* prefix, uint32_t stride, PocketConv& out, Error& error)
{
    char name[256];
    std::snprintf(name, sizeof(name), "%s.weight", prefix);
    const GgufTensor* t = file.find_tensor(name);
    if (t == nullptr || t->n_dims != 3)
    {
        error.set("tensor '%s' is missing or not 3-d", name);
        return false;
    }
    if (t->type != GgmlType::F32 && t->type != GgmlType::F16)
    {
        error.set("tensor '%s' must be F32 or F16", name);
        return false;
    }
    const size_t k = static_cast<size_t>(t->dims[0]);
    const size_t oc = static_cast<size_t>(t->dims[1]);
    const size_t ic = static_cast<size_t>(t->dims[2]);
    if (k != 2 * static_cast<size_t>(stride))
    {
        error.set("tensor '%s': a transposed conv kernel of %zu with stride %u is not supported", name, k, stride);
        return false;
    }
    float* dst = reserve(k * ic * oc);
    if (dst == nullptr)
    {
        error.set("out of memory for '%s'", name);
        return false;
    }
    const float* f32 = reinterpret_cast<const float*>(t->data);
    const uint16_t* f16 = reinterpret_cast<const uint16_t*>(t->data);
    for (size_t p = 0; p < stride; ++p)
    {
        for (size_t o = 0; o < oc; ++o)
        {
            for (size_t i = 0; i < ic; ++i)
            {
                const size_t src = (i * oc + o) * k;
                float* taps = dst + ((p * oc + o) * ic + i) * 2;
                if (t->type == GgmlType::F32)
                {
                    taps[0] = f32[src + p + stride];
                    taps[1] = f32[src + p];
                }
                else
                {
                    taps[0] = f16_to_f32(f16[src + p + stride]);
                    taps[1] = f16_to_f32(f16[src + p]);
                }
            }
        }
    }
    out = PocketConv();
    out.w = dst;
    out.k = 2;
    out.oc = static_cast<uint32_t>(oc);
    out.ic = static_cast<uint32_t>(ic);
    out.stride = stride;
    out.dilation = 1;
    std::snprintf(name, sizeof(name), "%s.bias", prefix);
    return vector(file, name, out.oc, out.b, error);
}

bool PocketModel::tfm_layer(const GgufFile& file, const char* prefix, bool layer_scale, PocketTfmLayer& out, Error& error)
{
    const bool mimi = std::strncmp(prefix, "mimi", 4) == 0;
    const uint32_t dim = mimi ? m_hparams.tfm_dim : m_hparams.dim;
    const uint32_t ffn = mimi ? m_hparams.tfm_ffn_dim : m_hparams.ffn_dim;
    char name[256];
    const auto named = [&](const char* suffix)
    {
        std::snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
        return name;
    };
    out = PocketTfmLayer();
    bool ok = vector(file, named("norm1.weight"), dim, out.norm1_w, error) && vector(file, named("norm1.bias"), dim, out.norm1_b, error) &&
              vector(file, named("norm2.weight"), dim, out.norm2_w, error) && vector(file, named("norm2.bias"), dim, out.norm2_b, error) &&
              matrix(file, named("self_attn.in_proj.weight"), dim, 3 * dim, out.in_proj, error) &&
              matrix(file, named("self_attn.out_proj.weight"), dim, dim, out.out_proj, error) &&
              matrix(file, named("linear1.weight"), dim, ffn, out.linear1, error) &&
              matrix(file, named("linear2.weight"), ffn, dim, out.linear2, error);
    if (ok && layer_scale)
    {
        ok = vector(file, named("layer_scale_1.scale"), dim, out.ls1, error) && vector(file, named("layer_scale_2.scale"), dim, out.ls2, error);
    }
    return ok;
}

bool PocketModel::res_unit(const GgufFile& file, const char* prefix, uint32_t dilation, PocketResUnit& out, Error& error)
{
    char name[160];
    std::snprintf(name, sizeof(name), "%s.block.1.conv", prefix);
    if (!conv(file, name, 1, dilation, true, out.c1, error))
    {
        return false;
    }
    std::snprintf(name, sizeof(name), "%s.block.3.conv", prefix);
    return conv(file, name, 1, 1, true, out.c2, error);
}

bool PocketModel::load(const char* path, Error& error, bool quantize)
{
    GgufFile file;
    if (!file.open(path, error))
    {
        return false;
    }
    std::string_view arch;
    if (!file.get_string("general.architecture", arch) || arch != "pocket-tts")
    {
        error.set("'%s' is not a Pocket TTS pack (general.architecture is not pocket-tts)", path);
        return false;
    }
    if (!read_hparams(file, error) || !read_rules(file, error) || !m_tokenizer.load(file, error))
    {
        return false;
    }
    m_quantize = quantize;

    // one float32 pool sized by every tensor, one packed pool sized by every matrix
    size_t floats = 0;
    size_t packed = 0;
    for (size_t i = 0; i < file.tensors().size(); ++i)
    {
        const GgufTensor& t = file.tensors()[i];
        floats += static_cast<size_t>(t.n_elements);
        if (t.n_dims == 2 && t.dims[0] % Q8_0_BLOCK == 0)
        {
            packed += q8_0_packed_bytes(static_cast<size_t>(t.dims[0]), static_cast<size_t>(t.dims[1]));
        }
    }
    if (!m_storage.resize_uninitialized(floats) || !m_packed.resize_uninitialized(packed))
    {
        error.set("out of memory for the model");
        return false;
    }
    m_used = 0;
    m_packed_used = 0;

    const PocketHparams& h = m_hparams;
    const size_t dim = h.dim;
    bool ok = matrix(file, "flow_lm.conditioner.embed.weight", dim, h.n_bins + 1, m_embed, error, &m_embed_rows) &&
              matrix(file, "flow_lm.input_linear.weight", h.ldim, dim, m_input_linear, error) &&
              vector(file, "flow_lm.bos_before_voice", dim, m_bos_before_voice, error, h.bos_before_voice) &&
              vector(file, "flow_lm.bos_emb", h.ldim, m_bos_emb, error) &&
              matrix(file, "flow_lm.speaker_proj_weight", h.inner_dim, dim, m_speaker_proj, error) &&
              vector(file, "flow_lm.emb_mean", h.ldim, m_emb_mean, error) && vector(file, "flow_lm.emb_std", h.ldim, m_emb_std, error) &&
              vector(file, "flow_lm.out_norm.weight", dim, m_out_norm_w, error) &&
              vector(file, "flow_lm.out_norm.bias", dim, m_out_norm_b, error) &&
              vector(file, "flow_lm.out_eos.weight", dim, m_out_eos_w, error);
    if (!ok)
    {
        return false;
    }
    {
        const float* b = nullptr;
        if (!vector(file, "flow_lm.out_eos.bias", 1, b, error))
        {
            return false;
        }
        m_out_eos_b = b[0];
    }
    if (!m_layers.resize_uninitialized(h.n_layers))
    {
        error.set("out of memory");
        return false;
    }
    char prefix[128];
    for (uint32_t i = 0; i < h.n_layers; ++i)
    {
        std::snprintf(prefix, sizeof(prefix), "flow_lm.transformer.layers.%u", i);
        if (!tfm_layer(file, prefix, false, m_layers[i], error))
        {
            return false;
        }
    }

    const size_t hd = h.head_dim;
    ok = matrix(file, "flow_lm.flow_net.cond_embed.weight", dim, hd, m_cond, error) &&
         vector(file, "flow_lm.flow_net.cond_embed.bias", hd, m_cond_b, error) &&
         matrix(file, "flow_lm.flow_net.input_proj.weight", h.ldim, hd, m_in, error) &&
         vector(file, "flow_lm.flow_net.input_proj.bias", hd, m_in_b, error) &&
         matrix(file, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight", hd, 2 * hd, m_final_ada, error) &&
         vector(file, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias", 2 * hd, m_final_ada_b, error) &&
         matrix(file, "flow_lm.flow_net.final_layer.linear.weight", hd, h.ldim, m_final, error) &&
         vector(file, "flow_lm.flow_net.final_layer.linear.bias", h.ldim, m_final_b, error);
    if (!ok)
    {
        return false;
    }
    for (uint32_t i = 0; i < h.n_time_conds; ++i)
    {
        PocketTimeEmbed& te = m_temb[i];
        te = PocketTimeEmbed();
        char name[160];
        const auto named = [&](const char* suffix)
        {
            std::snprintf(name, sizeof(name), "flow_lm.flow_net.time_embed.%u.%s", i, suffix);
            return name;
        };
        const GgufTensor* freqs = file.find_tensor(named("freqs"));
        if (freqs == nullptr)
        {
            error.set("tensor '%s' is missing", name);
            return false;
        }
        const size_t n_freqs = static_cast<size_t>(freqs->n_elements);
        ok = vector(file, named("freqs"), n_freqs, te.freqs, error) &&
             matrix(file, named("mlp.0.weight"), 2 * n_freqs, hd, te.l0, error) &&
             vector(file, named("mlp.0.bias"), hd, te.l0_b, error) &&
             matrix(file, named("mlp.2.weight"), hd, hd, te.l2, error) &&
             vector(file, named("mlp.2.bias"), hd, te.l2_b, error) &&
             vector(file, named("mlp.3.alpha"), hd, te.alpha, error);
        if (!ok)
        {
            return false;
        }
    }
    if (!m_blocks.resize_uninitialized(h.head_depth))
    {
        error.set("out of memory");
        return false;
    }
    for (uint32_t i = 0; i < h.head_depth; ++i)
    {
        PocketResBlock& b = m_blocks[i];
        b = PocketResBlock();
        char name[160];
        const auto named = [&](const char* suffix)
        {
            std::snprintf(name, sizeof(name), "flow_lm.flow_net.res_blocks.%u.%s", i, suffix);
            return name;
        };
        ok = vector(file, named("in_ln.weight"), hd, b.ln_w, error) && vector(file, named("in_ln.bias"), hd, b.ln_b, error) &&
             matrix(file, named("mlp.0.weight"), hd, hd, b.mlp0, error) && vector(file, named("mlp.0.bias"), hd, b.mlp0_b, error) &&
             matrix(file, named("mlp.2.weight"), hd, hd, b.mlp2, error) && vector(file, named("mlp.2.bias"), hd, b.mlp2_b, error) &&
             matrix(file, named("adaLN_modulation.1.weight"), hd, 3 * hd, b.ada, error) &&
             vector(file, named("adaLN_modulation.1.bias"), 3 * hd, b.ada_b, error);
        if (!ok)
        {
            return false;
        }
    }

    // Mimi decoder
    if (!matrix(file, "mimi.quantizer.output_proj.weight", h.ldim, h.outer_dim, m_quant_proj, error))
    {
        return false;
    }
    {
        const GgufTensor* up = file.find_tensor("mimi.upsample.convtr.convtr.weight");
        if (up == nullptr || up->n_dims != 3 || up->dims[2] != h.outer_dim)
        {
            error.set("mimi.upsample.convtr.convtr.weight is missing or has an unexpected shape");
            return false;
        }
        m_upsample_k = static_cast<uint32_t>(up->dims[0]);
        if (!vector(file, "mimi.upsample.convtr.convtr.weight", static_cast<size_t>(m_upsample_k) * h.outer_dim, m_upsample_w, error))
        {
            return false;
        }
    }
    if (!m_dec_layers.resize_uninitialized(h.tfm_n_layers) || !m_enc_layers.resize_uninitialized(h.tfm_n_layers) ||
        !m_dec_stages.resize_uninitialized(h.n_ratios) || !m_enc_stages.resize_uninitialized(h.n_ratios))
    {
        error.set("out of memory");
        return false;
    }
    for (uint32_t i = 0; i < h.tfm_n_layers; ++i)
    {
        std::snprintf(prefix, sizeof(prefix), "mimi.decoder_transformer.transformer.layers.%u", i);
        if (!tfm_layer(file, prefix, true, m_dec_layers[i], error))
        {
            return false;
        }
        std::snprintf(prefix, sizeof(prefix), "mimi.encoder_transformer.transformer.layers.%u", i);
        if (!tfm_layer(file, prefix, true, m_enc_layers[i], error))
        {
            return false;
        }
    }
    uint32_t idx = 0;
    std::snprintf(prefix, sizeof(prefix), "mimi.decoder.model.%u.conv", idx++);
    if (!conv(file, prefix, 1, 1, true, m_dec_in, error))
    {
        return false;
    }
    for (uint32_t s = 0; s < h.n_ratios; ++s)
    {
        PocketStage& st = m_dec_stages[s];
        st = PocketStage();
        idx++; // ELU
        std::snprintf(prefix, sizeof(prefix), "mimi.decoder.model.%u.convtr", idx++);
        if (!convtr(file, prefix, h.ratios[s], st.resample, error))
        {
            return false;
        }
        for (uint32_t j = 0; j < h.n_residual_layers; ++j)
        {
            uint32_t d = 1;
            for (uint32_t e = 0; e < j; ++e)
            {
                d *= h.dilation_base;
            }
            std::snprintf(prefix, sizeof(prefix), "mimi.decoder.model.%u", idx++);
            if (!res_unit(file, prefix, d, st.res[j], error))
            {
                return false;
            }
            st.n_res++;
        }
    }
    idx++; // ELU
    std::snprintf(prefix, sizeof(prefix), "mimi.decoder.model.%u.conv", idx);
    if (!conv(file, prefix, 1, 1, true, m_dec_out, error))
    {
        return false;
    }

    // Mimi encoder
    idx = 0;
    std::snprintf(prefix, sizeof(prefix), "mimi.encoder.model.%u.conv", idx++);
    if (!conv(file, prefix, 1, 1, true, m_enc_in, error))
    {
        return false;
    }
    for (uint32_t s = 0; s < h.n_ratios; ++s)
    {
        PocketStage& st = m_enc_stages[s];
        st = PocketStage();
        for (uint32_t j = 0; j < h.n_residual_layers; ++j)
        {
            uint32_t d = 1;
            for (uint32_t e = 0; e < j; ++e)
            {
                d *= h.dilation_base;
            }
            std::snprintf(prefix, sizeof(prefix), "mimi.encoder.model.%u", idx++);
            if (!res_unit(file, prefix, d, st.res[j], error))
            {
                return false;
            }
            st.n_res++;
        }
        idx++; // ELU
        std::snprintf(prefix, sizeof(prefix), "mimi.encoder.model.%u.conv", idx++);
        if (!conv(file, prefix, h.ratios[h.n_ratios - 1 - s], 1, true, st.resample, error))
        {
            return false;
        }
    }
    idx++; // ELU
    std::snprintf(prefix, sizeof(prefix), "mimi.encoder.model.%u.conv", idx);
    if (!conv(file, prefix, 1, 1, true, m_enc_out, error) ||
        !conv(file, "mimi.downsample.conv.conv", h.upsample_stride, 1, false, m_downsample, error))
    {
        return false;
    }
    return true;
}

}
