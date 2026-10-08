#include "faisca/vulkan_runner.h"

#include "faisca/ops.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace faisca
{

VulkanRunner::~VulkanRunner()
{
    release();
}

void VulkanRunner::destroy_matrix(Matrix& matrix)
{
    m_device->destroy_buffer(matrix.scales);
    m_device->destroy_buffer(matrix.quants);
    m_device->destroy_buffer(matrix.f32);
    matrix = Matrix();
}

void VulkanRunner::release()
{
    if (m_device != nullptr)
    {
        for (size_t l = 0; l < m_layers.size(); ++l)
        {
            Layer& layer = m_layers[l];
            m_device->destroy_buffer(layer.attn_norm);
            m_device->destroy_buffer(layer.ffn_norm);
            destroy_matrix(layer.wq);
            destroy_matrix(layer.wk);
            destroy_matrix(layer.wv);
            destroy_matrix(layer.wo);
            destroy_matrix(layer.ffn_gate);
            destroy_matrix(layer.ffn_up);
            destroy_matrix(layer.ffn_down);
        }
        destroy_matrix(m_token_embd);
        if (!m_output_tied)
        {
            destroy_matrix(m_output);
        }
        m_output = Matrix();
        VulkanBuffer* buffers[] = {&m_output_norm, &m_rope_factors, &m_tokens, &m_x, &m_h, &m_q, &m_k, &m_v, &m_attn, &m_gate, &m_up,
                                   &m_act, &m_logits, &m_k_cache, &m_v_cache, &m_scores};
        for (VulkanBuffer* buffer : buffers)
        {
            m_device->destroy_buffer(*buffer);
        }
        m_kernels.release();
    }
    m_layers.clear();
    m_device = nullptr;
    m_n_ctx = 0;
    m_pos = 0;
    m_weight_bytes = 0;
}

bool VulkanRunner::upload_vector(const GgufTensor* tensor, VulkanBuffer& out, Error& error)
{
    const size_t count = static_cast<size_t>(tensor->n_elements);
    Array<float> values;
    if (!values.resize_uninitialized(count))
    {
        error.set("out of memory");
        return false;
    }
    if (tensor->type == GgmlType::F32)
    {
        std::memcpy(values.data(), tensor->data, count * sizeof(float));
    }
    else if (tensor->type == GgmlType::F16)
    {
        for (size_t i = 0; i < count; ++i)
        {
            uint16_t bits = 0;
            std::memcpy(&bits, tensor->data + 2 * i, 2);
            values[i] = f16_to_f32(bits);
        }
    }
    else
    {
        error.set("tensor %.*s is %s, a norm must be F32 or F16", static_cast<int>(tensor->name.size()), tensor->name.data(),
                  ggml_type_info(tensor->type)->name);
        return false;
    }
    m_weight_bytes += count * sizeof(float);
    return m_device->create_buffer(count * sizeof(float), out, error) &&
           m_device->upload(out, values.data(), count * sizeof(float), error);
}

bool VulkanRunner::upload_matrix(const GgufTensor* tensor, Matrix& out, Error& error, bool quantized)
{
    out.n_in = static_cast<uint32_t>(tensor->dims[0]);
    out.n_out = static_cast<uint32_t>(tensor->dims[1]);
    const size_t n_in = out.n_in;
    const size_t n_out = out.n_out;
    const bool split = tensor->type == GgmlType::Q8_0 || tensor->type == GgmlType::Q4_0 || tensor->type == GgmlType::Q5_0;
    const bool k_quant = tensor->type == GgmlType::Q4_K || tensor->type == GgmlType::Q6_K;
    if ((split || k_quant) && (quantized || tensor->type == GgmlType::Q8_0))
    {
        Array<uint32_t> scales;
        Array<uint32_t> quants;
        size_t quant_words = 0;
        switch (tensor->type)
        {
            case GgmlType::Q8_0:
                quant_words = q8_0_split_quant_words(n_in, n_out);
                break;
            case GgmlType::Q4_0:
                quant_words = q4_0_split_quant_words(n_in, n_out);
                break;
            case GgmlType::Q5_0:
                quant_words = q5_0_split_quant_words(n_in, n_out);
                break;
            case GgmlType::Q4_K:
                quant_words = q4_k_device_words(n_in, n_out);
                break;
            default:
                quant_words = q6_k_device_words(n_in, n_out);
                break;
        }
        if (!quants.resize_uninitialized(quant_words) ||
            (split && !scales.resize_uninitialized(q8_0_split_scale_words(n_in, n_out))))
        {
            error.set("out of memory");
            return false;
        }
        switch (tensor->type)
        {
            case GgmlType::Q8_0:
                split_q8_0(tensor->data, n_in, n_out, scales.data(), quants.data());
                break;
            case GgmlType::Q4_0:
                split_q4_0(tensor->data, n_in, n_out, scales.data(), quants.data());
                break;
            case GgmlType::Q5_0:
                split_q5_0(tensor->data, n_in, n_out, scales.data(), quants.data());
                break;
            case GgmlType::Q4_K:
                pack_q4_k_device(tensor->data, n_in, n_out, quants.data());
                break;
            default:
                pack_q6_k_device(tensor->data, n_in, n_out, quants.data());
                break;
        }
        out.type = tensor->type;
        m_weight_bytes += (scales.size() + quants.size()) * 4;
        if (split && (!m_device->create_buffer(scales.size() * 4, out.scales, error) ||
                      !m_device->upload(out.scales, scales.data(), scales.size() * 4, error)))
        {
            return false;
        }
        return m_device->create_buffer(quants.size() * 4, out.quants, error) &&
               m_device->upload(out.quants, quants.data(), quants.size() * 4, error);
    }

    Array<float> values;
    if (!values.resize_uninitialized(n_in * n_out))
    {
        error.set("out of memory");
        return false;
    }
    Weight weight;
    weight.type = tensor->type;
    weight.data = tensor->data;
    weight.n_in = n_in;
    weight.n_out = n_out;
    switch (tensor->type)
    {
        case GgmlType::F32:
            std::memcpy(values.data(), tensor->data, n_in * n_out * sizeof(float));
            break;
        case GgmlType::Q4_0:
        case GgmlType::Q5_0:
        case GgmlType::Q4_K:
        case GgmlType::Q6_K:
            for (size_t o = 0; o < n_out; ++o)
            {
                weight.row(o, values.data() + o * n_in);
            }
            break;
        case GgmlType::F16:
            for (size_t i = 0; i < n_in * n_out; ++i)
            {
                uint16_t bits = 0;
                std::memcpy(&bits, tensor->data + 2 * i, 2);
                values[i] = f16_to_f32(bits);
            }
            break;
        default:
            error.set("tensor %.*s is %s, the Vulkan runner reads F32, F16, Q8_0, Q4_0, Q5_0, Q4_K and Q6_K",
                      static_cast<int>(tensor->name.size()), tensor->name.data(), ggml_type_info(tensor->type)->name);
            return false;
    }
    out.type = GgmlType::F32;
    m_weight_bytes += values.size() * 4;
    return m_device->create_buffer(values.size() * 4, out.f32, error) &&
           m_device->upload(out.f32, values.data(), values.size() * 4, error);
}

bool VulkanRunner::init(VulkanDevice& device, const GgufFile& file, uint32_t n_ctx, Error& error)
{
    release();
    if (!device.ready())
    {
        error.set("the Vulkan device is not initialized");
        return false;
    }
    if (n_ctx == 0)
    {
        error.set("the context size must be at least 1");
        return false;
    }
    m_device = &device;
    if (!LlamaModel::read_hparams(file, m_hparams, error) || !m_kernels.init(device, error))
    {
        release();
        return false;
    }
    const LlamaHparams& h = m_hparams;
    m_n_ctx = n_ctx;

    const GgufTensor* token_embd = file.find_tensor("token_embd.weight");
    const GgufTensor* output = file.find_tensor("output.weight");
    const GgufTensor* output_norm = file.find_tensor("output_norm.weight");
    if (token_embd == nullptr || output_norm == nullptr || token_embd->n_dims != 2 ||
        token_embd->dims[0] != h.n_embd || token_embd->dims[1] != h.n_vocab)
    {
        error.set("token_embd.weight or output_norm.weight missing or of the wrong shape");
        release();
        return false;
    }
    // the embedding gather reads Q8_0 or float32 only
    if (!upload_matrix(token_embd, m_token_embd, error, false) || !upload_vector(output_norm, m_output_norm, error))
    {
        release();
        return false;
    }
    const GgufTensor* rope_freqs = file.find_tensor("rope_freqs.weight");
    m_has_rope_factors = rope_freqs != nullptr;
    if (rope_freqs != nullptr)
    {
        if (rope_freqs->n_dims != 1 || rope_freqs->dims[0] != h.n_rot / 2 || !upload_vector(rope_freqs, m_rope_factors, error))
        {
            error.set("rope_freqs.weight must hold %" PRIu32 " float values", h.n_rot / 2);
            release();
            return false;
        }
    }
    else if (!device.create_buffer(sizeof(float), m_rope_factors, error))
    {
        release();
        return false;
    }
    m_output_tied = output == nullptr;
    if (m_output_tied)
    {
        m_output = m_token_embd;
    }
    else if (!upload_matrix(output, m_output, error))
    {
        release();
        return false;
    }

    if (!m_layers.resize_uninitialized(h.n_layer))
    {
        error.set("out of memory");
        release();
        return false;
    }
    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        m_layers[l] = Layer();
    }
    char name[128];
    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        Layer& layer = m_layers[l];
        struct Item
        {
            const char* suffix;
            Matrix* matrix;
            VulkanBuffer* vector;
        };
        const Item items[] = {
            {"attn_norm", nullptr, &layer.attn_norm}, {"attn_q", &layer.wq, nullptr}, {"attn_k", &layer.wk, nullptr},
            {"attn_v", &layer.wv, nullptr}, {"attn_output", &layer.wo, nullptr}, {"ffn_norm", nullptr, &layer.ffn_norm},
            {"ffn_gate", &layer.ffn_gate, nullptr}, {"ffn_up", &layer.ffn_up, nullptr}, {"ffn_down", &layer.ffn_down, nullptr},
        };
        for (const Item& item : items)
        {
            std::snprintf(name, sizeof(name), "blk.%" PRIu32 ".%s.weight", l, item.suffix);
            const GgufTensor* tensor = file.find_tensor(name);
            if (tensor == nullptr)
            {
                error.set("tensor %s is missing", name);
                release();
                return false;
            }
            const bool ok = item.matrix != nullptr ? upload_matrix(tensor, *item.matrix, error)
                                                   : upload_vector(tensor, *item.vector, error);
            if (!ok)
            {
                release();
                return false;
            }
        }
    }

    const size_t n_kv = static_cast<size_t>(h.n_head_kv) * h.head_dim;
    const size_t n_q = static_cast<size_t>(h.n_head) * h.head_dim;
    const size_t cache = static_cast<size_t>(h.n_layer) * n_ctx * n_kv * sizeof(float);
    const bool ok = device.create_host_buffer(BATCH * sizeof(uint32_t), m_tokens, error) &&
                    device.create_host_buffer(BATCH * h.n_embd * sizeof(float), m_x, error) &&
                    device.create_host_buffer(BATCH * h.n_embd * sizeof(float), m_h, error) &&
                    device.create_host_buffer(BATCH * n_q * sizeof(float), m_q, error) &&
                    device.create_host_buffer(BATCH * n_kv * sizeof(float), m_k, error) &&
                    device.create_host_buffer(BATCH * n_kv * sizeof(float), m_v, error) &&
                    device.create_host_buffer(BATCH * n_q * sizeof(float), m_attn, error) &&
                    device.create_host_buffer(BATCH * h.n_ff * sizeof(float), m_gate, error) &&
                    device.create_host_buffer(BATCH * h.n_ff * sizeof(float), m_up, error) &&
                    device.create_host_buffer(BATCH * h.n_ff * sizeof(float), m_act, error) &&
                    device.create_host_buffer(BATCH * h.n_vocab * sizeof(float), m_logits, error) &&
                    device.create_buffer(cache, m_k_cache, error) && device.create_buffer(cache, m_v_cache, error) &&
                    device.create_buffer(BATCH * n_q / h.head_dim * static_cast<size_t>(n_ctx) * sizeof(float), m_scores, error);
    if (!ok)
    {
        release();
        return false;
    }
    m_pos = 0;
    return true;
}

bool VulkanRunner::matvec(const Matrix& w, const VulkanBuffer& x, VulkanBuffer& y, uint32_t n_tokens, Error& error)
{
    switch (w.type)
    {
        case GgmlType::Q8_0:
            return m_kernels.matvec_q8_0(w.scales, w.quants, x, y, w.n_in, w.n_out, n_tokens, w.n_in, w.n_out, error);
        case GgmlType::Q4_0:
            return m_kernels.matvec_q4_0(w.scales, w.quants, x, y, w.n_in, w.n_out, n_tokens, w.n_in, w.n_out, error);
        case GgmlType::Q5_0:
            return m_kernels.matvec_q5_0(w.scales, w.quants, x, y, w.n_in, w.n_out, n_tokens, w.n_in, w.n_out, error);
        case GgmlType::Q4_K:
            return m_kernels.matvec_q4_k(w.quants, x, y, w.n_in, w.n_out, n_tokens, w.n_in, w.n_out, error);
        case GgmlType::Q6_K:
            return m_kernels.matvec_q6_k(w.quants, x, y, w.n_in, w.n_out, n_tokens, w.n_in, w.n_out, error);
        default:
            return m_kernels.matvec_f32(w.f32, x, y, w.n_in, w.n_out, n_tokens, w.n_in, w.n_out, error);
    }
}

bool VulkanRunner::trace(const char* name, int layer, const VulkanBuffer& buffer, size_t count, size_t n_tokens,
                         Error& error)
{
    if (m_trace == nullptr)
    {
        return true;
    }
    // the dispatches so far must finish before the values are read, then recording resumes
    if (!m_device->submit(error) || !m_staging.resize_uninitialized(count * n_tokens) ||
        !m_device->download(buffer, m_staging.data(), count * n_tokens * sizeof(float), error))
    {
        return false;
    }
    for (size_t t = 0; t < n_tokens; ++t)
    {
        m_trace(m_trace_user, name, layer, m_staging.data() + t * count, count);
    }
    return m_device->begin(error);
}

bool VulkanRunner::forward_batch(size_t count, uint32_t pos, float* logits, bool all_logits, Error& error)
{
    const LlamaHparams& h = m_hparams;
    const uint32_t n = static_cast<uint32_t>(count);
    const uint32_t n_kv = h.n_head_kv * h.head_dim;
    const uint32_t n_q = h.n_head * h.head_dim;
    VulkanKernels& k = m_kernels;

    if (!m_device->begin(error))
    {
        return false;
    }
    bool ok = m_token_embd.type == GgmlType::Q8_0
                  ? k.embed_q8_0(m_token_embd.scales, m_token_embd.quants, m_tokens, m_x, h.n_embd, n, error)
                  : k.embed_f32(m_token_embd.f32, m_tokens, m_x, h.n_embd, n, error);
    ok = ok && trace("inp_embd", -1, m_x, h.n_embd, count, error);

    for (uint32_t l = 0; ok && l < h.n_layer; ++l)
    {
        const Layer& layer = m_layers[l];
        const int il = static_cast<int>(l);
        ok = k.rms_norm(m_x, layer.attn_norm, m_h, h.n_embd, n, h.rms_eps, error) &&
             trace("attn_norm", il, m_h, h.n_embd, count, error) &&
             matvec(layer.wq, m_h, m_q, n, error) && matvec(layer.wk, m_h, m_k, n, error) &&
             matvec(layer.wv, m_h, m_v, n, error) &&
             k.rope(m_q, m_rope_factors, m_has_rope_factors, h.n_head, h.head_dim, h.n_rot, pos, h.rope_freq_base, n, error) &&
             k.rope(m_k, m_rope_factors, m_has_rope_factors, h.n_head_kv, h.head_dim, h.n_rot, pos, h.rope_freq_base, n, error) &&
             k.kv_store(m_k, m_v, m_k_cache, m_v_cache, h.n_head_kv, h.head_dim, m_n_ctx, pos, l, n, error) &&
             trace("Qcur", il, m_q, n_q, count, error) && trace("Kcur", il, m_k, n_kv, count, error) &&
             trace("Vcur", il, m_v, n_kv, count, error) &&
             k.attention(m_q, m_k_cache, m_v_cache, m_scores, m_attn, h.n_head, h.n_head_kv, h.head_dim, m_n_ctx, pos, l, n,
                         error) &&
             matvec(layer.wo, m_attn, m_h, n, error) && trace("attn_out", il, m_h, h.n_embd, count, error) &&
             k.add(m_x, m_h, n * h.n_embd, error) && trace("ffn_inp", il, m_x, h.n_embd, count, error) &&
             k.rms_norm(m_x, layer.ffn_norm, m_h, h.n_embd, n, h.rms_eps, error) &&
             trace("ffn_norm", il, m_h, h.n_embd, count, error) &&
             matvec(layer.ffn_gate, m_h, m_gate, n, error) && matvec(layer.ffn_up, m_h, m_up, n, error) &&
             trace("ffn_gate", il, m_gate, h.n_ff, count, error) && trace("ffn_up", il, m_up, h.n_ff, count, error) &&
             k.swiglu(m_gate, m_up, m_act, n * h.n_ff, error) && trace("ffn_swiglu", il, m_act, h.n_ff, count, error) &&
             matvec(layer.ffn_down, m_act, m_h, n, error) && trace("ffn_out", il, m_h, h.n_embd, count, error) &&
             k.add(m_x, m_h, n * h.n_embd, error) && trace("l_out", il, m_x, h.n_embd, count, error);
    }
    if (!ok)
    {
        return false;
    }

    const bool all = all_logits || m_trace != nullptr;
    if (all)
    {
        ok = k.rms_norm(m_x, m_output_norm, m_h, h.n_embd, n, h.rms_eps, error) &&
             trace("result_norm", -1, m_h, h.n_embd, count, error) && matvec(m_output, m_h, m_logits, n, error) &&
             trace("result_output", -1, m_logits, h.n_vocab, count, error);
    }
    else
    {
        ok = k.rms_norm(m_x, m_output_norm, m_h, h.n_embd, n, h.rms_eps, error) && matvec(m_output, m_h, m_logits, n, error);
    }
    if (!ok || !m_device->submit(error))
    {
        return false;
    }
    if (all_logits)
    {
        return m_device->download(m_logits, logits, count * h.n_vocab * sizeof(float), error);
    }
    if (!m_staging.resize_uninitialized(count * h.n_vocab) ||
        !m_device->download(m_logits, m_staging.data(), count * h.n_vocab * sizeof(float), error))
    {
        return false;
    }
    std::memcpy(logits, m_staging.data() + (count - 1) * h.n_vocab, h.n_vocab * sizeof(float));
    return true;
}

bool VulkanRunner::forward(const int32_t* tokens, size_t count, Array<float>& logits, Error& error, bool all_logits)
{
    if (m_device == nullptr)
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
    const LlamaHparams& h = m_hparams;
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
    for (size_t done = 0; done < count;)
    {
        const size_t batch = count - done < BATCH ? count - done : BATCH;
        uint32_t ids[BATCH];
        for (size_t i = 0; i < batch; ++i)
        {
            ids[i] = static_cast<uint32_t>(tokens[done + i]);
        }
        if (!m_device->upload(m_tokens, ids, batch * sizeof(uint32_t), error) ||
            !forward_batch(batch, m_pos, logits.data() + (all_logits ? done * h.n_vocab : 0), all_logits, error))
        {
            return false;
        }
        m_pos += static_cast<uint32_t>(batch);
        done += batch;
    }
    return true;
}

}
