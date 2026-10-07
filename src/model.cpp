#include "faisca/model.h"

#include "faisca/memory.h"
#include "faisca/ops.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

bool get_u32(const GgufFile& file, const char* key, uint32_t& out, bool required, Error& error)
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
    if (value == 0 || value > 0xffffffffu)
    {
        error.set("%s has an invalid value %" PRIu64, key, value);
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

}

size_t Weight::row_bytes() const
{
    switch (type)
    {
        case GgmlType::Q8_0:
            return n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
        case GgmlType::Q4_0:
            return n_in / Q4_0_BLOCK * Q4_0_BLOCK_BYTES;
        default:
            return n_in * sizeof(float);
    }
}

void Weight::row(size_t index, float* out) const
{
    if (packed)
    {
        if (type == GgmlType::Q4_0)
        {
            row_q4_0_packed(data, n_in, index, out);
        }
        else
        {
            row_q8_0_packed(data, n_in, index, out);
        }
        return;
    }
    const uint8_t* source = data + index * row_bytes();
    if (type == GgmlType::Q8_0)
    {
        dequantize_q8_0(source, n_in, out);
        return;
    }
    if (type == GgmlType::Q4_0)
    {
        dequantize_q4_0(source, n_in, out);
        return;
    }
    std::memcpy(out, source, n_in * sizeof(float));
}

void Weight::matvec(const float* x, const uint8_t* xq, const int32_t* x_sums, size_t first, size_t last,
                    float* out) const
{
    if (first >= last)
    {
        return;
    }
    if (packed)
    {
        if (type == GgmlType::Q4_0)
        {
            matmul_q4_0_packed(data, xq, 0, x_sums, 0, 1, n_in, n_out, first, last, out, 0);
        }
        else
        {
            matvec_q8_0_packed(data, xq, x_sums, n_in, n_out, first, last, out);
        }
        return;
    }
    if (type == GgmlType::Q8_0)
    {
        matvec_q8_0(data + first * row_bytes(), xq, n_in, last - first, out + first);
        return;
    }
    faisca::matvec(f32() + first * n_in, x, n_in, last - first, out + first);
}

void Weight::matmul(const float* x, size_t x_stride, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                    size_t sums_stride, size_t n_tokens, size_t first, size_t last, float* out, size_t out_stride) const
{
    if (first >= last)
    {
        return;
    }
    if (packed)
    {
        if (type == GgmlType::Q4_0)
        {
            matmul_q4_0_packed(data, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in, n_out, first, last, out, out_stride);
        }
        else
        {
            matmul_q8_0_packed(data, xq, xq_stride, x_sums, sums_stride, n_tokens, n_in, n_out, first, last, out, out_stride);
        }
        return;
    }
    for (size_t t = 0; t < n_tokens; ++t)
    {
        matvec(x + t * x_stride, xq + t * xq_stride, x_sums + t * sums_stride, first, last, out + t * out_stride);
    }
}

void LlamaModel::release()
{
    free_large(m_packed, m_packed_bytes);
    m_packed = nullptr;
    m_packed_bytes = 0;
    m_packed_used = 0;
    m_pack = false;
    m_storage.release();
    m_layers.release();
    m_token_embd = Weight();
    m_output_norm = nullptr;
    m_output = Weight();
    m_hparams = LlamaHparams();
    m_dequantize = false;
}

size_t LlamaModel::weight_bytes() const
{
    size_t bytes = m_output.row_bytes() * m_output.n_out;
    for (const LlamaLayer& layer : m_layers)
    {
        for (const Weight* w : {&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.ffn_gate, &layer.ffn_up, &layer.ffn_down})
        {
            bytes += w->row_bytes() * w->n_out;
        }
        bytes += 2 * m_hparams.n_embd * sizeof(float);
    }
    return bytes;
}

bool LlamaModel::load_hparams(const GgufFile& file, Error& error)
{
    std::string_view architecture;
    if (!file.get_string("general.architecture", architecture))
    {
        error.set("general.architecture is missing");
        return false;
    }
    if (architecture != "llama")
    {
        error.set("architecture '%.*s' is not supported, only 'llama'", static_cast<int>(architecture.size()),
                  architecture.data());
        return false;
    }

    LlamaHparams& h = m_hparams;
    if (!get_u32(file, "llama.embedding_length", h.n_embd, true, error) ||
        !get_u32(file, "llama.block_count", h.n_layer, true, error) ||
        !get_u32(file, "llama.feed_forward_length", h.n_ff, true, error) ||
        !get_u32(file, "llama.attention.head_count", h.n_head, true, error) ||
        !get_u32(file, "llama.context_length", h.n_ctx_train, true, error))
    {
        return false;
    }

    h.n_head_kv = h.n_head;
    if (!get_u32(file, "llama.attention.head_count_kv", h.n_head_kv, false, error))
    {
        return false;
    }
    if (h.n_embd % h.n_head != 0 || h.n_head % h.n_head_kv != 0)
    {
        error.set("embedding length %" PRIu32 ", %" PRIu32 " heads and %" PRIu32 " kv heads do not divide", h.n_embd,
                  h.n_head, h.n_head_kv);
        return false;
    }
    h.head_dim = h.n_embd / h.n_head;

    h.n_rot = h.head_dim;
    if (!get_u32(file, "llama.rope.dimension_count", h.n_rot, false, error))
    {
        return false;
    }
    if (h.n_rot > h.head_dim || h.n_rot % 2 != 0)
    {
        error.set("rope dimension count %" PRIu32 " does not fit a head of %" PRIu32, h.n_rot, h.head_dim);
        return false;
    }

    const GgufArray* tokens = file.get_array("tokenizer.ggml.tokens");
    uint64_t vocab = tokens != nullptr ? tokens->count : 0;
    file.get_uint("llama.vocab_size", vocab);
    if (vocab == 0 || vocab > 0xffffffffu)
    {
        error.set("the vocabulary size is missing");
        return false;
    }
    h.n_vocab = static_cast<uint32_t>(vocab);

    double value = 0.0;
    if (file.get_float("llama.attention.layer_norm_rms_epsilon", value))
    {
        h.rms_eps = static_cast<float>(value);
    }
    if (file.get_float("llama.rope.freq_base", value))
    {
        h.rope_freq_base = static_cast<float>(value);
    }

    std::string_view scaling;
    if (file.get_string("llama.rope.scaling.type", scaling) && !scaling.empty() && scaling != "none" &&
        scaling != "linear")
    {
        error.set("rope scaling '%.*s' is not supported yet", static_cast<int>(scaling.size()), scaling.data());
        return false;
    }
    if (file.get_float("llama.rope.scaling.factor", value) && value != 1.0)
    {
        error.set("rope scaling factor %g is not supported yet", value);
        return false;
    }
    return true;
}

const GgufTensor* LlamaModel::find(const GgufFile& file, const char* name, int layer, uint64_t n0, uint64_t n1,
                                   bool required, Error& error) const
{
    char full[128];
    if (layer >= 0)
    {
        std::snprintf(full, sizeof(full), "blk.%d.%s.weight", layer, name);
    }
    else
    {
        std::snprintf(full, sizeof(full), "%s.weight", name);
    }

    const GgufTensor* tensor = file.find_tensor(full);
    if (tensor == nullptr)
    {
        if (required)
        {
            error.set("tensor %s is missing", full);
        }
        return nullptr;
    }

    const uint32_t dims = n1 == 0 ? 1 : 2;
    if (tensor->n_dims != dims || tensor->dims[0] != n0 || (dims == 2 && tensor->dims[1] != n1))
    {
        error.set("tensor %s has shape [%" PRIu64 ", %" PRIu64 "], expected [%" PRIu64 ", %" PRIu64 "]", full,
                  tensor->dims[0], tensor->n_dims > 1 ? tensor->dims[1] : 0, n0, n1);
        return nullptr;
    }
    if (tensor->type != GgmlType::F32 && tensor->type != GgmlType::F16 && tensor->type != GgmlType::Q8_0 &&
        tensor->type != GgmlType::Q4_0)
    {
        error.set("tensor %s is %s, the runner reads F32, F16, Q8_0 and Q4_0", full, ggml_type_info(tensor->type)->name);
        return nullptr;
    }
    if (dims == 1 && tensor->type != GgmlType::F32 && tensor->type != GgmlType::F16)
    {
        error.set("tensor %s is %s, a norm must be F32 or F16", full, ggml_type_info(tensor->type)->name);
        return nullptr;
    }
    if ((tensor->type == GgmlType::Q8_0 || tensor->type == GgmlType::Q4_0) && tensor->dims[0] % Q8_0_BLOCK != 0)
    {
        error.set("tensor %s has a row of %" PRIu64 " values, not a multiple of %zu", full, tensor->dims[0], Q8_0_BLOCK);
        return nullptr;
    }
    return tensor;
}

bool LlamaModel::count_storage(const GgufTensor* tensor, size_t& elements, Error& error) const
{
    if (tensor == nullptr || tensor->type == GgmlType::F32)
    {
        return true;
    }
    // Q8_0 computes in place or packed; Q4_0 packed or, without VNNI kernels, as float32
    if (tensor->type == GgmlType::Q8_0 && !m_dequantize)
    {
        return true;
    }
    if (tensor->type == GgmlType::Q4_0 && m_pack)
    {
        return true;
    }
    if (tensor->n_elements > static_cast<uint64_t>(static_cast<size_t>(-1) / sizeof(float)) - elements)
    {
        error.set("the model does not fit in memory");
        return false;
    }
    elements += static_cast<size_t>(tensor->n_elements);
    return true;
}

Weight LlamaModel::materialize(const GgufTensor* tensor)
{
    Weight weight;
    if (tensor == nullptr)
    {
        return weight;
    }
    weight.n_in = static_cast<size_t>(tensor->dims[0]);
    weight.n_out = tensor->n_dims > 1 ? static_cast<size_t>(tensor->dims[1]) : 1;

    if ((tensor->type == GgmlType::Q8_0 || tensor->type == GgmlType::Q4_0) && m_pack)
    {
        weight.type = tensor->type;
        weight.packed = true;
        weight.data = m_packed + m_packed_used;
        if (tensor->type == GgmlType::Q4_0)
        {
            pack_q4_0(tensor->data, weight.n_in, weight.n_out, m_packed + m_packed_used);
            m_packed_used += q4_0_packed_bytes(weight.n_in, weight.n_out);
        }
        else
        {
            pack_q8_0(tensor->data, weight.n_in, weight.n_out, m_packed + m_packed_used);
            m_packed_used += q8_0_packed_bytes(weight.n_in, weight.n_out);
        }
        return weight;
    }
    if (tensor->type == GgmlType::F32 || (tensor->type == GgmlType::Q8_0 && !m_dequantize))
    {
        weight.type = tensor->type;
        weight.data = tensor->data;
        return weight;
    }

    const size_t count = static_cast<size_t>(tensor->n_elements);
    float* out = m_storage.data() + m_storage.size();
    m_storage.resize_uninitialized(m_storage.size() + count);
    weight.type = GgmlType::F32;
    weight.data = reinterpret_cast<const uint8_t*>(out);

    if (tensor->type == GgmlType::F16)
    {
        const uint8_t* src = tensor->data;
        for (size_t i = 0; i < count; ++i)
        {
            uint16_t bits = 0;
            std::memcpy(&bits, src + i * 2, 2);
            out[i] = f16_to_f32(bits);
        }
        return weight;
    }

    if (tensor->type == GgmlType::Q4_0)
    {
        dequantize_q4_0(tensor->data, count, out);
        return weight;
    }
    dequantize_q8_0(tensor->data, count, out);
    return weight;
}

bool LlamaModel::load(const GgufFile& file, Error& error, bool dequantize)
{
    release();
    m_dequantize = dequantize;
    m_pack = !dequantize && q8_0_packed_supported();
    if (!load_hparams(file, error))
    {
        return false;
    }

    const LlamaHparams& h = m_hparams;
    const uint64_t n_embd = h.n_embd;
    const uint64_t n_q = static_cast<uint64_t>(h.n_head) * h.head_dim;
    const uint64_t n_kv = static_cast<uint64_t>(h.n_head_kv) * h.head_dim;

    struct LayerTensors
    {
        const GgufTensor* tensors[9];
    };

    const GgufTensor* token_embd = find(file, "token_embd", -1, n_embd, h.n_vocab, true, error);
    const GgufTensor* output_norm = find(file, "output_norm", -1, n_embd, 0, true, error);
    if (token_embd == nullptr || output_norm == nullptr)
    {
        return false;
    }
    const GgufTensor* output = find(file, "output", -1, n_embd, h.n_vocab, false, error);
    if (output == nullptr && !error.empty())
    {
        return false;
    }

    Array<LayerTensors> layers;
    if (!layers.reserve(h.n_layer) || !m_layers.reserve(h.n_layer))
    {
        error.set("out of memory");
        return false;
    }

    size_t elements = 0;
    if (!count_storage(token_embd, elements, error) || !count_storage(output_norm, elements, error) ||
        !count_storage(output, elements, error))
    {
        return false;
    }

    for (uint32_t l = 0; l < h.n_layer; ++l)
    {
        const int il = static_cast<int>(l);
        LayerTensors t;
        t.tensors[0] = find(file, "attn_norm", il, n_embd, 0, true, error);
        t.tensors[1] = find(file, "attn_q", il, n_embd, n_q, true, error);
        t.tensors[2] = find(file, "attn_k", il, n_embd, n_kv, true, error);
        t.tensors[3] = find(file, "attn_v", il, n_embd, n_kv, true, error);
        t.tensors[4] = find(file, "attn_output", il, n_q, n_embd, true, error);
        t.tensors[5] = find(file, "ffn_norm", il, n_embd, 0, true, error);
        t.tensors[6] = find(file, "ffn_gate", il, n_embd, h.n_ff, true, error);
        t.tensors[7] = find(file, "ffn_up", il, n_embd, h.n_ff, true, error);
        t.tensors[8] = find(file, "ffn_down", il, h.n_ff, n_embd, true, error);
        for (const GgufTensor* tensor : t.tensors)
        {
            if (tensor == nullptr || !count_storage(tensor, elements, error))
            {
                return false;
            }
        }
        layers.push_back_unchecked(t);
    }

    if (!m_storage.reserve(elements))
    {
        error.set("out of memory: %zu MiB needed for the converted weights", elements * sizeof(float) >> 20);
        return false;
    }

    if (m_pack)
    {
        size_t packed_bytes = 0;
        const auto count_packed = [&packed_bytes](const GgufTensor* tensor)
        {
            if (tensor != nullptr && tensor->type == GgmlType::Q8_0)
            {
                packed_bytes += q8_0_packed_bytes(static_cast<size_t>(tensor->dims[0]), static_cast<size_t>(tensor->dims[1]));
            }
            else if (tensor != nullptr && tensor->type == GgmlType::Q4_0)
            {
                packed_bytes += q4_0_packed_bytes(static_cast<size_t>(tensor->dims[0]), static_cast<size_t>(tensor->dims[1]));
            }
        };
        count_packed(token_embd);
        count_packed(output);
        for (const LayerTensors& t : layers)
        {
            for (const GgufTensor* tensor : t.tensors)
            {
                count_packed(tensor);
            }
        }
        if (packed_bytes != 0)
        {
            m_packed = static_cast<uint8_t*>(allocate_large(packed_bytes));
            if (m_packed == nullptr)
            {
                error.set("out of memory: %zu MiB needed for the packed weights", packed_bytes >> 20);
                return false;
            }
            m_packed_bytes = packed_bytes;
        }
        else
        {
            m_pack = false;
        }
    }

    m_token_embd = materialize(token_embd);
    m_output_norm = materialize(output_norm).f32();
    m_output = output != nullptr ? materialize(output) : m_token_embd;
    for (const LayerTensors& t : layers)
    {
        LlamaLayer layer;
        layer.attn_norm = materialize(t.tensors[0]).f32();
        layer.wq = materialize(t.tensors[1]);
        layer.wk = materialize(t.tensors[2]);
        layer.wv = materialize(t.tensors[3]);
        layer.wo = materialize(t.tensors[4]);
        layer.ffn_norm = materialize(t.tensors[5]).f32();
        layer.ffn_gate = materialize(t.tensors[6]);
        layer.ffn_up = materialize(t.tensors[7]);
        layer.ffn_down = materialize(t.tensors[8]);
        m_layers.push_back_unchecked(layer);
    }
    return true;
}

}
