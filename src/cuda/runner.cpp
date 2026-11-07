#include "faisca/cuda_runner.h"
#include "faisca/ops.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace faisca
{

CudaRunner::~CudaRunner()
{
    release();
}

void CudaRunner::destroy_matrix(Matrix& matrix)
{
    Error ignored;
    m_device->destroy_buffer(matrix.data, ignored);
    matrix = Matrix();
}

void CudaRunner::release()
{
    if (m_device != nullptr)
    {
        Error ignored;
        m_device->synchronize(ignored);
        for (Layer& layer : m_layers)
        {
            m_device->destroy_buffer(layer.attn_norm, ignored);
            m_device->destroy_buffer(layer.ffn_norm, ignored);
            destroy_matrix(layer.wq);
            destroy_matrix(layer.wk);
            destroy_matrix(layer.wv);
            destroy_matrix(layer.wo);
            destroy_matrix(layer.gate);
            destroy_matrix(layer.up);
            destroy_matrix(layer.down);
        }
        destroy_matrix(m_embedding);
        if (!m_output_tied)
        {
            destroy_matrix(m_output);
        }
        m_output = Matrix();
        CudaBuffer* buffers[] = {&m_output_norm, &m_rope_factors, &m_tokens, &m_x, &m_h, &m_q, &m_k, &m_v,
                                &m_attn, &m_gate, &m_up, &m_act, &m_logits, &m_key_cache, &m_value_cache, &m_scores,
                                &m_attention_counters, &m_xq, &m_xs};
        for (CudaBuffer* buffer : buffers)
        {
            m_device->destroy_buffer(*buffer, ignored);
            *buffer = CudaBuffer();
        }
        m_device->destroy_graph(m_decode_graph, ignored);
        m_device->destroy_host_buffer(m_host_input, ignored);
        m_device->destroy_host_buffer(m_host_logits, ignored);
        m_kernels.release();
        m_device->destroy_stream(m_stream, ignored);
    }
    m_stream = CudaStream();
    m_decode_graph = CudaGraph();
    m_host_input = CudaHostBuffer();
    m_host_logits = CudaHostBuffer();
    m_layers.clear();
    m_device = nullptr;
    m_n_ctx = 0;
    m_pos = 0;
    m_weight_bytes = 0;
}

bool CudaRunner::upload_vector(const GgufTensor* tensor, CudaBuffer& out, size_t count, Error& error)
{
    if (tensor == nullptr || tensor->n_dims != 1 || tensor->n_elements != count ||
        (tensor->type != GgmlType::F32 && tensor->type != GgmlType::F16))
    {
        error.set("CUDA vector is missing or has an unsupported type/shape");
        return false;
    }
    Array<float> values;
    if (!values.resize_uninitialized(count))
    {
        error.set("out of memory uploading CUDA vector");
        return false;
    }
    if (tensor->type == GgmlType::F32)
    {
        std::memcpy(values.data(), tensor->data, count * 4);
    }
    else
    {
        for (size_t index = 0; index < count; ++index)
        {
            uint16_t bits = 0;
            std::memcpy(&bits, tensor->data + index * 2, 2);
            values[index] = f16_to_f32(bits);
        }
    }
    if (!m_device->create_buffer(count * 4, out, error) || !m_device->upload(out, values.data(), count * 4, error))
    {
        return false;
    }
    m_weight_bytes += count * 4;
    return true;
}

bool CudaRunner::upload_matrix(const GgufTensor* tensor, Matrix& out, uint32_t n_in, uint32_t n_out,
                               bool dequantize, Error& error)
{
    if (tensor == nullptr || tensor->n_dims != 2 || tensor->dims[0] != n_in || tensor->dims[1] != n_out ||
        (tensor->type != GgmlType::Q8_0 && tensor->type != GgmlType::F32 && tensor->type != GgmlType::F16))
    {
        error.set("CUDA matrix is missing or has an unsupported type/shape (requires F32, F16 or Q8_0)");
        return false;
    }
    if (tensor->type == GgmlType::Q8_0 && n_in % 32 != 0)
    {
        error.set("CUDA Q8_0 rows must contain a multiple of 32 values");
        return false;
    }
    out.n_in = n_in;
    out.n_out = n_out;
    out.q8 = tensor->type == GgmlType::Q8_0 && !dequantize;
    if (out.q8)
    {
        const size_t bytes = cuda_q8_0_bytes(n_in, n_out);
        Array<uint8_t> packed;
        if (!packed.resize_uninitialized(bytes))
        {
            error.set("out of memory packing CUDA Q8_0 weights");
            return false;
        }
        cuda_pack_q8_0(tensor->data, n_in, n_out, packed.data());
        if (!m_device->create_buffer(bytes, out.data, error) ||
            !m_device->upload(out.data, packed.data(), bytes, error))
        {
            return false;
        }
        m_weight_bytes += bytes;
        return true;
    }
    const size_t count = static_cast<size_t>(n_in) * n_out;
    Array<float> values;
    if (!values.resize_uninitialized(count))
    {
        error.set("out of memory decoding CUDA weights");
        return false;
    }
    Weight weight;
    weight.type = tensor->type;
    weight.data = tensor->data;
    weight.n_in = n_in;
    weight.n_out = n_out;
    for (uint32_t row = 0; row < n_out; ++row)
    {
        weight.row(row, values.data() + static_cast<size_t>(row) * n_in);
    }
    if (!m_device->create_buffer(count * 4, out.data, error) ||
        !m_device->upload(out.data, values.data(), count * 4, error))
    {
        return false;
    }
    m_weight_bytes += count * 4;
    return true;
}

bool CudaRunner::init(CudaDevice& device, const GgufFile& file, uint32_t n_ctx, Error& error, bool dequantize, bool kv_f16)
{
    release();
    if (!device.ready() || n_ctx == 0)
    {
        error.set("CUDA runner requires an initialized device and a positive context size");
        return false;
    }
    m_device = &device;
    m_kv_f16 = kv_f16;
    if (!LlamaModel::read_hparams(file, m_hparams, error))
    {
        release();
        return false;
    }
    if (m_hparams.arch != Arch::Llama)
    {
        error.set("CUDA currently supports the llama architecture only");
        release();
        return false;
    }
    if (m_hparams.rope_freq_scale != 1.0f || m_hparams.rope_neox || m_hparams.qk_norm ||
        m_hparams.post_norms || m_hparams.gelu_ffn || m_hparams.embd_scale != 1.0f ||
        m_hparams.attn_scale != 0.0f || m_hparams.n_swa != 0 || m_hparams.final_softcap != 0.0f)
    {
        error.set("CUDA does not yet support this model's scaling, window or normalization options");
        release();
        return false;
    }
    if (!m_kernels.init(device, error) || !device.create_stream(m_stream, error))
    {
        release();
        return false;
    }
    const LlamaHparams& h = m_hparams;
    m_n_ctx = n_ctx;
    const uint32_t n_query = h.n_head * h.head_dim;
    const uint32_t n_kv = h.n_head_kv * h.head_dim;
    if (!upload_matrix(file.find_tensor("token_embd.weight"), m_embedding, h.n_embd, h.n_vocab, dequantize, error) ||
        !upload_vector(file.find_tensor("output_norm.weight"), m_output_norm, h.n_embd, error))
    {
        release();
        return false;
    }
    const GgufTensor* output = file.find_tensor("output.weight");
    m_output_tied = output == nullptr;
    if (m_output_tied)
    {
        m_output = m_embedding;
    }
    else if (!upload_matrix(output, m_output, h.n_embd, h.n_vocab, dequantize, error))
    {
        release();
        return false;
    }
    const GgufTensor* factors = file.find_tensor("rope_freqs.weight");
    m_has_rope_factors = factors != nullptr;
    if (m_has_rope_factors ? !upload_vector(factors, m_rope_factors, h.n_rot / 2, error)
                           : !device.create_buffer(4, m_rope_factors, error))
    {
        release();
        return false;
    }
    if (!m_layers.resize_uninitialized(h.n_layer))
    {
        error.set("out of memory allocating CUDA layers");
        release();
        return false;
    }
    for (Layer& layer : m_layers)
    {
        layer = Layer();
    }
    for (uint32_t index = 0; index < h.n_layer; ++index)
    {
        Layer& layer = m_layers[index];
        struct Item
        {
            const char* name;
            Matrix* matrix;
            CudaBuffer* vector;
            uint32_t n_in;
            uint32_t n_out;
        };
        const Item items[] = {{"attn_norm", nullptr, &layer.attn_norm, h.n_embd, 1},
                              {"ffn_norm", nullptr, &layer.ffn_norm, h.n_embd, 1},
                              {"attn_q", &layer.wq, nullptr, h.n_embd, n_query},
                              {"attn_k", &layer.wk, nullptr, h.n_embd, n_kv},
                              {"attn_v", &layer.wv, nullptr, h.n_embd, n_kv},
                              {"attn_output", &layer.wo, nullptr, n_query, h.n_embd},
                              {"ffn_gate", &layer.gate, nullptr, h.n_embd, h.n_ff},
                              {"ffn_up", &layer.up, nullptr, h.n_embd, h.n_ff},
                              {"ffn_down", &layer.down, nullptr, h.n_ff, h.n_embd}};
        for (const Item& item : items)
        {
            char name[128];
            std::snprintf(name, sizeof(name), "blk.%" PRIu32 ".%s.weight", index, item.name);
            const GgufTensor* tensor = file.find_tensor(name);
            const bool ok = item.matrix != nullptr
                                ? upload_matrix(tensor, *item.matrix, item.n_in, item.n_out, dequantize, error)
                                : upload_vector(tensor, *item.vector, item.n_in, error);
            if (!ok)
            {
                release();
                return false;
            }
        }
    }
    const size_t cache_bytes = static_cast<size_t>(h.n_layer) * n_ctx * n_kv * (kv_f16 ? 2 : 4);
    size_t widest = h.n_embd > h.n_ff ? h.n_embd : h.n_ff;
    widest = widest > n_query ? widest : n_query;
    // attention_split keeps head_dim + 4 floats per split of 128 positions
    const size_t splits = (static_cast<size_t>(n_ctx) + 127) / 128;
    const size_t scratch = splits * (h.head_dim + 4) > n_ctx ? splits * (h.head_dim + 4) : n_ctx;
    const bool ok = device.create_buffer((BATCH + 1) * 4, m_tokens, error) &&
                    device.create_host_buffer((BATCH + 1) * 4, m_host_input, error) &&
                    device.create_host_buffer(static_cast<size_t>(h.n_vocab) * 4, m_host_logits, error) &&
                    device.create_buffer(BATCH * h.n_embd * 4, m_x, error) &&
                    device.create_buffer(BATCH * h.n_embd * 4, m_h, error) &&
                    device.create_buffer(BATCH * n_query * 4, m_q, error) &&
                    device.create_buffer(BATCH * n_kv * 4, m_k, error) &&
                    device.create_buffer(BATCH * n_kv * 4, m_v, error) &&
                    device.create_buffer(BATCH * n_query * 4, m_attn, error) &&
                    device.create_buffer(BATCH * h.n_ff * 4, m_gate, error) &&
                    device.create_buffer(BATCH * h.n_ff * 4, m_up, error) &&
                    device.create_buffer(BATCH * h.n_ff * 4, m_act, error) &&
                    device.create_buffer(BATCH * widest, m_xq, error) &&
                    device.create_buffer(BATCH * widest / 32 * 2, m_xs, error) &&
                    device.create_buffer(BATCH * h.n_vocab * 4, m_logits, error) &&
                    device.create_buffer(cache_bytes, m_key_cache, error) &&
                    device.create_buffer(cache_bytes, m_value_cache, error) &&
                    device.create_buffer(BATCH * h.n_head * scratch * 4, m_scores, error) &&
                    device.create_buffer(BATCH * h.n_head * 4, m_attention_counters, error);
    Array<uint32_t> zeros;
    if (ok && zeros.resize_uninitialized(BATCH * h.n_head))
    {
        std::memset(zeros.data(), 0, zeros.size() * 4);
    }
    if (!ok || zeros.size() != BATCH * h.n_head ||
        !device.upload(m_attention_counters, zeros.data(), zeros.size() * 4, error))
    {
        release();
        return false;
    }
    return true;
}

bool CudaRunner::matvec(const Matrix& matrix, const CudaBuffer& input, const CudaBuffer& output,
                         uint32_t count, Error& error, bool accumulate)
{
    if (gemm_ready(matrix, count))
    {
        return quantize(input, matrix.n_in, count, error) && gemm(matrix, output, count, error, accumulate);
    }
    if (accumulate && !matrix.q8)
    {
        error.set("CUDA float32 matvec cannot accumulate");
        return false;
    }
    return matrix.q8 ? m_kernels.matvec_q8_0(matrix.data, input, output, matrix.n_in, matrix.n_out, count, m_stream,
                                             error, accumulate)
                     : m_kernels.matvec_f32(matrix.data, input, output, matrix.n_in, matrix.n_out, count, m_stream, error);
}

bool CudaRunner::quantize(const CudaBuffer& input, uint32_t n_in, uint32_t count, Error& error)
{
    return m_kernels.quantize_q8_0(input, m_xq, m_xs, n_in, count, BATCH, m_stream, error);
}

bool CudaRunner::gemm(const Matrix& matrix, const CudaBuffer& output, uint32_t count, Error& error, bool accumulate)
{
    return m_kernels.gemm_q8_0(matrix.data, m_xq, m_xs, BATCH, output, matrix.n_in, matrix.n_out, count, m_stream,
                               error, accumulate);
}

bool CudaRunner::trace(const char* name, int layer, const CudaBuffer& buffer, size_t count,
                       uint32_t n_tokens, Error& error)
{
    if (m_trace == nullptr)
    {
        return true;
    }
    if (!m_staging.resize_uninitialized(count * n_tokens))
    {
        error.set("out of memory tracing CUDA runner");
        return false;
    }
    if (!m_device->synchronize(m_stream, error) ||
        !m_device->download(buffer, m_staging.data(), count * n_tokens * 4, error))
    {
        return false;
    }
    for (uint32_t token = 0; token < n_tokens; ++token)
    {
        m_trace(m_trace_user, name, layer, m_staging.data() + token * count, count);
    }
    return true;
}

bool CudaRunner::forward_batch(uint32_t count, float* logits, bool all_logits, Error& error)
{
    const LlamaHparams& h = m_hparams;
    const bool all = all_logits || m_trace != nullptr;
    if (!enqueue_batch(count, all, 0, error) || !m_device->synchronize(m_stream, error))
    {
        return false;
    }
    const size_t offset = !all_logits && all ? static_cast<size_t>(count - 1) * h.n_vocab * 4 : 0;
    const size_t bytes = static_cast<size_t>(all_logits ? count : 1) * h.n_vocab * 4;
    return m_device->download(m_logits, logits, bytes, error, offset);
}

bool CudaRunner::forward_decode(uint32_t token, float* logits, Error& error)
{
    uint32_t* input = static_cast<uint32_t*>(m_host_input.data);
    input[0] = token;
    input[BATCH] = m_pos;
    if (!m_device->upload_async(m_tokens, m_host_input, (BATCH + 1) * 4, m_stream, error))
    {
        return false;
    }
    if (m_decode_graph.handle == nullptr)
    {
        if (!m_device->begin_capture(m_stream, error))
        {
            return false;
        }
        const bool recorded = enqueue_batch(1, false, m_tokens.address + BATCH * 4, error) &&
                              m_device->download_async(m_logits, m_host_logits, m_host_logits.bytes, m_stream, error);
        Error capture_error;
        const bool captured = m_device->end_capture(m_stream, m_decode_graph, recorded ? error : capture_error);
        if (!recorded || !captured)
        {
            return false;
        }
    }
    if (!m_device->launch_graph(m_decode_graph, m_stream, error) || !m_device->synchronize(m_stream, error))
    {
        return false;
    }
    std::memcpy(logits, m_host_logits.data, m_host_logits.bytes);
    return true;
}

bool CudaRunner::enqueue_batch(uint32_t count, bool all, uint64_t position_at, Error& error)
{
    const LlamaHparams& h = m_hparams;
    const uint32_t n_query = h.n_head * h.head_dim;
    const uint32_t n_kv = h.n_head_kv * h.head_dim;
    CudaKernels& kernels = m_kernels;
    bool ok = kernels.embed(m_embedding.data, m_tokens, m_x, h.n_embd, m_embedding.n_out, count, m_embedding.q8, m_stream, error) &&
              trace("inp_embd", -1, m_x, h.n_embd, count, error);
    for (uint32_t index = 0; ok && index < h.n_layer; ++index)
    {
        const Layer& layer = m_layers[index];
        const int traced_layer = static_cast<int>(index);
        // Without a trace the q/k/v products share one launch and the residual adds and swiglu
        // fold into the matvecs, with the same float operations as the separate kernels.
        const bool fused = m_trace == nullptr && layer.wq.q8 && layer.wk.q8 && layer.wv.q8 && layer.wo.q8 &&
                           layer.gate.q8 && layer.up.q8 && layer.down.q8;
        if (fused)
        {
            const CudaBuffer weights[] = {layer.wq.data, layer.wk.data, layer.wv.data};
            const CudaBuffer outputs[] = {m_q, m_k, m_v};
            const uint32_t rows[] = {n_query, n_kv, n_kv};
            const bool batched = gemm_ready(layer.wq, count) && gemm_ready(layer.wk, count) &&
                                 gemm_ready(layer.wv, count) && gemm_ready(layer.wo, count) &&
                                 gemm_ready(layer.gate, count) && gemm_ready(layer.up, count) &&
                                 gemm_ready(layer.down, count);
            if (batched)
            {
                ok = kernels.rms_norm(m_x, layer.attn_norm, m_h, h.n_embd, count, h.rms_eps, m_stream, error) &&
                     quantize(m_h, h.n_embd, count, error) && gemm(layer.wq, m_q, count, error) &&
                     gemm(layer.wk, m_k, count, error) && gemm(layer.wv, m_v, count, error) &&
                     kernels.rope_store(m_q, m_k, m_v, m_key_cache, m_value_cache, m_rope_factors, m_has_rope_factors,
                                        h.n_head, h.n_head_kv, h.head_dim, h.n_rot, m_n_ctx, m_pos, index,
                                        h.rope_freq_base, count, m_stream, error, position_at, m_kv_f16) &&
                     kernels.attention(m_q, m_key_cache, m_value_cache, m_scores, m_attn, h.n_head, h.n_head_kv,
                                       h.head_dim, m_n_ctx, m_pos, index, count, m_stream, error, position_at,
                                       m_kv_f16, &m_attention_counters) &&
                     matvec(layer.wo, m_attn, m_x, count, error, true) &&
                     kernels.rms_norm(m_x, layer.ffn_norm, m_h, h.n_embd, count, h.rms_eps, m_stream, error) &&
                     quantize(m_h, h.n_embd, count, error) && gemm(layer.gate, m_gate, count, error) &&
                     gemm(layer.up, m_up, count, error) &&
                     kernels.swiglu(m_gate, m_up, m_act, count * h.n_ff, m_stream, error) &&
                     matvec(layer.down, m_act, m_x, count, error, true);
                continue;
            }
            ok = kernels.rms_norm(m_x, layer.attn_norm, m_h, h.n_embd, count, h.rms_eps, m_stream, error) &&
                 kernels.matvec_q8_0_stacked(weights, outputs, rows, 3, m_h, h.n_embd, count, m_stream, error) &&
                 kernels.rope_store(m_q, m_k, m_v, m_key_cache, m_value_cache, m_rope_factors, m_has_rope_factors,
                                    h.n_head, h.n_head_kv, h.head_dim, h.n_rot, m_n_ctx, m_pos, index,
                                    h.rope_freq_base, count, m_stream, error, position_at, m_kv_f16) &&
                 kernels.attention(m_q, m_key_cache, m_value_cache, m_scores, m_attn, h.n_head, h.n_head_kv,
                                   h.head_dim, m_n_ctx, m_pos, index, count, m_stream, error, position_at, m_kv_f16,
                                   &m_attention_counters) &&
                 kernels.matvec_q8_0(layer.wo.data, m_attn, m_x, n_query, h.n_embd, count, m_stream, error, true) &&
                 kernels.rms_norm(m_x, layer.ffn_norm, m_h, h.n_embd, count, h.rms_eps, m_stream, error) &&
                 kernels.matvec_q8_0_swiglu(layer.gate.data, layer.up.data, m_h, m_act, h.n_embd, h.n_ff, count,
                                            m_stream, error) &&
                 kernels.matvec_q8_0(layer.down.data, m_act, m_x, h.n_ff, h.n_embd, count, m_stream, error, true);
            continue;
        }
        ok = kernels.rms_norm(m_x, layer.attn_norm, m_h, h.n_embd, count, h.rms_eps, m_stream, error) &&
             trace("attn_norm", traced_layer, m_h, h.n_embd, count, error) &&
             matvec(layer.wq, m_h, m_q, count, error) && matvec(layer.wk, m_h, m_k, count, error) &&
             matvec(layer.wv, m_h, m_v, count, error) &&
             kernels.rope(m_q, m_rope_factors, m_has_rope_factors, h.n_head, h.head_dim, h.n_rot, m_pos,
                          h.rope_freq_base, count, m_stream, error, position_at) &&
             kernels.rope(m_k, m_rope_factors, m_has_rope_factors, h.n_head_kv, h.head_dim, h.n_rot, m_pos,
                          h.rope_freq_base, count, m_stream, error, position_at) &&
             kernels.kv_store(m_k, m_v, m_key_cache, m_value_cache, h.n_head_kv, h.head_dim, m_n_ctx,
                              m_pos, index, count, m_stream, error, position_at, m_kv_f16) &&
             trace("Qcur", traced_layer, m_q, n_query, count, error) &&
             trace("Kcur", traced_layer, m_k, n_kv, count, error) && trace("Vcur", traced_layer, m_v, n_kv, count, error) &&
             kernels.attention(m_q, m_key_cache, m_value_cache, m_scores, m_attn, h.n_head, h.n_head_kv,
                               h.head_dim, m_n_ctx, m_pos, index, count, m_stream, error, position_at, m_kv_f16) &&
             matvec(layer.wo, m_attn, m_h, count, error) && trace("attn_out", traced_layer, m_h, h.n_embd, count, error) &&
             kernels.add(m_x, m_h, count * h.n_embd, m_stream, error) && trace("ffn_inp", traced_layer, m_x, h.n_embd, count, error) &&
             kernels.rms_norm(m_x, layer.ffn_norm, m_h, h.n_embd, count, h.rms_eps, m_stream, error) &&
             trace("ffn_norm", traced_layer, m_h, h.n_embd, count, error) &&
             matvec(layer.gate, m_h, m_gate, count, error) && matvec(layer.up, m_h, m_up, count, error) &&
             trace("ffn_gate", traced_layer, m_gate, h.n_ff, count, error) && trace("ffn_up", traced_layer, m_up, h.n_ff, count, error) &&
             kernels.swiglu(m_gate, m_up, m_act, count * h.n_ff, m_stream, error) &&
             trace("ffn_swiglu", traced_layer, m_act, h.n_ff, count, error) &&
             matvec(layer.down, m_act, m_h, count, error) && trace("ffn_out", traced_layer, m_h, h.n_embd, count, error) &&
             kernels.add(m_x, m_h, count * h.n_embd, m_stream, error) && trace("l_out", traced_layer, m_x, h.n_embd, count, error);
    }
    if (!ok)
    {
        return false;
    }
    CudaBuffer input = m_x;
    if (!all)
    {
        const size_t offset = static_cast<size_t>(count - 1) * h.n_embd * 4;
        input.address += offset;
        input.bytes -= offset;
    }
    const uint32_t output_count = all ? count : 1;
    if (!kernels.rms_norm(input, m_output_norm, m_h, h.n_embd, output_count, h.rms_eps, m_stream, error) ||
        !trace("result_norm", -1, m_h, h.n_embd, output_count, error) ||
        !matvec(m_output, m_h, m_logits, output_count, error) ||
        !trace("result_output", -1, m_logits, h.n_vocab, output_count, error))
    {
        return false;
    }
    return true;
}

bool CudaRunner::forward(const int32_t* tokens, size_t count, Array<float>& logits, Error& error, bool all_logits)
{
    if (m_device == nullptr || count == 0 || tokens == nullptr)
    {
        error.set("CUDA runner is not initialized or has no input tokens");
        return false;
    }
    if (count > m_n_ctx - m_pos)
    {
        error.set("CUDA context full: %zu tokens at position %" PRIu32 " exceed %" PRIu32, count, m_pos, m_n_ctx);
        return false;
    }
    for (size_t index = 0; index < count; ++index)
    {
        if (tokens[index] < 0 || static_cast<uint32_t>(tokens[index]) >= m_hparams.n_vocab)
        {
            error.set("CUDA token %" PRId32 " is outside the vocabulary", tokens[index]);
            return false;
        }
    }
    if (!logits.resize_uninitialized((all_logits ? count : 1) * m_hparams.n_vocab))
    {
        error.set("out of memory allocating CUDA logits");
        return false;
    }
    for (size_t done = 0; done < count;)
    {
        const uint32_t batch = static_cast<uint32_t>(count - done < BATCH ? count - done : BATCH);
        uint32_t ids[BATCH];
        for (uint32_t index = 0; index < batch; ++index)
        {
            ids[index] = static_cast<uint32_t>(tokens[done + index]);
        }
        float* out = logits.data() + (all_logits ? done * m_hparams.n_vocab : 0);
        if (batch == 1 && m_trace == nullptr)
        {
            if (!forward_decode(ids[0], out, error))
            {
                return false;
            }
        }
        else if (!m_device->upload(m_tokens, ids, static_cast<size_t>(batch) * 4, error) ||
                 !forward_batch(batch, out, all_logits, error))
        {
            return false;
        }
        m_pos += batch;
        done += batch;
    }
    return true;
}

}