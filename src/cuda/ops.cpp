#include "faisca/cuda_ops.h"

#include "kernels_ptx.h"
#include <cmath>
#include <cstring>

namespace faisca
{

size_t cuda_q8_0_bytes(uint32_t n_in, uint32_t n_rows)
{
    // quants, f16 scales row by row, the same scales block by block
    return static_cast<size_t>(n_in) * n_rows + static_cast<size_t>(n_in / 32) * n_rows * 4;
}

void cuda_pack_q8_0(const void* blocks, uint32_t n_in, uint32_t n_rows, uint8_t* out)
{
    const uint8_t* source = static_cast<const uint8_t*>(blocks);
    const size_t per_row = n_in / 32;
    const size_t block_count = per_row * n_rows;
    uint8_t* quants = out;
    uint8_t* scales = out + static_cast<size_t>(n_in) * n_rows;
    uint8_t* scales_by_block = scales + block_count * 2;
    for (size_t block = 0; block < block_count; ++block)
    {
        const uint8_t* in = source + block * 34;
        scales[block * 2] = in[0];
        scales[block * 2 + 1] = in[1];
        const size_t row = block / per_row;
        const size_t column = block % per_row;
        scales_by_block[(column * n_rows + row) * 2] = in[0];
        scales_by_block[(column * n_rows + row) * 2 + 1] = in[1];
        std::memcpy(quants + block * 32, in + 2, 32);
    }
}


CudaKernels::~CudaKernels()
{
    release();
}

bool CudaKernels::init(CudaDevice& device, Error& error)
{
    release();
    m_device = &device;
    if (!device.load_module(ptx_matvec_f32, m_matvec_f32_module, error) ||
        !device.find_kernel(m_matvec_f32_module, "matvec_f32", m_matvec_f32, error) ||
        !device.load_module(ptx_matvec_q8_0, m_matvec_q8_0_module, error) ||
        !device.find_kernel(m_matvec_q8_0_module, "matvec_q8_0", m_matvec_q8_0, error) ||
        !device.find_kernel(m_matvec_q8_0_module, "matvec_q8_0_swiglu", m_matvec_q8_0_swiglu, error) ||
        !device.find_kernel(m_matvec_q8_0_module, "matvec_q8_0_rows4", m_matvec_q8_0_rows4, error) ||
        !device.find_kernel(m_matvec_q8_0_module, "matvec_q8_0_swiglu_rows2", m_matvec_q8_0_swiglu_rows2, error) ||
        !device.load_module(ptx_embed, m_embed_module, error) ||
        !device.find_kernel(m_embed_module, "embed", m_embed, error) ||
        !device.load_module(ptx_rms_norm, m_rms_norm_module, error) ||
        !device.find_kernel(m_rms_norm_module, "rms_norm", m_rms_norm, error) ||
        !device.load_module(ptx_rope, m_rope_module, error) ||
        !device.find_kernel(m_rope_module, "rope", m_rope, error) ||
        !device.find_kernel(m_rope_module, "rope_store", m_rope_store, error) ||
        !device.load_module(ptx_kv_store, m_kv_store_module, error) ||
        !device.find_kernel(m_kv_store_module, "kv_store", m_kv_store, error) ||
        !device.load_module(ptx_attention, m_attention_module, error) ||
        !device.find_kernel(m_attention_module, "attention", m_attention, error) ||
        !device.find_kernel(m_attention_module, "attention_split", m_attention_split, error) ||
        !device.find_kernel(m_attention_module, "attention_split_64", m_attention_split_64, error) ||
        !device.find_kernel(m_attention_module, "attention_split_128", m_attention_split_128, error) ||
        !device.find_kernel(m_attention_module, "attention_prefill_64", m_attention_prefill_64, error) ||
        !device.find_kernel(m_attention_module, "attention_prefill_128", m_attention_prefill_128, error) ||
        !device.find_kernel(m_attention_module, "attention_prefill_tc_64", m_attention_prefill_tc_64, error) ||
        !device.find_kernel(m_attention_module, "attention_combine", m_attention_combine, error) ||
        !device.load_module(ptx_swiglu, m_swiglu_module, error) ||
        !device.find_kernel(m_swiglu_module, "swiglu", m_swiglu, error) ||
        !device.load_module(ptx_gemm_q8_0, m_gemm_module, error) ||
        !device.find_kernel(m_gemm_module, "quantize_q8_0", m_quantize_q8_0, error) ||
        !device.find_kernel(m_gemm_module, "gemm_q8_0", m_gemm_q8_0, error) ||
        !device.load_module(ptx_gemm_f16, m_gemm_f16_module, error) ||
        !device.find_kernel(m_gemm_f16_module, "gemm_f16", m_gemm_f16, error) ||
        !device.find_kernel(m_gemm_f16_module, "gemm_bf16", m_gemm_bf16, error) ||
        !device.find_kernel(m_gemm_f16_module, "gemm_f16_half", m_gemm_outputs[0][0], error) ||
        !device.find_kernel(m_gemm_f16_module, "gemm_f16_gelu", m_gemm_outputs[0][1], error) ||
        !device.find_kernel(m_gemm_f16_module, "gemm_f16_residual", m_gemm_outputs[0][2], error) ||
        !device.find_kernel(m_gemm_f16_module, "gemm_bf16_half", m_gemm_outputs[1][0], error) ||
        !device.find_kernel(m_gemm_f16_module, "gemm_bf16_gelu", m_gemm_outputs[1][1], error) ||
        !device.find_kernel(m_gemm_f16_module, "gemm_bf16_residual", m_gemm_outputs[1][2], error) ||
        !device.load_module(ptx_dit, m_dit_module, error) ||
        !device.find_kernel(m_dit_module, "dit_modulate_norm_f16", m_dit_modulate_norm[0], error) ||
        !device.find_kernel(m_dit_module, "dit_modulate_norm_bf16", m_dit_modulate_norm[1], error) ||
        !device.find_kernel(m_dit_module, "dit_qk_norm_rope_f16", m_dit_qk_norm_rope[0], error) ||
        !device.find_kernel(m_dit_module, "dit_qk_norm_rope_bf16", m_dit_qk_norm_rope[1], error) ||
        !device.find_kernel(m_dit_module, "dit_matvec_bf16", m_dit_matvec, error) ||
        !device.find_kernel(m_dit_module, "dit_copy", m_dit_copy, error) ||
        !device.load_module(ptx_attention_f16, m_attention_f16_module, error) ||
        !device.find_kernel(m_attention_f16_module, "attention_f16", m_attention_f16, error) ||
        !device.find_kernel(m_attention_f16_module, "attention_bf16", m_attention_bf16, error) ||
        !device.load_module(ptx_add, m_add_module, error) ||
        !device.find_kernel(m_add_module, "add", m_add, error))
    {
        release();
        return false;
    }
    return true;
}

void CudaKernels::release()
{
    if (m_device != nullptr)
    {
        Error ignored;
        m_device->destroy_module(m_matvec_f32_module, ignored);
        m_device->destroy_module(m_matvec_q8_0_module, ignored);
        m_device->destroy_module(m_embed_module, ignored);
        m_device->destroy_module(m_rms_norm_module, ignored);
        m_device->destroy_module(m_rope_module, ignored);
        m_device->destroy_module(m_kv_store_module, ignored);
        m_device->destroy_module(m_attention_module, ignored);
        m_device->destroy_module(m_swiglu_module, ignored);
        m_device->destroy_module(m_add_module, ignored);
        m_device->destroy_module(m_gemm_module, ignored);
        m_device->destroy_module(m_gemm_f16_module, ignored);
        m_device->destroy_module(m_attention_f16_module, ignored);
        m_device->destroy_module(m_dit_module, ignored);
    }
    m_matvec_f32_module = CudaModule();
    m_matvec_f32 = CudaKernel();
    m_matvec_q8_0_module = CudaModule();
    m_matvec_q8_0 = CudaKernel();
    m_matvec_q8_0_swiglu = CudaKernel();
    m_matvec_q8_0_rows4 = CudaKernel();
    m_matvec_q8_0_swiglu_rows2 = CudaKernel();
    m_embed_module = CudaModule();
    m_embed = CudaKernel();
    m_rms_norm_module = CudaModule();
    m_rms_norm = CudaKernel();
    m_rope_module = CudaModule();
    m_rope = CudaKernel();
    m_rope_store = CudaKernel();
    m_kv_store_module = CudaModule();
    m_kv_store = CudaKernel();
    m_attention_module = CudaModule();
    m_attention = CudaKernel();
    m_attention_split = CudaKernel();
    m_attention_split_64 = CudaKernel();
    m_attention_split_128 = CudaKernel();
    m_attention_prefill_64 = CudaKernel();
    m_attention_prefill_128 = CudaKernel();
    m_attention_prefill_tc_64 = CudaKernel();
    m_attention_combine = CudaKernel();
    m_swiglu_module = CudaModule();
    m_swiglu = CudaKernel();
    m_add_module = CudaModule();
    m_gemm_module = CudaModule();
    m_quantize_q8_0 = CudaKernel();
    m_gemm_q8_0 = CudaKernel();
    m_gemm_f16_module = CudaModule();
    m_gemm_f16 = CudaKernel();
    m_gemm_bf16 = CudaKernel();
    m_attention_f16_module = CudaModule();
    m_dit_module = CudaModule();
    m_dit_matvec = CudaKernel();
    m_dit_copy = CudaKernel();
    for (int t = 0; t < 2; ++t)
    {
        for (int o = 0; o < 3; ++o)
        {
            m_gemm_outputs[t][o] = CudaKernel();
        }
        m_dit_modulate_norm[t] = CudaKernel();
        m_dit_qk_norm_rope[t] = CudaKernel();
    }
    m_attention_f16 = CudaKernel();
    m_attention_bf16 = CudaKernel();
    m_add = CudaKernel();
    m_device = nullptr;
}

bool CudaKernels::matvec_f32(const CudaBuffer& weight, const CudaBuffer& input, const CudaBuffer& output,
                            uint32_t n_in, uint32_t n_out, uint32_t n_tokens, const CudaStream& stream, Error& error)
{
    if (m_device == nullptr || n_in == 0 || n_out == 0 || n_tokens == 0 ||
        weight.bytes < static_cast<size_t>(n_in) * n_out * sizeof(float) ||
        input.bytes < static_cast<size_t>(n_in) * n_tokens * sizeof(float) ||
        output.bytes < static_cast<size_t>(n_out) * n_tokens * sizeof(float))
    {
        error.set("invalid CUDA matvec dimensions or buffer size");
        return false;
    }
    uint64_t addresses[] = {weight.address, input.address, output.address};
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &n_in, &n_out, &n_tokens};
    CudaLaunch launch;
    launch.grid_x = (n_out + 3) / 4;
    launch.grid_y = (n_tokens + 3) / 4;
    launch.block_x = 128;
    return m_device->launch(m_matvec_f32, stream, launch, arguments, error);
}

bool CudaKernels::embed(const CudaBuffer& weight, const CudaBuffer& tokens, const CudaBuffer& output,
                       uint32_t n_embd, uint32_t n_rows, uint32_t n_tokens, bool q8, const CudaStream& stream,
                       Error& error)
{
    if (m_device == nullptr || n_embd == 0 || n_rows == 0 || n_tokens == 0 || (q8 && n_embd % 32 != 0) ||
        weight.bytes < (q8 ? cuda_q8_0_bytes(n_embd, n_rows) : static_cast<size_t>(n_embd) * n_rows * 4) ||
        tokens.bytes < static_cast<size_t>(n_tokens) * 4 || output.bytes < static_cast<size_t>(n_embd) * n_tokens * 4)
    {
        error.set("invalid CUDA embedding dimensions");
        return false;
    }
    uint64_t addresses[] = {weight.address, q8 ? weight.address + static_cast<uint64_t>(n_embd) * n_rows : 0,
                            tokens.address, output.address};
    uint32_t quantized = q8 ? 1u : 0u;
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &n_embd, &quantized};
    CudaLaunch launch;
    launch.grid_x = (n_embd + 255) / 256;
    launch.grid_y = n_tokens;
    launch.block_x = 256;
    return m_device->launch(m_embed, stream, launch, arguments, error);
}

bool CudaKernels::rms_norm(const CudaBuffer& input, const CudaBuffer& weight, const CudaBuffer& output,
                          uint32_t width, uint32_t n_tokens, float epsilon, const CudaStream& stream, Error& error)
{
    if (m_device == nullptr || width == 0 || n_tokens == 0 || epsilon < 0 ||
        input.bytes < static_cast<size_t>(width) * n_tokens * 4 || weight.bytes < static_cast<size_t>(width) * 4 ||
        output.bytes < static_cast<size_t>(width) * n_tokens * 4)
    {
        error.set("invalid CUDA RMS norm dimensions");
        return false;
    }
    uint64_t addresses[] = {input.address, weight.address, output.address};
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &width, &epsilon};
    CudaLaunch launch;
    launch.grid_x = n_tokens;
    launch.block_x = 256;
    return m_device->launch(m_rms_norm, stream, launch, arguments, error);
}

bool CudaKernels::rope(const CudaBuffer& input, const CudaBuffer& factors, bool has_factors, uint32_t n_head,
                      uint32_t head_dim, uint32_t n_rot, uint32_t position, float freq_base, uint32_t n_tokens,
                      const CudaStream& stream, Error& error, uint64_t position_at)
{
    if (m_device == nullptr || n_head == 0 || head_dim == 0 || n_tokens == 0 || n_rot == 0 ||
        n_rot > head_dim || n_rot % 2 != 0 || freq_base <= 0 ||
        input.bytes < static_cast<size_t>(n_tokens) * n_head * head_dim * 4 ||
        (has_factors && factors.bytes < static_cast<size_t>(n_rot / 2) * 4))
    {
        error.set("invalid CUDA RoPE dimensions");
        return false;
    }
    uint64_t addresses[] = {input.address, factors.address};
    uint32_t present = has_factors ? 1u : 0u;
    float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));
    void* arguments[] = {&addresses[0], &addresses[1], &present, &n_head, &head_dim, &n_rot, &position, &position_at,
                         &theta_scale};
    CudaLaunch launch;
    launch.grid_x = n_tokens;
    launch.grid_y = n_head;
    launch.block_x = 128;
    return m_device->launch(m_rope, stream, launch, arguments, error);
}

bool CudaKernels::rope_store(const CudaBuffer& query, const CudaBuffer& keys, const CudaBuffer& values,
                            const CudaBuffer& key_cache, const CudaBuffer& value_cache, const CudaBuffer& factors,
                            bool has_factors, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
                            uint32_t n_ctx, uint32_t position, uint32_t layer, float freq_base, uint32_t n_tokens,
                            const CudaStream& stream, Error& error, uint64_t position_at, bool half_cache)
{
    const size_t cache_bytes = (static_cast<size_t>(layer) + 1) * n_head_kv * n_ctx * head_dim * (half_cache ? 2 : 4);
    const size_t kv_bytes = static_cast<size_t>(n_tokens) * n_head_kv * head_dim * 4;
    if (m_device == nullptr || n_head == 0 || n_head_kv == 0 || head_dim == 0 || n_tokens == 0 || n_rot == 0 ||
        n_rot > head_dim || n_rot % 2 != 0 || freq_base <= 0 || position > n_ctx || n_tokens > n_ctx - position ||
        query.bytes < static_cast<size_t>(n_tokens) * n_head * head_dim * 4 || keys.bytes < kv_bytes ||
        values.bytes < kv_bytes || key_cache.bytes < cache_bytes || value_cache.bytes < cache_bytes ||
        (has_factors && factors.bytes < static_cast<size_t>(n_rot / 2) * 4))
    {
        error.set("invalid CUDA RoPE and KV cache dimensions");
        return false;
    }
    uint64_t addresses[] = {query.address, keys.address, values.address, key_cache.address, value_cache.address,
                            factors.address};
    uint32_t present = has_factors ? 1u : 0u;
    uint32_t half = half_cache ? 1u : 0u;
    float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(n_rot));
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &addresses[4], &addresses[5],
                         &present, &n_head, &n_head_kv, &head_dim, &n_rot, &n_ctx, &position, &position_at, &layer,
                         &theta_scale, &half};
    CudaLaunch launch;
    launch.grid_x = n_tokens;
    launch.grid_y = n_head + n_head_kv;
    launch.block_x = 128;
    return m_device->launch(m_rope_store, stream, launch, arguments, error);
}

bool CudaKernels::kv_store(const CudaBuffer& keys, const CudaBuffer& values, const CudaBuffer& key_cache,
                          const CudaBuffer& value_cache, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_ctx,
                          uint32_t position, uint32_t layer, uint32_t n_tokens, const CudaStream& stream, Error& error,
                          uint64_t position_at, bool half_cache)
{
    const size_t cache_bytes = (static_cast<size_t>(layer) + 1) * n_head_kv * n_ctx * head_dim * (half_cache ? 2 : 4);
    const size_t input_bytes = static_cast<size_t>(n_tokens) * n_head_kv * head_dim * 4;
    if (m_device == nullptr || n_head_kv == 0 || head_dim == 0 || n_tokens == 0 || position > n_ctx ||
        n_tokens > n_ctx - position || keys.bytes < input_bytes || values.bytes < input_bytes ||
        key_cache.bytes < cache_bytes || value_cache.bytes < cache_bytes)
    {
        error.set("invalid CUDA KV cache dimensions");
        return false;
    }
    uint64_t addresses[] = {keys.address, values.address, key_cache.address, value_cache.address};
    uint32_t half = half_cache ? 1u : 0u;
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &n_head_kv, &head_dim,
                         &n_ctx, &position, &position_at, &layer, &half};
    CudaLaunch launch;
    launch.grid_x = n_tokens;
    launch.grid_y = n_head_kv;
    launch.block_x = 128;
    return m_device->launch(m_kv_store, stream, launch, arguments, error);
}

constexpr uint32_t kAttentionSplit = 128;

bool CudaKernels::attention(const CudaBuffer& query, const CudaBuffer& key_cache, const CudaBuffer& value_cache,
                           const CudaBuffer& scratch, const CudaBuffer& output, uint32_t n_head, uint32_t n_head_kv,
                           uint32_t head_dim, uint32_t n_ctx, uint32_t position, uint32_t layer, uint32_t n_tokens,
                           const CudaStream& stream, Error& error, uint64_t position_at, bool half_cache,
                           const CudaBuffer* counters)
{
    const size_t cache_bytes = (static_cast<size_t>(layer) + 1) * n_head_kv * n_ctx * head_dim * (half_cache ? 2 : 4);
    const size_t rows_bytes = static_cast<size_t>(n_tokens) * n_head * head_dim * 4;
    if (m_device == nullptr || n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 || head_dim == 0 ||
        n_tokens == 0 || position > n_ctx || n_tokens > n_ctx - position || query.bytes < rows_bytes ||
        output.bytes < rows_bytes || key_cache.bytes < cache_bytes || value_cache.bytes < cache_bytes ||
        scratch.bytes < static_cast<size_t>(n_tokens) * n_head * n_ctx * 4 ||
        (counters != nullptr && counters->bytes < static_cast<size_t>(n_tokens) * n_head * 4))
    {
        error.set("invalid CUDA attention dimensions");
        return false;
    }
    uint64_t addresses[] = {query.address, key_cache.address, value_cache.address, scratch.address, output.address};
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    uint32_t half = half_cache ? 1u : 0u;
    if (n_tokens > 1 && position_at == 0 && (head_dim == 64 || head_dim == 128))
    {
        // a prompt batch: tiles of keys shared by 64 queries
        void* prefill_arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[4], &n_head, &n_head_kv,
                                     &n_ctx, &position, &layer, &scale, &n_tokens, &half};
        CudaLaunch prefill_launch;
        prefill_launch.grid_x = n_head;
        prefill_launch.grid_y = (n_tokens + 63) / 64;
        // head_dim 64: 3xTF32 tensor cores, four warps of 16 queries; 128: CUDA cores
        prefill_launch.block_x = head_dim == 64 ? 128 : 64 * (head_dim / 16);
        return m_device->launch(head_dim == 64 ? m_attention_prefill_tc_64 : m_attention_prefill_128, stream,
                                prefill_launch, prefill_arguments, error);
    }
    // A position read on the device can be anything up to the context, so every split is launched.
    uint32_t splits = (position_at != 0 ? n_ctx : position + n_tokens) + kAttentionSplit - 1;
    splits /= kAttentionSplit;
    if (head_dim % 16 == 0 && head_dim <= 256 &&
        scratch.bytes >= static_cast<size_t>(n_tokens) * n_head * splits * (head_dim + 4) * 4)
    {
        uint64_t counter_address = counters != nullptr ? counters->address : 0;
        void* split_arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3],
                                   &n_head, &n_head_kv, &head_dim, &n_ctx, &position, &position_at, &layer, &scale,
                                   &splits, &half, &counter_address, &addresses[4]};
        CudaLaunch split_launch;
        split_launch.grid_x = n_head;
        split_launch.grid_y = splits;
        split_launch.grid_z = n_tokens;
        split_launch.block_x = kAttentionSplit;
        void* combine_arguments[] = {&addresses[3], &addresses[4], &n_head, &head_dim, &position, &position_at,
                                     &splits};
        CudaLaunch combine_launch;
        combine_launch.grid_x = n_head;
        combine_launch.grid_y = n_tokens;
        combine_launch.block_x = head_dim < 32 ? 32 : (head_dim + 31) / 32 * 32;
        const CudaKernel& split_kernel = head_dim == 64    ? m_attention_split_64
                                         : head_dim == 128 ? m_attention_split_128
                                                           : m_attention_split;
        return m_device->launch(split_kernel, stream, split_launch, split_arguments, error) &&
               (counters != nullptr ||
                m_device->launch(m_attention_combine, stream, combine_launch, combine_arguments, error));
    }
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &addresses[4],
                         &n_head, &n_head_kv, &head_dim, &n_ctx, &position, &position_at, &layer, &scale, &half};
    CudaLaunch launch;
    launch.grid_x = n_head;
    launch.grid_y = n_tokens;
    launch.block_x = 256;
    return m_device->launch(m_attention, stream, launch, arguments, error);
}

bool CudaKernels::swiglu(const CudaBuffer& gate, const CudaBuffer& up, const CudaBuffer& output,
                        uint32_t count, const CudaStream& stream, Error& error)
{
    if (m_device == nullptr || count == 0 || gate.bytes < static_cast<size_t>(count) * 4 ||
        up.bytes < static_cast<size_t>(count) * 4 || output.bytes < static_cast<size_t>(count) * 4)
    {
        error.set("invalid CUDA SwiGLU dimensions");
        return false;
    }
    uint64_t addresses[] = {gate.address, up.address, output.address};
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &count};
    CudaLaunch launch;
    launch.grid_x = (count + 255) / 256;
    launch.block_x = 256;
    return m_device->launch(m_swiglu, stream, launch, arguments, error);
}

bool CudaKernels::add(const CudaBuffer& input, const CudaBuffer& residual, uint32_t count,
                     const CudaStream& stream, Error& error)
{
    if (m_device == nullptr || count == 0 || input.bytes < static_cast<size_t>(count) * 4 ||
        residual.bytes < static_cast<size_t>(count) * 4)
    {
        error.set("invalid CUDA residual dimensions");
        return false;
    }
    uint64_t addresses[] = {input.address, residual.address};
    void* arguments[] = {&addresses[0], &addresses[1], &count};
    CudaLaunch launch;
    launch.grid_x = (count + 255) / 256;
    launch.block_x = 256;
    return m_device->launch(m_add, stream, launch, arguments, error);
}

namespace
{

// block size of the kernels that give each warp several rows
constexpr uint32_t kRows4Threads = 128;

// kernels/matvec_q8_0.cu MatvecSegments
struct MatvecSegments
{
    uint64_t quants[3] = {};
    uint64_t scales[3] = {};
    uint64_t output[3] = {};
    uint32_t rows[3] = {};
};

}

bool CudaKernels::matvec_q8_0(const CudaBuffer& weight, const CudaBuffer& input, const CudaBuffer& output,
                             uint32_t n_in, uint32_t n_out, uint32_t n_tokens, const CudaStream& stream, Error& error,
                             bool accumulate)
{
    return matvec_q8_0_stacked(&weight, &output, &n_out, 1, input, n_in, n_tokens, stream, error, accumulate);
}

bool CudaKernels::matvec_q8_0_stacked(const CudaBuffer* weights, const CudaBuffer* outputs, const uint32_t* rows,
                                     uint32_t count, const CudaBuffer& input, uint32_t n_in, uint32_t n_tokens,
                                     const CudaStream& stream, Error& error, bool accumulate)
{
    if (m_device == nullptr || count == 0 || count > 3 || n_in == 0 || n_in % 32 != 0 || n_tokens == 0 ||
        input.bytes < static_cast<size_t>(n_in) * n_tokens * sizeof(float))
    {
        error.set("invalid CUDA Q8_0 matvec dimensions or buffer size");
        return false;
    }
    MatvecSegments segments;
    uint32_t total = 0;
    for (uint32_t index = 0; index < count; ++index)
    {
        if (rows[index] == 0 || weights[index].bytes < cuda_q8_0_bytes(n_in, rows[index]) ||
            outputs[index].bytes < static_cast<size_t>(rows[index]) * n_tokens * sizeof(float))
        {
            error.set("invalid CUDA Q8_0 matvec dimensions or buffer size");
            return false;
        }
        segments.quants[index] = weights[index].address;
        segments.scales[index] = weights[index].address + static_cast<uint64_t>(n_in) * rows[index];
        segments.output[index] = outputs[index].address;
        segments.rows[index] = rows[index];
        total += rows[index];
    }
    uint64_t input_address = input.address;
    uint32_t add = accumulate ? 1u : 0u;
    void* arguments[] = {&segments, &input_address, &n_in, &n_tokens, &add};
    // four rows per warp when every segment allows it: the input is loaded once for the four
    bool rows4 = true;
    for (uint32_t index = 0; index < count; ++index)
    {
        rows4 = rows4 && rows[index] % 4 == 0;
    }
    CudaLaunch launch;
    launch.block_x = rows4 ? kRows4Threads : 256;
    const uint32_t rows_per_block = rows4 ? kRows4Threads / 32 * 4 : 8;
    launch.grid_x = (total + rows_per_block - 1) / rows_per_block;
    launch.grid_y = (n_tokens + 3) / 4;
    return m_device->launch(rows4 ? m_matvec_q8_0_rows4 : m_matvec_q8_0, stream, launch, arguments, error);
}

bool CudaKernels::matvec_q8_0_swiglu(const CudaBuffer& gate, const CudaBuffer& up, const CudaBuffer& input,
                                    const CudaBuffer& output, uint32_t n_in, uint32_t n_out, uint32_t n_tokens,
                                    const CudaStream& stream, Error& error)
{
    if (m_device == nullptr || n_in == 0 || n_in % 32 != 0 || n_out == 0 || n_tokens == 0 ||
        gate.bytes < cuda_q8_0_bytes(n_in, n_out) || up.bytes < cuda_q8_0_bytes(n_in, n_out) ||
        input.bytes < static_cast<size_t>(n_in) * n_tokens * sizeof(float) ||
        output.bytes < static_cast<size_t>(n_out) * n_tokens * sizeof(float))
    {
        error.set("invalid CUDA Q8_0 swiglu matvec dimensions or buffer size");
        return false;
    }
    const uint64_t scales_offset = static_cast<uint64_t>(n_in) * n_out;
    uint64_t addresses[] = {gate.address, gate.address + scales_offset, up.address, up.address + scales_offset,
                            input.address, output.address};
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &addresses[4], &addresses[5],
                         &n_in, &n_out, &n_tokens};
    const bool rows2 = n_out % 2 == 0;
    CudaLaunch launch;
    launch.block_x = rows2 ? kRows4Threads : 256;
    const uint32_t rows_per_block = rows2 ? kRows4Threads / 32 * 2 : 8;
    launch.grid_x = (n_out + rows_per_block - 1) / rows_per_block;
    launch.grid_y = (n_tokens + 3) / 4;
    return m_device->launch(rows2 ? m_matvec_q8_0_swiglu_rows2 : m_matvec_q8_0_swiglu, stream, launch, arguments, error);
}

bool CudaKernels::quantize_q8_0(const CudaBuffer& input, const CudaBuffer& quants, const CudaBuffer& scales,
                               uint32_t n_in, uint32_t n_tokens, uint32_t token_stride, const CudaStream& stream,
                               Error& error)
{
    if (m_device == nullptr || n_in == 0 || n_in % 32 != 0 || n_tokens == 0 || token_stride < n_tokens ||
        input.bytes < static_cast<size_t>(n_in) * n_tokens * 4 || quants.bytes < static_cast<size_t>(n_in) * n_tokens ||
        scales.bytes < static_cast<size_t>(n_in / 32) * token_stride * 2)
    {
        error.set("invalid CUDA Q8_0 quantization dimensions");
        return false;
    }
    uint64_t addresses[] = {input.address, quants.address, scales.address};
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &n_in, &token_stride};
    CudaLaunch launch;
    launch.grid_x = (n_in / 32 + 3) / 4;
    launch.grid_y = n_tokens;
    launch.block_x = 128;
    return m_device->launch(m_quantize_q8_0, stream, launch, arguments, error);
}

bool CudaKernels::gemm_q8_0(const CudaBuffer& weight, const CudaBuffer& quants, const CudaBuffer& scales,
                           uint32_t token_stride, const CudaBuffer& output, uint32_t n_in, uint32_t n_out,
                           uint32_t n_tokens, const CudaStream& stream, Error& error, bool accumulate)
{
    // the token tiles of 8 read whole tiles of quants; the scales are copied in groups of 8 tokens
    const size_t padded = (static_cast<size_t>(n_tokens) + 7) / 8 * 8;
    if (m_device == nullptr || n_in == 0 || n_in % 32 != 0 || n_out == 0 || n_out % 16 != 0 || n_tokens == 0 ||
        token_stride < padded || token_stride % 8 != 0 || weight.bytes < cuda_q8_0_bytes(n_in, n_out) ||
        quants.bytes < n_in * padded || scales.bytes < static_cast<size_t>(n_in / 32) * token_stride * 2 ||
        output.bytes < static_cast<size_t>(n_out) * n_tokens * 4)
    {
        error.set("invalid CUDA Q8_0 GEMM dimensions");
        return false;
    }
    const uint64_t scales_by_block = weight.address + static_cast<uint64_t>(n_in) * n_out + (n_in / 32) * n_out * 2ull;
    uint64_t addresses[] = {weight.address, scales_by_block, quants.address, scales.address, output.address};
    uint32_t add = accumulate ? 1u : 0u;
    uint32_t padded_tokens = static_cast<uint32_t>(padded);
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &addresses[4],
                         &n_in, &n_out, &n_tokens, &padded_tokens, &token_stride, &add};
    CudaLaunch launch;
    launch.block_x = 128; // four warps of 32 rows by 32 tokens
    launch.grid_x = (n_out + 63) / 64;
    launch.grid_y = (n_tokens + 63) / 64;
    return m_device->launch(m_gemm_q8_0, stream, launch, arguments, error);
}

bool CudaKernels::gemm_f16(const CudaBuffer& input, const CudaBuffer& weight, const CudaBuffer& output, uint32_t m,
                           uint32_t n, uint32_t k, bool bf16, const CudaStream& stream, Error& error, GemmOutput how,
                           const CudaBuffer* gate, uint64_t gate_offset)
{
    const size_t out_size = how == GemmOutput::Half || how == GemmOutput::HalfGelu ? 2 : 4;
    if (m_device == nullptr || m == 0 || n == 0 || n % 8 != 0 || k == 0 || k % 32 != 0 ||
        input.bytes < static_cast<size_t>(m) * k * 2 || weight.bytes < static_cast<size_t>(n) * k * 2 ||
        output.bytes < static_cast<size_t>(m) * n * out_size ||
        (how == GemmOutput::Residual && gate == nullptr) ||
        (gate != nullptr && gate->bytes < gate_offset + n * 4ull))
    {
        error.set("invalid CUDA f16 GEMM dimensions or buffer size");
        return false;
    }
    uint64_t addresses[] = {input.address, weight.address, output.address,
                            gate != nullptr ? gate->address + gate_offset : 0};
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &m, &n, &k};
    CudaLaunch launch;
    launch.grid_x = (n + 127) / 128;
    launch.grid_y = (m + 127) / 128;
    launch.block_x = 128;
    const int t = bf16 ? 1 : 0;
    const CudaKernel& kernel = how == GemmOutput::Float      ? (bf16 ? m_gemm_bf16 : m_gemm_f16)
                               : how == GemmOutput::Half     ? m_gemm_outputs[t][0]
                               : how == GemmOutput::HalfGelu ? m_gemm_outputs[t][1]
                                                             : m_gemm_outputs[t][2];
    return m_device->launch(kernel, stream, launch, arguments, error);
}

bool CudaKernels::dit_modulate_norm(const CudaBuffer& x, const CudaBuffer& modulation, uint64_t shift_offset,
                                    uint64_t scale_offset, const CudaBuffer& output, uint32_t rows, uint32_t width,
                                    bool bf16, const CudaStream& stream, Error& error)
{
    if (m_device == nullptr || rows == 0 || width == 0 || width % 256 != 0 || width > 4096 ||
        x.bytes < static_cast<size_t>(rows) * width * 4 || output.bytes < static_cast<size_t>(rows) * width * 2 ||
        modulation.bytes < shift_offset + width * 4ull || modulation.bytes < scale_offset + width * 4ull)
    {
        error.set("invalid CUDA DiT modulate norm dimensions or buffer size");
        return false;
    }
    uint64_t addresses[] = {x.address, modulation.address + shift_offset, modulation.address + scale_offset,
                            output.address};
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &width};
    CudaLaunch launch;
    launch.grid_x = rows;
    launch.block_x = 256;
    return m_device->launch(m_dit_modulate_norm[bf16 ? 1 : 0], stream, launch, arguments, error);
}

bool CudaKernels::dit_matvec(const CudaBuffer& weight, const CudaBuffer& input, uint64_t input_offset,
                             const CudaBuffer* add, uint64_t add_offset, const CudaBuffer& output,
                             uint64_t output_offset, uint32_t n, uint32_t k, bool silu, const CudaStream& stream,
                             Error& error)
{
    if (m_device == nullptr || n == 0 || k == 0 || k % 256 != 0 || weight.bytes < static_cast<size_t>(n) * k * 2 ||
        input.bytes < input_offset + k * 4ull || output.bytes < output_offset + n * 4ull ||
        (add != nullptr && add->bytes < add_offset + n * 4ull))
    {
        error.set("invalid CUDA DiT matvec dimensions or buffer size");
        return false;
    }
    uint64_t addresses[] = {weight.address, input.address + input_offset, add != nullptr ? add->address + add_offset : 0,
                            output.address + output_offset};
    uint32_t apply = silu ? 1u : 0u;
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &n, &k, &apply};
    CudaLaunch launch;
    launch.grid_x = (n + 7) / 8;
    launch.block_x = 256;
    return m_device->launch(m_dit_matvec, stream, launch, arguments, error);
}

bool CudaKernels::dit_copy(const CudaBuffer& src, uint64_t src_offset, const CudaBuffer& dst, uint64_t dst_offset,
                           uint64_t bytes, const CudaStream& stream, Error& error)
{
    if (m_device == nullptr || bytes % 16 != 0 || (src_offset | dst_offset) % 16 != 0 ||
        src.bytes < src_offset + bytes || dst.bytes < dst_offset + bytes)
    {
        error.set("invalid CUDA DiT copy range");
        return false;
    }
    if (bytes == 0)
    {
        return true;
    }
    uint64_t addresses[] = {src.address + src_offset, dst.address + dst_offset};
    unsigned long long count = bytes / 16;
    void* arguments[] = {&addresses[0], &addresses[1], &count};
    CudaLaunch launch;
    launch.grid_x = static_cast<uint32_t>((count + 255) / 256 < 4096 ? (count + 255) / 256 : 4096);
    launch.block_x = 256;
    return m_device->launch(m_dit_copy, stream, launch, arguments, error);
}

bool CudaKernels::dit_qk_norm_rope(const CudaBuffer& data, uint32_t stride, uint32_t first, const CudaBuffer* gamma,
                                   const CudaBuffer* angles, uint32_t rows, uint32_t heads, bool bf16,
                                   const CudaStream& stream, Error& error)
{
    if (m_device == nullptr || rows == 0 || heads == 0 || first + heads * 128 > stride ||
        data.bytes < static_cast<size_t>(rows) * stride * 2 || (gamma != nullptr && gamma->bytes < 256) ||
        (angles != nullptr && angles->bytes < static_cast<size_t>(rows) * 64 * 4))
    {
        error.set("invalid CUDA DiT q / k norm dimensions or buffer size");
        return false;
    }
    uint64_t addresses[] = {data.address, gamma != nullptr ? gamma->address : 0, angles != nullptr ? angles->address : 0};
    void* arguments[] = {&addresses[0], &stride, &first, &addresses[1], &addresses[2], &rows, &heads};
    CudaLaunch launch;
    launch.grid_x = (rows * heads + 7) / 8;
    launch.block_x = 256;
    return m_device->launch(m_dit_qk_norm_rope[bf16 ? 1 : 0], stream, launch, arguments, error);
}

bool CudaKernels::attention_f16(const CudaBuffer& query, uint32_t q_stride, const CudaBuffer& keys,
                                const CudaBuffer& values, uint32_t kv_stride, const CudaBuffer& output,
                                uint32_t out_stride, uint32_t n_q, uint32_t n_kv, uint32_t heads, float scale,
                                bool bf16, const CudaStream& stream, Error& error, uint32_t kv_heads, bool causal)
{
    const uint32_t width = heads * 128;
    if (kv_heads == 0)
    {
        kv_heads = heads;
    }
    const uint32_t kv_width = kv_heads * 128;
    if (m_device == nullptr || n_q == 0 || n_kv == 0 || heads == 0 || heads % kv_heads != 0 || q_stride < width ||
        kv_stride < kv_width || out_stride < width || q_stride % 8 != 0 || kv_stride % 8 != 0 ||
        out_stride % 2 != 0 || (causal && n_q != n_kv) ||
        query.bytes < (static_cast<size_t>(n_q - 1) * q_stride + width) * 2 ||
        keys.bytes < (static_cast<size_t>(n_kv - 1) * kv_stride + kv_width) * 2 ||
        values.bytes < (static_cast<size_t>(n_kv - 1) * kv_stride + kv_width) * 2 ||
        output.bytes < (static_cast<size_t>(n_q - 1) * out_stride + width) * 2)
    {
        error.set("invalid CUDA f16 attention dimensions or buffer size");
        return false;
    }
    // softmax_scale * M_LOG2E in float, as flash-attention's params.scale_softmax_log2
    float scale_log2 = scale * static_cast<float>(1.4426950408889634);
    uint32_t group = heads / kv_heads;
    uint32_t mask = causal ? 1u : 0u;
    uint64_t addresses[] = {query.address, keys.address, values.address, output.address};
    void* arguments[] = {&addresses[0], &q_stride, &addresses[1], &addresses[2], &kv_stride,
                         &addresses[3], &out_stride, &n_q,          &n_kv,         &scale_log2,
                         &group,        &mask};
    CudaLaunch launch;
    launch.grid_x = (n_q + 127) / 128;
    launch.grid_y = heads;
    launch.block_x = 128;
    return m_device->launch(bf16 ? m_attention_bf16 : m_attention_f16, stream, launch, arguments, error);
}

}
