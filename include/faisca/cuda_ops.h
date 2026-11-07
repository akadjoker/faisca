#pragma once

#include "faisca/cuda.h"

namespace faisca
{

// Q8_0 matrices on the GPU: all int8 quants row by row (n_rows * n_in bytes), then all f16 block
// scales (n_rows * n_in / 32), so a warp reads 16 contiguous quants per lane.
// Packed Q8_0 for the CUDA kernels: quants rows of n_in bytes, f16 scales row by row, then the
// same scales block by block (scales_by_block[block * n_rows + row]).
size_t cuda_q8_0_bytes(uint32_t n_in, uint32_t n_rows);
void cuda_pack_q8_0(const void* blocks, uint32_t n_in, uint32_t n_rows, uint8_t* out);

class CudaKernels
{
public:
    CudaKernels() = default;
    ~CudaKernels();
    CudaKernels(const CudaKernels&) = delete;
    CudaKernels& operator=(const CudaKernels&) = delete;

    bool init(CudaDevice& device, Error& error);
    void release();
    bool matvec_f32(const CudaBuffer& weight, const CudaBuffer& input, const CudaBuffer& output,
                    uint32_t n_in, uint32_t n_out, uint32_t n_tokens, const CudaStream& stream, Error& error);
    // accumulate adds the products to output (the residual connection)
    bool matvec_q8_0(const CudaBuffer& weight, const CudaBuffer& input, const CudaBuffer& output,
                     uint32_t n_in, uint32_t n_out, uint32_t n_tokens, const CudaStream& stream, Error& error,
                     bool accumulate = false);
    // Up to three Q8_0 matrices over one input in one launch, each into its own output.
    bool matvec_q8_0_stacked(const CudaBuffer* weights, const CudaBuffer* outputs, const uint32_t* rows,
                             uint32_t count, const CudaBuffer& input, uint32_t n_in, uint32_t n_tokens,
                             const CudaStream& stream, Error& error, bool accumulate = false);
    // swiglu(gate * input, up * input), the products never stored.
    bool matvec_q8_0_swiglu(const CudaBuffer& gate, const CudaBuffer& up, const CudaBuffer& input,
                            const CudaBuffer& output, uint32_t n_in, uint32_t n_out, uint32_t n_tokens,
                            const CudaStream& stream, Error& error);
    bool embed(const CudaBuffer& weight, const CudaBuffer& tokens, const CudaBuffer& output,
               uint32_t n_embd, uint32_t n_rows, uint32_t n_tokens, bool q8, const CudaStream& stream,
               Error& error);
    bool rms_norm(const CudaBuffer& input, const CudaBuffer& weight, const CudaBuffer& output,
                  uint32_t width, uint32_t n_tokens, float epsilon, const CudaStream& stream, Error& error);
    bool rope(const CudaBuffer& input, const CudaBuffer& factors, bool has_factors, uint32_t n_head,
              uint32_t head_dim, uint32_t n_rot, uint32_t position, float freq_base, uint32_t n_tokens,
              const CudaStream& stream, Error& error, uint64_t position_at = 0);

    // rope on query and keys, then the keys and values into the caches (rope twice and kv_store)
    bool rope_store(const CudaBuffer& query, const CudaBuffer& keys, const CudaBuffer& values,
                    const CudaBuffer& key_cache, const CudaBuffer& value_cache, const CudaBuffer& factors,
                    bool has_factors, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
                    uint32_t n_ctx, uint32_t position, uint32_t layer, float freq_base, uint32_t n_tokens,
                    const CudaStream& stream, Error& error, uint64_t position_at = 0, bool half_cache = false);
    // position_at, when not zero, is the device address of a uint32 read in place of position, so a
    // captured CUDA graph can replay at any position. half_cache: the caches hold f16.
    bool kv_store(const CudaBuffer& keys, const CudaBuffer& values, const CudaBuffer& key_cache,
                  const CudaBuffer& value_cache, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_ctx,
                  uint32_t position, uint32_t layer, uint32_t n_tokens, const CudaStream& stream, Error& error,
                  uint64_t position_at = 0, bool half_cache = false);
    // counters (n_tokens * n_head zeros, left zero) lets the last split block merge the splits
    // instead of a second launch.
    bool attention(const CudaBuffer& query, const CudaBuffer& key_cache, const CudaBuffer& value_cache,
                   const CudaBuffer& scratch, const CudaBuffer& output, uint32_t n_head, uint32_t n_head_kv,
                   uint32_t head_dim, uint32_t n_ctx, uint32_t position, uint32_t layer, uint32_t n_tokens,
                   const CudaStream& stream, Error& error, uint64_t position_at = 0, bool half_cache = false,
                   const CudaBuffer* counters = nullptr);
    bool swiglu(const CudaBuffer& gate, const CudaBuffer& up, const CudaBuffer& output,
                uint32_t count, const CudaStream& stream, Error& error);
    bool add(const CudaBuffer& input, const CudaBuffer& residual, uint32_t count,
             const CudaStream& stream, Error& error);

    // Activations to Q8_0 as the CPU runner quantizes them: quants rows of n_in int8, f16 scales
    // block major (scales[block * token_stride + token]).
    bool quantize_q8_0(const CudaBuffer& input, const CudaBuffer& quants, const CudaBuffer& scales, uint32_t n_in,
                       uint32_t n_tokens, uint32_t token_stride, const CudaStream& stream, Error& error);
    // Packed Q8_0 weights times quantized activations on the int8 tensor cores (quants and
    // scales hold whole tiles of 8 tokens), n_out a multiple of 16; the same bits as the CPU's
    // matmul_q8_0_packed.
    bool gemm_q8_0(const CudaBuffer& weight, const CudaBuffer& quants, const CudaBuffer& scales,
                   uint32_t token_stride, const CudaBuffer& output, uint32_t n_in, uint32_t n_out,
                   uint32_t n_tokens, const CudaStream& stream, Error& error, bool accumulate = false);
    enum class GemmOutput
    {
        Float,    // output: m x n floats
        Half,     // output: m x n halves of the input type
        HalfGelu, // the same through the exact (erf) GELU
        Residual, // output: m x n floats, output += gate[column] * product
    };
    // sum_k input[m][k] * weight[n][k] on the f16 (or bf16) tensor cores with f32 accumulation,
    // stored as `how` says; input and weight hold halves, k a multiple of 32, n a multiple of 8.
    bool gemm_f16(const CudaBuffer& input, const CudaBuffer& weight, const CudaBuffer& output, uint32_t m,
                  uint32_t n, uint32_t k, bool bf16, const CudaStream& stream, Error& error,
                  GemmOutput how = GemmOutput::Float, const CudaBuffer* gate = nullptr, uint64_t gate_offset = 0);
    // DiT: output (rows x width halves) = layer_norm(x) * (1 + scale) + shift, x f32 rows, shift and
    // scale width floats at their buffer offsets in bytes
    bool dit_modulate_norm(const CudaBuffer& x, const CudaBuffer& modulation, uint64_t shift_offset,
                           uint64_t scale_offset, const CudaBuffer& output, uint32_t rows, uint32_t width,
                           bool bf16, const CudaStream& stream, Error& error);
    // DiT: in place RMSNorm (gamma, 128 halves) of the heads of 128 at column first of rows
    // stride halves apart, then RoPE by angles (rows x 64 floats) unless angles is null
    bool dit_qk_norm_rope(const CudaBuffer& data, uint32_t stride, uint32_t first, const CudaBuffer& gamma,
                          const CudaBuffer* angles, uint32_t rows, uint32_t heads, bool bf16,
                          const CudaStream& stream, Error& error);
    // softmax(q k^T * scale) v per head of 128 over f16 (or bf16) rows, no mask, FlashAttention-2's
    // forward kernel; head h at column h * 128 of every row, strides in elements, output f16.
    bool attention_f16(const CudaBuffer& query, uint32_t q_stride, const CudaBuffer& keys, const CudaBuffer& values,
                       uint32_t kv_stride, const CudaBuffer& output, uint32_t out_stride, uint32_t n_q,
                       uint32_t n_kv, uint32_t heads, float scale, bool bf16, const CudaStream& stream,
                       Error& error);

private:
    CudaDevice* m_device = nullptr;
    CudaModule m_matvec_f32_module;
    CudaKernel m_matvec_f32;
    CudaModule m_matvec_q8_0_module;
    CudaKernel m_matvec_q8_0;
    CudaKernel m_matvec_q8_0_swiglu;
    CudaKernel m_matvec_q8_0_rows4;
    CudaKernel m_matvec_q8_0_swiglu_rows2;
    CudaModule m_embed_module;
    CudaKernel m_embed;
    CudaModule m_rms_norm_module;
    CudaKernel m_rms_norm;
    CudaModule m_rope_module;
    CudaKernel m_rope;
    CudaKernel m_rope_store;
    CudaModule m_kv_store_module;
    CudaKernel m_kv_store;
    CudaModule m_attention_module;
    CudaKernel m_attention;
    CudaKernel m_attention_split;
    CudaKernel m_attention_split_64;
    CudaKernel m_attention_split_128;
    CudaKernel m_attention_prefill_64;
    CudaKernel m_attention_prefill_128;
    CudaKernel m_attention_prefill_tc_64;
    CudaKernel m_attention_combine;
    CudaModule m_swiglu_module;
    CudaKernel m_swiglu;
    CudaModule m_add_module;
    CudaKernel m_add;
    CudaModule m_gemm_module;
    CudaKernel m_quantize_q8_0;
    CudaKernel m_gemm_q8_0;
    CudaModule m_gemm_f16_module;
    CudaKernel m_gemm_f16;
    CudaKernel m_gemm_bf16;
    CudaKernel m_gemm_outputs[2][3];
    CudaModule m_dit_module;
    CudaKernel m_dit_modulate_norm[2];
    CudaKernel m_dit_qk_norm_rope[2];
    CudaModule m_attention_f16_module;
    CudaKernel m_attention_f16;
    CudaKernel m_attention_bf16;
};

}