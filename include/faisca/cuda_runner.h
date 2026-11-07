#pragma once

#include "faisca/cuda_ops.h"
#include "faisca/model.h"
#include "faisca/runner.h"

namespace faisca
{

class CudaRunner
{
public:
    static constexpr size_t BATCH = 512;

    CudaRunner() = default;
    ~CudaRunner();
    CudaRunner(const CudaRunner&) = delete;
    CudaRunner& operator=(const CudaRunner&) = delete;

    bool init(CudaDevice& device, const GgufFile& file, uint32_t n_ctx, Error& error, bool dequantize = false,
              bool kv_f16 = false);
    void release();
    void reset()
    {
        m_pos = 0;
    }
    void truncate(uint32_t position)
    {
        m_pos = position < m_pos ? position : m_pos;
    }
    uint32_t position() const
    {
        return m_pos;
    }
    uint32_t context_size() const
    {
        return m_n_ctx;
    }
    const LlamaHparams& hparams() const
    {
        return m_hparams;
    }
    size_t weight_bytes() const
    {
        return m_weight_bytes;
    }
    void set_trace(RunnerTrace callback, void* user)
    {
        m_trace = callback;
        m_trace_user = user;
    }
    bool forward(const int32_t* tokens, size_t count, Array<float>& logits, Error& error, bool all_logits = false);

private:
    struct Matrix
    {
        CudaBuffer data;
        uint32_t n_in = 0;
        uint32_t n_out = 0;
        bool q8 = false;
    };
    struct Layer
    {
        CudaBuffer attn_norm;
        CudaBuffer ffn_norm;
        Matrix wq;
        Matrix wk;
        Matrix wv;
        Matrix wo;
        Matrix gate;
        Matrix up;
        Matrix down;
    };

    bool upload_vector(const GgufTensor* tensor, CudaBuffer& out, size_t count, Error& error);
    bool upload_matrix(const GgufTensor* tensor, Matrix& out, uint32_t n_in, uint32_t n_out,
                       bool dequantize, Error& error);
    // Several tokens of Q8_0 weights go through quantize_q8_0 and the tensor core GEMM, as the
    // CPU runner quantizes them; one token stays on the float32 matvec.
    bool matvec(const Matrix& matrix, const CudaBuffer& input, const CudaBuffer& output, uint32_t count, Error& error,
                bool accumulate = false);
    bool gemm_ready(const Matrix& matrix, uint32_t count) const
    {
        return count > 1 && matrix.q8 && matrix.n_out % 16 == 0;
    }
    // gemm on the input last given to quantize
    bool quantize(const CudaBuffer& input, uint32_t n_in, uint32_t count, Error& error);
    bool gemm(const Matrix& matrix, const CudaBuffer& output, uint32_t count, Error& error, bool accumulate = false);
    bool forward_batch(uint32_t count, float* logits, bool all_logits, Error& error);
    // Enqueues one batch up to the logits; position_at as in CudaKernels::kv_store.
    bool enqueue_batch(uint32_t count, bool all, uint64_t position_at, Error& error);
    bool forward_decode(uint32_t token, float* logits, Error& error);
    bool trace(const char* name, int layer, const CudaBuffer& buffer, size_t count, uint32_t n_tokens, Error& error);
    void destroy_matrix(Matrix& matrix);

    CudaDevice* m_device = nullptr;
    CudaKernels m_kernels;
    CudaStream m_stream;
    LlamaHparams m_hparams;
    uint32_t m_n_ctx = 0;
    uint32_t m_pos = 0;
    bool m_kv_f16 = false;
    size_t m_weight_bytes = 0;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;
    Array<Layer> m_layers;
    Matrix m_embedding;
    Matrix m_output;
    bool m_output_tied = true;
    CudaBuffer m_output_norm;
    CudaBuffer m_rope_factors;
    bool m_has_rope_factors = false;
    CudaBuffer m_tokens;
    CudaBuffer m_x;
    CudaBuffer m_h;
    CudaBuffer m_q;
    CudaBuffer m_k;
    CudaBuffer m_v;
    CudaBuffer m_attn;
    CudaBuffer m_gate;
    CudaBuffer m_up;
    CudaBuffer m_act;
    CudaBuffer m_logits;
    CudaBuffer m_key_cache;
    CudaBuffer m_value_cache;
    CudaBuffer m_scores;
    CudaBuffer m_attention_counters; // zeros, see CudaKernels::attention
    CudaBuffer m_xq;                 // quantized activations of a batch
    CudaBuffer m_xs;                 // and their f16 scales
    CudaHostBuffer m_host_input; // token ids, then the position at index BATCH
    CudaGraph m_decode_graph;    // one token, captured on first use
    CudaHostBuffer m_host_logits; // the decode graph copies the logits here
    Array<float> m_staging;
};

}