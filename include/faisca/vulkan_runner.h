#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/runner.h"
#include "faisca/vulkan.h"
#include "faisca/vulkan_ops.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// the forward pass of Runner on a Vulkan device: the weights live on the device
// (Q8_0 as scales and quants, everything else as float32), activations and the
// KV cache stay float32, one command buffer per batch of tokens
class VulkanRunner
{
public:
    static constexpr size_t BATCH = 32;

    VulkanRunner() = default;
    ~VulkanRunner();

    VulkanRunner(const VulkanRunner&) = delete;
    VulkanRunner& operator=(const VulkanRunner&) = delete;

    bool init(VulkanDevice& device, const GgufFile& file, uint32_t n_ctx, Error& error);
    void release();

    void reset()
    {
        m_pos = 0;
    }

    void truncate(uint32_t pos)
    {
        m_pos = pos < m_pos ? pos : m_pos;
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
        GgmlType type = GgmlType::F32; // the type the device holds: Q8_0, Q4_0, Q5_0, Q4_K, Q6_K or F32
        VulkanBuffer scales; // Q8_0, Q4_0, Q5_0
        VulkanBuffer quants; // the quants of those, or the whole blocks of Q4_K and Q6_K
        VulkanBuffer f32;
        uint32_t n_in = 0;
        uint32_t n_out = 0;
    };

    struct Layer
    {
        VulkanBuffer attn_norm;
        VulkanBuffer ffn_norm;
        Matrix wq;
        Matrix wk;
        Matrix wv;
        Matrix wo;
        Matrix ffn_gate;
        Matrix ffn_up;
        Matrix ffn_down;
    };

    bool upload_matrix(const GgufTensor* tensor, Matrix& out, Error& error, bool quantized = true);
    bool upload_vector(const GgufTensor* tensor, VulkanBuffer& out, Error& error);
    bool matvec(const Matrix& w, const VulkanBuffer& x, VulkanBuffer& y, uint32_t n_tokens, Error& error);
    bool forward_batch(size_t count, uint32_t pos, float* logits, bool all_logits, Error& error);
    bool trace(const char* name, int layer, const VulkanBuffer& buffer, size_t count, size_t n_tokens, Error& error);
    void destroy_matrix(Matrix& matrix);

    VulkanDevice* m_device = nullptr;
    VulkanKernels m_kernels;
    LlamaHparams m_hparams;
    uint32_t m_n_ctx = 0;
    uint32_t m_pos = 0;
    size_t m_weight_bytes = 0;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;

    Matrix m_token_embd;
    Matrix m_output;
    bool m_output_tied = true;
    VulkanBuffer m_output_norm;
    Array<Layer> m_layers;

    VulkanBuffer m_tokens;
    VulkanBuffer m_x;
    VulkanBuffer m_h;
    VulkanBuffer m_q;
    VulkanBuffer m_k;
    VulkanBuffer m_v;
    VulkanBuffer m_attn;
    VulkanBuffer m_gate;
    VulkanBuffer m_up;
    VulkanBuffer m_act;
    VulkanBuffer m_logits;
    VulkanBuffer m_k_cache;
    VulkanBuffer m_v_cache;
    VulkanBuffer m_scores;
    Array<float> m_staging;
};

}
