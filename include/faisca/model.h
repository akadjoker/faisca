#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/gguf.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

struct LlamaHparams
{
    uint32_t n_vocab = 0;
    uint32_t n_embd = 0;
    uint32_t n_layer = 0;
    uint32_t n_ff = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_rot = 0;
    uint32_t n_ctx_train = 0;
    float rms_eps = 1e-5f;
    float rope_freq_base = 10000.0f;
};

// every weight is float32: n_out rows of n_in values
struct LlamaLayer
{
    const float* attn_norm;
    const float* wq;
    const float* wk;
    const float* wv;
    const float* wo;
    const float* ffn_norm;
    const float* ffn_gate;
    const float* ffn_up;
    const float* ffn_down;
};

// the llama architecture in float32. F32 tensors are used in place in the mapped
// file, F16 and Q8_0 tensors are dequantized at load. The GGUF file must stay open.
class LlamaModel
{
public:
    LlamaModel() = default;

    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    bool load(const GgufFile& file, Error& error);

    void release();

    bool is_loaded() const
    {
        return m_token_embd != nullptr;
    }

    const LlamaHparams& hparams() const
    {
        return m_hparams;
    }

    const float* token_embd() const
    {
        return m_token_embd;
    }

    const float* output_norm() const
    {
        return m_output_norm;
    }

    const float* output() const
    {
        return m_output;
    }

    bool tied_output() const
    {
        return m_output == m_token_embd;
    }

    const LlamaLayer& layer(size_t index) const
    {
        return m_layers[index];
    }

    size_t dequantized_bytes() const
    {
        return m_storage.size() * sizeof(float);
    }

private:
    bool load_hparams(const GgufFile& file, Error& error);
    const GgufTensor* find(const GgufFile& file, const char* name, int layer, uint64_t n0, uint64_t n1, bool required,
                           Error& error) const;
    bool count_storage(const GgufTensor* tensor, size_t& elements, Error& error) const;
    const float* materialize(const GgufTensor* tensor);

    LlamaHparams m_hparams;
    Array<float> m_storage;
    Array<LlamaLayer> m_layers;
    const float* m_token_embd = nullptr;
    const float* m_output_norm = nullptr;
    const float* m_output = nullptr;
};

}
