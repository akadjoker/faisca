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

// a matrix of n_out rows of n_in values, F32 or Q8_0, in the file's own layout
struct Weight
{
    GgmlType type = GgmlType::F32;
    const uint8_t* data = nullptr;
    size_t n_in = 0;
    size_t n_out = 0;

    const float* f32() const
    {
        return reinterpret_cast<const float*>(data);
    }

    size_t row_bytes() const;

    // one row as float32
    void row(size_t index, float* out) const;
};

struct LlamaLayer
{
    const float* attn_norm;
    Weight wq;
    Weight wk;
    Weight wv;
    Weight wo;
    const float* ffn_norm;
    Weight ffn_gate;
    Weight ffn_up;
    Weight ffn_down;
};

// the llama architecture. F32 and Q8_0 tensors are used in place in the mapped
// file, F16 tensors are converted to float32 at load. With dequantize set, Q8_0
// is converted to float32 too. The GGUF file must stay open.
class LlamaModel
{
public:
    LlamaModel() = default;

    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    bool load(const GgufFile& file, Error& error, bool dequantize = false);

    void release();

    bool is_loaded() const
    {
        return m_token_embd.data != nullptr;
    }

    const LlamaHparams& hparams() const
    {
        return m_hparams;
    }

    const Weight& token_embd() const
    {
        return m_token_embd;
    }

    const float* output_norm() const
    {
        return m_output_norm;
    }

    const Weight& output() const
    {
        return m_output;
    }

    bool tied_output() const
    {
        return m_output.data == m_token_embd.data;
    }

    const LlamaLayer& layer(size_t index) const
    {
        return m_layers[index];
    }

    size_t converted_bytes() const
    {
        return m_storage.size() * sizeof(float);
    }

    // bytes the forward pass reads per token
    size_t weight_bytes() const;

private:
    bool load_hparams(const GgufFile& file, Error& error);
    const GgufTensor* find(const GgufFile& file, const char* name, int layer, uint64_t n0, uint64_t n1, bool required,
                           Error& error) const;
    bool count_storage(const GgufTensor* tensor, size_t& elements, Error& error) const;
    Weight materialize(const GgufTensor* tensor);

    LlamaHparams m_hparams;
    bool m_dequantize = false;
    Array<float> m_storage;
    Array<LlamaLayer> m_layers;
    Weight m_token_embd;
    const float* m_output_norm = nullptr;
    Weight m_output;
};

}
