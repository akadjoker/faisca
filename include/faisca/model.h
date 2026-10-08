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

// a matrix of n_out rows of n_in values: F32, Q8_0 in the file's own layout, or
// Q8_0 packed in groups of 16 rows (see ops.h)
// what the quantized kernels take as input: Q8_0 blocks of 32 or, for the K-quants,
// Q8_K blocks of 256 (see ops.h), each with its block sums
enum class Activation
{
    F32,
    Q8_0,
    Q8_K,
};

struct Weight
{
    GgmlType type = GgmlType::F32;
    bool packed = false;
    const uint8_t* data = nullptr;
    size_t n_in = 0;
    size_t n_out = 0;

    Activation activation() const
    {
        if (type == GgmlType::F32)
        {
            return Activation::F32;
        }
        return type == GgmlType::Q4_K || type == GgmlType::Q6_K ? Activation::Q8_K : Activation::Q8_0;
    }

    const float* f32() const
    {
        return reinterpret_cast<const float*>(data);
    }

    // bytes of one row in the file's layout, also the bytes per row of the packed form
    size_t row_bytes() const;

    // one row as float32
    void row(size_t index, float* out) const;

    // out[o] for o in [first, last): x is the float32 input, xq its quantized form as
    // activation() says, with its block sums; the quantized forms may be null for F32 weights
    void matvec(const float* x, const uint8_t* xq, const int32_t* x_sums, size_t first, size_t last, float* out) const;

    // the same for n_tokens inputs, input t at x + t * x_stride (xq, x_sums alike), output t at out + t * out_stride
    void matmul(const float* x, size_t x_stride, const uint8_t* xq, size_t xq_stride, const int32_t* x_sums,
                size_t sums_stride, size_t n_tokens, size_t first, size_t last, float* out, size_t out_stride) const;

    // rows per unit of work: results do not depend on how units are split between threads
    size_t row_group() const
    {
        return packed ? 16 : 4;
    }
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

// the llama architecture. F32 tensors are used in place in the mapped file, F16
// tensors are converted to float32 at load. Q8_0 tensors are packed for the VNNI
// kernels into their own memory when the build has them, used in place otherwise,
// and converted to float32 with dequantize set. The GGUF file must stay open.
class LlamaModel
{
public:
    LlamaModel() = default;

    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    bool load(const GgufFile& file, Error& error, bool dequantize = false);

    // the hyperparameters alone, for runners that keep the weights elsewhere
    static bool read_hparams(const GgufFile& file, LlamaHparams& out, Error& error);

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
        return m_storage.size() * sizeof(float) + m_packed_bytes;
    }

    bool packed() const
    {
        return m_packed != nullptr;
    }

    // bytes the forward pass reads per token
    size_t weight_bytes() const;

private:
    bool load_hparams(const GgufFile& file, Error& error);
    const GgufTensor* find(const GgufFile& file, const char* name, int layer, uint64_t n0, uint64_t n1, bool required,
                           Error& error) const;
    bool count_storage(const GgufTensor* tensor, size_t& elements, Error& error) const;
    Weight materialize(const GgufTensor* tensor);
    bool packs(GgmlType type) const;
    size_t packed_bytes(const GgufTensor* tensor) const;

    LlamaHparams m_hparams;
    bool m_dequantize = false;
    bool m_pack = false;
    bool m_pack_k = false;
    Array<float> m_storage;
    uint8_t* m_packed = nullptr;
    size_t m_packed_bytes = 0;
    size_t m_packed_used = 0;
    Array<LlamaLayer> m_layers;
    Weight m_token_embd;
    const float* m_output_norm = nullptr;
    Weight m_output;
};

}
