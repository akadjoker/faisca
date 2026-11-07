#pragma once

#include "faisca/array.h"
#include "faisca/cuda.h"
#include "faisca/cuda_ops.h"
#include "faisca/error.h"
#include "faisca/safetensors.h"
#include "faisca/tokenizer.h"

namespace faisca
{

// The Cosmos-Reason1-7B text encoder of Cosmos-Predict 2.5 / OmniDreams on the GPU (flashdreams
// infra/encoder/text/cosmos_reason1.py CosmosReason1TextEncoder, full_concat): the chat template
// with the image-generator system prompt, padded to 512 tokens with the pad token, the Qwen2.5 text
// model (28 layers, GQA 28/4, RoPE theta 1e6), every layer's hidden state (the last one through
// the final norm) normalized to mean 0 and std 1 per token and concatenated: 512 x 100352 bf16.
// bf16 weights read from the Hugging Face safetensors, each layer's 466 MB streamed through two
// device slots while the one before computes; residual stream in f32.
class CosmosTextCudaEncoder
{
public:
    static constexpr uint32_t TOKENS = 512;
    static constexpr uint32_t WIDTH = 3584;
    static constexpr uint32_t LAYERS = 28;
    static constexpr uint32_t OUT_WIDTH = WIDTH * LAYERS;

    CosmosTextCudaEncoder() = default;
    ~CosmosTextCudaEncoder();
    CosmosTextCudaEncoder(const CosmosTextCudaEncoder&) = delete;
    CosmosTextCudaEncoder& operator=(const CosmosTextCudaEncoder&) = delete;

    // model_dir: the snapshot with tokenizer.json and model.safetensors.index.json
    bool init(CudaDevice& device, const char* model_dir, Error& error);
    void release();

    // out gets TOKENS x OUT_WIDTH bf16
    bool encode(const char* prompt, Array<uint16_t>& out, Error& error);

    // the token ids of the last encode (padded to TOKENS), for checks
    const Array<int32_t>& tokens() const
    {
        return m_ids;
    }

private:
    struct Layer
    {
        const uint8_t* q;
        const uint8_t* k;
        const uint8_t* v;
        const uint8_t* o;
        const uint8_t* gate;
        const uint8_t* up;
        const uint8_t* down;
        CudaBuffer bias;         // q | k | v biases, f32
        CudaBuffer norm1, norm2; // bf16
    };

    bool stage(uint32_t layer, Error& error);

    CudaDevice* m_device = nullptr;
    CudaKernels m_kernels;
    CudaStream m_stream, m_copy_stream;
    CudaEvent m_uploaded[2], m_used[2];
    CudaModule m_module;
    CudaKernel m_rms_norm, m_swiglu, m_hidden_out;
    SafetensorsModel m_weights;
    Tokenizer m_tokenizer;
    int32_t m_pad = 0;
    const uint16_t* m_embedding = nullptr;
    Layer m_layers[LAYERS];
    CudaBuffer m_final_norm, m_ones;
    CudaHostBuffer m_staging[2];
    CudaBuffer m_slots[2];
    CudaBuffer m_x, m_h, m_qkv, m_attn, m_gate_up, m_mid, m_angles, m_out;
    Array<int32_t> m_ids;
    Array<float> m_host_x;
};

}
