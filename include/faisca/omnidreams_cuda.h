#pragma once

#include "faisca/array.h"
#include "faisca/cuda.h"
#include "faisca/cuda_ops.h"

namespace faisca
{

// The OmniDreams DiT (Cosmos-Predict2.5 2B, FlashDreams integrations_v2/omnidreams/impl/transformer
// network.py CosmosDiTNetwork.forward) on the CUDA kernels: bf16 weights, f32 residual stream.

constexpr uint32_t DIT_WIDTH = 2048;
constexpr uint32_t DIT_HEADS = 16;
constexpr uint32_t DIT_MLP = 8192;
constexpr uint32_t DIT_LORA = 256;
constexpr uint32_t DIT_TEXT_TOKENS = 512;
constexpr uint32_t DIT_TEXT_IN = 100352;
constexpr uint32_t DIT_TEXT_WIDTH = 1024;
constexpr uint32_t DIT_PATCH_IN = 68;  // 16 latent channels and the condition mask, 2 x 2 patches
constexpr uint32_t DIT_HDMAP_IN = 64;  // 16 HD map latent channels, 2 x 2 patches
constexpr uint32_t DIT_PATCH_OUT = 64; // 16 channels, 2 x 2 patches
constexpr uint32_t DIT_MAX_BLOCKS = 28;

// FlashDreams core/attention/kvcache.py BlockKVCache, the bookkeeping only: [sink | window]
// along the tokens, one chunk written per new chunk index, the window rolled left once full,
// the same chunk index again rewrites the same positions.
class DitKvCache
{
public:
    void init(uint32_t chunk, uint32_t window, uint32_t sink);
    void reset();
    // shift > 0: rows [sink + shift, sink + shift + keep) move to [sink, sink + keep) first
    bool before_update(int64_t chunk_index, uint32_t& shift, uint32_t& keep, Error& error);
    // rows [read_start, read_end) of the chunk go to cache rows [write_start, write_end)
    void write_bounds(uint32_t& write_start, uint32_t& write_end, uint32_t& read_start, uint32_t& read_end) const;
    uint32_t visible_end() const;
    void after_update();
    uint32_t total() const
    {
        return m_sink + m_window;
    }

private:
    bool overlaps_sink() const;

    uint32_t m_chunk = 0;
    uint32_t m_window = 0;
    uint32_t m_sink = 0;
    int64_t m_previous = -1;
    int64_t m_current = -1;
    bool m_updating = false;
    uint32_t m_cached = 0;
};

// bf16 weights on the host, as the checkpoint names them (the padding mask columns of
// x_embedder dropped and final_layer.linear's rows shuffled, network.py
// update_parameters_after_loading_checkpoint)
struct DitHostBlock
{
    const uint16_t* self_q;       // 2048 x 2048
    const uint16_t* self_k;       // 2048 x 2048
    const uint16_t* self_v;       // 2048 x 2048
    const uint16_t* self_q_norm;  // 128
    const uint16_t* self_k_norm;  // 128
    const uint16_t* self_out;     // 2048 x 2048
    const uint16_t* cross_q;      // 2048 x 2048
    const uint16_t* cross_k;      // 2048 x 1024
    const uint16_t* cross_v;      // 2048 x 1024
    const uint16_t* cross_q_norm; // 128
    const uint16_t* cross_k_norm; // 128
    const uint16_t* cross_out;    // 2048 x 2048
    const uint16_t* mlp_up;       // 8192 x 2048
    const uint16_t* mlp_down;     // 2048 x 8192
    const uint16_t* mod_down[3];  // 256 x 2048, self attention, cross attention, MLP
    const uint16_t* mod_up[3];    // 6144 x 256
};

struct DitHostWeights
{
    const uint16_t* x_embedder;     // 2048 x 68
    const uint16_t* hdmap_embedder; // 2048 x 64, or null without HD map
    const uint16_t* t_linear1;      // 2048 x 2048
    const uint16_t* t_linear2;      // 6144 x 2048
    const uint16_t* t_norm;         // 2048
    const uint16_t* final_mod_down; // 256 x 2048
    const uint16_t* final_mod_up;   // 4096 x 256
    const uint16_t* final_linear;   // 64 x 2048
    const uint16_t* text_proj;      // 1024 x 100352
    const uint16_t* text_proj_bias; // 1024
    DitHostBlock blocks[DIT_MAX_BLOCKS];
    uint32_t block_count;
};

struct DitConfig
{
    uint32_t frames = 2;  // latent frames per chunk
    uint32_t height = 44; // patches
    uint32_t width = 80;
    uint32_t window_chunks = 3; // self attention window, in chunks
    uint32_t sink = 0;          // tokens never evicted
    uint32_t short_history_blocks = 0; // first blocks whose window is one chunk
    float h_extrapolation = 3.0f;
    float w_extrapolation = 3.0f;
    float t_extrapolation = 1.0f;
};

class DitCudaRunner
{
public:
    DitCudaRunner() = default;
    ~DitCudaRunner();
    DitCudaRunner(const DitCudaRunner&) = delete;
    DitCudaRunner& operator=(const DitCudaRunner&) = delete;

    bool init(CudaDevice& device, const DitHostWeights& weights, const DitConfig& config, Error& error);
    void release();

    uint32_t chunk_tokens() const
    {
        return m_chunk;
    }

    // the text embeddings (512 x 100352 bf16): projected once, every block's cross attention
    // keys and values (Block.initialize_cache, network.py replace_text_embeddings)
    bool set_text(const uint16_t* text, Error& error);
    // one pass over a chunk: tokens chunk_tokens() x 68 bf16 (patches, (c kt kh kw), then the
    // condition mask), hdmap chunk_tokens() x 64 bf16 or null, timestep before the 0.001 scale;
    // out gets chunk_tokens() x 64 floats
    bool forward(const uint16_t* tokens, const uint16_t* hdmap, float timestep, int64_t chunk_index, float* out,
                 Error& error);
    void reset_cache();

private:
    struct Block
    {
        CudaBuffer qkv, q_norm, k_norm, out, cross_q, cross_kv, cross_q_norm, cross_k_norm, cross_out, mlp_up,
            mlp_down, mod_down[3], mod_up[3], text_kv, cache;
        DitKvCache kv;
    };

    bool upload(CudaBuffer& buffer, const void* data, size_t bytes, Error& error);
    bool run_block(Block& block, uint32_t index, int64_t chunk_index, Error& error);

    CudaDevice* m_device = nullptr;
    CudaKernels m_kernels;
    CudaStream m_stream;
    DitConfig m_config;
    uint32_t m_chunk = 0;
    uint32_t m_block_count = 0;
    bool m_hdmap = false;
    Block m_blocks[DIT_MAX_BLOCKS];
    CudaBuffer m_embedder, m_t_linear1, m_t_linear2, m_final_mod_down, m_final_mod_up, m_final_linear, m_text_proj,
        m_text_proj_bias;
    CudaBuffer m_input, m_x, m_h, m_q, m_attn, m_hidden, m_angles, m_vectors, m_modulation, m_text, m_text_in,
        m_output;
    Array<float> m_t_norm;
    Array<float> m_raw_t, m_raw_h, m_raw_w;
    Array<uint16_t> m_staging;
    Array<float> m_host_vectors;
};

}
