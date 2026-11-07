#pragma once

#include "faisca/array.h"
#include "faisca/cuda.h"
#include "faisca/cuda_ops.h"
#include "faisca/omnidreams.h"

namespace faisca
{

// The OmniDreams DiT on the CUDA kernels: bf16 weights, f32 residual stream.

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
    // the large block weights (qkv, out, cross q and out, MLP: 112 MB a block) stay in pinned
    // host memory and are copied block by block into two device slots on a second stream,
    // overlapped with the computation of the block before
    bool stream_weights = false;
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
        // the streamed ones first, in their slot layout
        CudaBuffer qkv, out, cross_q, cross_out, mlp_up, mlp_down;
        CudaBuffer q_norm, k_norm, cross_kv, cross_q_norm, cross_k_norm, mod_down[3], mod_up[3], text_kv, cache;
        DitKvCache kv;
    };

    bool upload(CudaBuffer& buffer, const void* data, size_t bytes, Error& error);
    bool run_block(Block& block, uint32_t index, int64_t chunk_index, Error& error);
    bool run_blocks(int64_t chunk_index, Error& error);
    bool copy_block(uint32_t index, Error& error);
    void point_at_slot(Block& block, uint32_t slot);

    CudaDevice* m_device = nullptr;
    CudaKernels m_kernels;
    CudaStream m_stream;
    CudaStream m_copy_stream;
    CudaEvent m_loaded[2];
    CudaEvent m_free[2];
    CudaHostBuffer m_host_blocks;
    CudaBuffer m_slots[2];
    bool m_streaming = false;
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
