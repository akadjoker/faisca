#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/torch_ckpt.h"

#include <cstdint>

namespace faisca
{

// The OmniDreams DiT (Cosmos-Predict2.5 2B, FlashDreams integrations_v2/omnidreams/impl/transformer
// network.py CosmosDiTNetwork).

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

// single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt (huggingface.co/nvidia/omni-dreams-models):
// the 570 bf16 tensors of the state dict; the weights point into the mapped file except the two
// network.py update_parameters_after_loading_checkpoint changes: x_embedder without its padding
// mask columns (72 -> 68) and final_layer.linear's rows from (kt kh kw c) to (c kt kh kw)
class DitCheckpoint
{
public:
    bool open(const char* path, Error& error);
    const DitHostWeights& weights() const
    {
        return m_weights;
    }

private:
    CkptFile m_file;
    DitHostWeights m_weights = {};
    Array<uint16_t> m_x_embedder;
    Array<uint16_t> m_final_linear;
};

}
