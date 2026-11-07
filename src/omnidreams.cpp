#include "faisca/omnidreams.h"

#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

const uint16_t* bf16_tensor(const CkptFile& file, const char* name, uint64_t rows, uint64_t columns, Error& error)
{
    const CkptTensor* t = file.find(name);
    if (t == nullptr)
    {
        error.set("OmniDreams checkpoint: no tensor %s", name);
        return nullptr;
    }
    const bool shape = columns == 0 ? t->n_dims == 1 && t->shape[0] == rows
                                    : t->n_dims == 2 && t->shape[0] == rows && t->shape[1] == columns;
    if (t->dtype != CkptDtype::BF16 || !shape)
    {
        error.set("OmniDreams checkpoint: %s is not bf16 of the expected shape", name);
        return nullptr;
    }
    return reinterpret_cast<const uint16_t*>(t->data);
}

}

bool DitCheckpoint::open(const char* path, Error& error)
{
    if (!m_file.open(path, error))
    {
        return false;
    }
    const uint64_t D = DIT_WIDTH;
    DitHostWeights& w = m_weights;
    w = {};
    const uint16_t* x_embedder = bf16_tensor(m_file, "x_embedder.proj.1.weight", D, DIT_PATCH_IN + 4, error);
    const uint16_t* final_linear = bf16_tensor(m_file, "final_layer.linear.weight", DIT_PATCH_OUT, D, error);
    if (x_embedder == nullptr || final_linear == nullptr)
    {
        return false;
    }
    w.hdmap_embedder = bf16_tensor(m_file, "additional_patch_embedding.proj.1.weight", D, DIT_HDMAP_IN, error);
    w.t_linear1 = bf16_tensor(m_file, "t_embedder.1.linear_1.weight", D, D, error);
    w.t_linear2 = bf16_tensor(m_file, "t_embedder.1.linear_2.weight", 3 * D, D, error);
    w.t_norm = bf16_tensor(m_file, "t_embedding_norm.weight", D, 0, error);
    w.final_mod_down = bf16_tensor(m_file, "final_layer.adaln_modulation.1.weight", DIT_LORA, D, error);
    w.final_mod_up = bf16_tensor(m_file, "final_layer.adaln_modulation.2.weight", 2 * D, DIT_LORA, error);
    w.text_proj = bf16_tensor(m_file, "crossattn_proj.0.weight", DIT_TEXT_WIDTH, DIT_TEXT_IN, error);
    w.text_proj_bias = bf16_tensor(m_file, "crossattn_proj.0.bias", DIT_TEXT_WIDTH, 0, error);
    bool ok = w.hdmap_embedder != nullptr && w.t_linear1 != nullptr && w.t_linear2 != nullptr &&
              w.t_norm != nullptr && w.final_mod_down != nullptr && w.final_mod_up != nullptr &&
              w.text_proj != nullptr && w.text_proj_bias != nullptr;
    // the input channels (c kt kh kw) of 16 latents, the condition mask and the padding mask:
    // the padding mask, always zero at inference, is the last 4 columns
    ok = ok && m_x_embedder.resize_uninitialized(D * DIT_PATCH_IN) &&
         m_final_linear.resize_uninitialized(DIT_PATCH_OUT * D);
    if (!ok)
    {
        if (error.message[0] == 0)
        {
            error.set("out of memory");
        }
        return false;
    }
    for (uint64_t r = 0; r < D; ++r)
    {
        std::memcpy(m_x_embedder.data() + r * DIT_PATCH_IN, x_embedder + r * (DIT_PATCH_IN + 4), DIT_PATCH_IN * 2);
    }
    // rearrange "(kt kh kw c) in -> (c kt kh kw) in", kt = 1, kh = kw = 2, c = 16
    for (uint64_t p = 0; p < 4; ++p)
    {
        for (uint64_t c = 0; c < 16; ++c)
        {
            std::memcpy(m_final_linear.data() + (c * 4 + p) * D, final_linear + (p * 16 + c) * D, D * 2);
        }
    }
    w.x_embedder = m_x_embedder.data();
    w.final_linear = m_final_linear.data();

    for (uint32_t b = 0; b < DIT_MAX_BLOCKS; ++b)
    {
        char name[96];
        const auto get = [&](const char* suffix, uint64_t rows, uint64_t columns) {
            std::snprintf(name, sizeof(name), "blocks.%u.%s", b, suffix);
            return bf16_tensor(m_file, name, rows, columns, error);
        };
        DitHostBlock& block = w.blocks[b];
        block.self_q = get("self_attn.q_proj.weight", D, D);
        block.self_k = get("self_attn.k_proj.weight", D, D);
        block.self_v = get("self_attn.v_proj.weight", D, D);
        block.self_out = get("self_attn.output_proj.weight", D, D);
        block.self_q_norm = get("self_attn.q_norm.weight", 128, 0);
        block.self_k_norm = get("self_attn.k_norm.weight", 128, 0);
        block.cross_q = get("cross_attn.q_proj.weight", D, D);
        block.cross_k = get("cross_attn.k_proj.weight", D, DIT_TEXT_WIDTH);
        block.cross_v = get("cross_attn.v_proj.weight", D, DIT_TEXT_WIDTH);
        block.cross_out = get("cross_attn.output_proj.weight", D, D);
        block.cross_q_norm = get("cross_attn.q_norm.weight", 128, 0);
        block.cross_k_norm = get("cross_attn.k_norm.weight", 128, 0);
        block.mlp_up = get("mlp.layer1.weight", DIT_MLP, D);
        block.mlp_down = get("mlp.layer2.weight", D, DIT_MLP);
        const char* branches[] = {"self_attn", "cross_attn", "mlp"};
        char suffix[64];
        for (int i = 0; i < 3; ++i)
        {
            std::snprintf(suffix, sizeof(suffix), "adaln_modulation_%s.1.weight", branches[i]);
            block.mod_down[i] = get(suffix, DIT_LORA, D);
            std::snprintf(suffix, sizeof(suffix), "adaln_modulation_%s.2.weight", branches[i]);
            block.mod_up[i] = get(suffix, 3 * D, DIT_LORA);
        }
        const uint16_t* all[] = {block.self_q,     block.self_k,       block.self_v,       block.self_out,
                                 block.self_q_norm, block.self_k_norm, block.cross_q,      block.cross_k,
                                 block.cross_v,    block.cross_out,    block.cross_q_norm, block.cross_k_norm,
                                 block.mlp_up,     block.mlp_down,     block.mod_down[0],  block.mod_down[1],
                                 block.mod_down[2], block.mod_up[0],   block.mod_up[1],    block.mod_up[2]};
        for (const uint16_t* p : all)
        {
            if (p == nullptr)
            {
                return false;
            }
        }
    }
    w.block_count = DIT_MAX_BLOCKS;
    if (m_file.tensor_count() != 10 + 20 * DIT_MAX_BLOCKS)
    {
        error.set("OmniDreams checkpoint: %zu tensors, expected %u", m_file.tensor_count(), 10 + 20 * DIT_MAX_BLOCKS);
        return false;
    }
    return true;
}

}
