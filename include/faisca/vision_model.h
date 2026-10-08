#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/gguf.h"
#include "faisca/model.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

struct VisionHparams
{
    uint32_t image_size = 0;
    uint32_t patch_size = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_ff_padded = 0; // n_ff rounded up to the Q8_0 block
    uint32_t n_layer = 0;
    uint32_t n_head = 0;
    uint32_t n_merge = 4; // side of the average pool of the projector
    uint32_t n_proj = 0; // the language model's embedding width
    float eps = 1e-6f;
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3] = {0.5f, 0.5f, 0.5f};

    uint32_t patches_per_side() const
    {
        return image_size / patch_size;
    }

    uint32_t n_patches() const
    {
        return patches_per_side() * patches_per_side();
    }

    uint32_t n_tokens() const
    {
        return (patches_per_side() / n_merge) * (patches_per_side() / n_merge);
    }
};

struct VisionLayer
{
    const float* ln1_w;
    const float* ln1_b;
    Weight q;
    const float* q_b;
    Weight k;
    const float* k_b;
    Weight v;
    const float* v_b;
    Weight out;
    const float* out_b;
    const float* ln2_w;
    const float* ln2_b;
    Weight up;
    const float* up_b;
    Weight down;
    const float* down_b;
};

// the SigLIP encoder and the gemma3 projector of a llama.cpp mmproj GGUF (clip.projector_type
// gemma3): the F16 matrices become packed Q8_0 when quantize is set, everything else float32
class VisionModel
{
public:
    VisionModel() = default;

    VisionModel(const VisionModel&) = delete;
    VisionModel& operator=(const VisionModel&) = delete;

    bool load(const char* path, Error& error, bool quantize = true);

    const VisionHparams& hparams() const
    {
        return m_hparams;
    }

    // [n_embd][3 * patch_size * patch_size], the taps in (channel, row, column) order
    const Weight& patch_embd() const
    {
        return m_patch;
    }

    const float* patch_bias() const
    {
        return m_patch_b;
    }

    // [n_patches][n_embd]
    const float* position_embd() const
    {
        return m_position;
    }

    const VisionLayer& layer(size_t i) const
    {
        return m_layers[i];
    }

    const float* post_ln_w() const
    {
        return m_post_ln_w;
    }

    const float* post_ln_b() const
    {
        return m_post_ln_b;
    }

    const float* soft_emb_norm() const
    {
        return m_soft_emb_norm;
    }

    // [n_proj][n_embd]
    const Weight& projection() const
    {
        return m_projection;
    }

    bool quantized() const
    {
        return m_quantized;
    }

private:
    bool vector(const GgufFile& file, const char* name, size_t count, const float*& out, Error& error);
    bool matrix(const GgufFile& file, const char* name, size_t n_in, size_t n_out, bool transpose, Weight& out, Error& error);
    float* reserve(size_t count);

    VisionHparams m_hparams;
    bool m_quantize = false;
    bool m_quantized = false;
    Array<float> m_storage;
    size_t m_used = 0;
    Array<uint8_t> m_packed;
    size_t m_packed_used = 0;

    Weight m_patch;
    const float* m_patch_b = nullptr;
    const float* m_position = nullptr;
    Array<VisionLayer> m_layers;
    const float* m_post_ln_w = nullptr;
    const float* m_post_ln_b = nullptr;
    const float* m_soft_emb_norm = nullptr;
    Weight m_projection;
};

}
