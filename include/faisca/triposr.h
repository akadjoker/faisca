#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/model.h"
#include "faisca/runner.h"
#include "faisca/thread.h"
#include "faisca/torch_ckpt.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

struct TripoVitLayer
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

// stabilityai/TripoSR model.ckpt: DINO ViT-B/16 image encoder, triplane transformer,
// upsampler and NeRF MLP (config.yaml of that release)
class TripoModel
{
public:
    static constexpr uint32_t VIT_DIM = 768;
    static constexpr uint32_t VIT_HEADS = 12;
    static constexpr uint32_t VIT_LAYERS = 12;
    static constexpr uint32_t VIT_FF = 3072;
    static constexpr uint32_t VIT_PATCH = 16;
    static constexpr uint32_t VIT_POS_SIDE = 14;
    static constexpr float VIT_EPS = 1e-12f;
    static constexpr uint32_t IMAGE_SIZE = 512;

    TripoModel() = default;

    TripoModel(const TripoModel&) = delete;
    TripoModel& operator=(const TripoModel&) = delete;

    bool open(const char* path, Error& error);

    const TripoVitLayer& vit_layer(uint32_t index) const
    {
        return m_vit_layers[index];
    }

    const float* vit_cls() const
    {
        return m_vit_cls;
    }

    const float* vit_pos() const
    {
        return m_vit_pos;
    }

    const Weight& vit_patch() const
    {
        return m_vit_patch;
    }

    const float* vit_patch_b() const
    {
        return m_vit_patch_b;
    }

    const float* vit_norm_w() const
    {
        return m_vit_norm_w;
    }

    const float* vit_norm_b() const
    {
        return m_vit_norm_b;
    }

private:
    const float* f32(const char* name, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3, Error& error);
    bool linear(const char* name, uint64_t n_out, uint64_t n_in, Weight& w, const float** bias, Error& error);

    CkptFile m_file;
    TripoVitLayer m_vit_layers[VIT_LAYERS] = {};
    const float* m_vit_cls = nullptr;
    const float* m_vit_pos = nullptr;
    Weight m_vit_patch;
    const float* m_vit_patch_b = nullptr;
    const float* m_vit_norm_w = nullptr;
    const float* m_vit_norm_b = nullptr;
};

// pos is side x side rows of dim values; out gets out_side x out_side rows, resampled as
// torch.nn.functional.interpolate(mode="bicubic", align_corners=False, scale_factor=scale)
// does on the CPU in float32
void tripo_bicubic_rows(const float* pos, uint32_t side, uint32_t dim, double scale, uint32_t out_side, float* out);

class TripoRunner
{
public:
    TripoRunner() = default;

    TripoRunner(const TripoRunner&) = delete;
    TripoRunner& operator=(const TripoRunner&) = delete;

    bool init(const TripoModel& model, ThreadPool& pool, Error& error);

    void set_trace(RunnerTrace callback, void* user)
    {
        m_trace = callback;
        m_trace_user = user;
    }

    // rgb: IMAGE_SIZE x IMAGE_SIZE pixels of R, G, B in [0, 1], as TripoSR's
    // ImagePreprocessor gives them; fills image_tokens()
    bool encode_image(const float* rgb, Error& error);

    // image_token_count() rows of VIT_DIM values: the CLS token, then the patches
    const float* image_tokens() const
    {
        return m_x.data();
    }

    size_t image_token_count() const
    {
        return m_n_tokens;
    }

private:
    void trace(const char* name, int layer, const float* data, size_t count) const;
    void matmul(const Weight& w, const float* bias, const float* x, size_t n_tokens, float* out);
    void norm_rows(const float* x, const float* w, const float* b, float eps, size_t rows, size_t width, float* out);
    void vit_attention(const TripoVitLayer& layer);

    const TripoModel* m_model = nullptr;
    ThreadPool* m_pool = nullptr;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;
    size_t m_n_tokens = 0;

    Array<float> m_pos;
    Array<float> m_patches;
    Array<float> m_x;
    Array<float> m_h;
    Array<float> m_q;
    Array<float> m_k;
    Array<float> m_v;
    Array<float> m_attn;
    Array<float> m_up;
    Array<float> m_scores;
};

}
