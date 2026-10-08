#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/pocket_text.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

struct PocketHparams
{
    uint32_t dim = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t ffn_dim = 0;
    float max_period = 10000.0f;
    uint32_t ldim = 0;
    uint32_t n_bins = 0;
    bool bos_before_voice = false;
    uint32_t head_dim = 0;
    uint32_t head_depth = 0;
    uint32_t n_time_conds = 0;
    uint32_t sample_rate = 0;
    float frame_rate = 0.0f;
    uint32_t n_residual_layers = 0;
    uint32_t ratios[4] = {0, 0, 0, 0};
    uint32_t n_ratios = 0;
    uint32_t dilation_base = 1;
    uint32_t inner_dim = 0;
    uint32_t outer_dim = 0;
    uint32_t tfm_dim = 0;
    uint32_t tfm_n_layers = 0;
    uint32_t tfm_n_heads = 0;
    uint32_t tfm_ffn_dim = 0;
    uint32_t tfm_context = 0;
    float tfm_max_period = 10000.0f;
    int32_t frames_after_eos = -1;
    float temperature = 0.3f;
    uint32_t hop_length = 0;
    uint32_t upsample_stride = 0;
    uint32_t frame_size = 0;
};

// pre LayerNorm, fused qkv, GELU MLP, optional layer scales
struct PocketTfmLayer
{
    const float* norm1_w = nullptr;
    const float* norm1_b = nullptr;
    const float* norm2_w = nullptr;
    const float* norm2_b = nullptr;
    Weight in_proj;
    Weight out_proj;
    Weight linear1;
    Weight linear2;
    const float* ls1 = nullptr;
    const float* ls2 = nullptr;
};

struct PocketTimeEmbed
{
    const float* freqs = nullptr;
    Weight l0;
    const float* l0_b = nullptr;
    Weight l2;
    const float* l2_b = nullptr;
    const float* alpha = nullptr;
};

struct PocketResBlock
{
    const float* ln_w = nullptr;
    const float* ln_b = nullptr;
    Weight mlp0;
    const float* mlp0_b = nullptr;
    Weight mlp2;
    const float* mlp2_b = nullptr;
    Weight ada;
    const float* ada_b = nullptr;
};

// conv kernels w[oc][ic][k]. A transposed conv with kernel 2 stride is stride causal convs
// of 2 taps, one per output phase p: w[p][oc][ic][2], tap 0 on x[t - 1] and tap 1 on x[t],
// out[o][t stride + p] = bias[o] + sum of the taps (k = 2 then)
struct PocketConv
{
    const float* w = nullptr;
    const float* b = nullptr;
    uint32_t k = 0;
    uint32_t stride = 1;
    uint32_t dilation = 1;
    uint32_t ic = 0;
    uint32_t oc = 0;
};

struct PocketResUnit
{
    PocketConv c1;
    PocketConv c2;
};

constexpr size_t POCKET_MAX_RES = 4;

struct PocketStage
{
    PocketConv resample;
    PocketResUnit res[POCKET_MAX_RES];
    uint32_t n_res = 0;
};

// one Pocket TTS language pack (pockettts.cpp's GGUF): flow LM, flow head, Mimi decoder and
// encoder, tokenizer and text rules
class PocketModel
{
public:
    PocketModel() = default;
    ~PocketModel();

    PocketModel(const PocketModel&) = delete;
    PocketModel& operator=(const PocketModel&) = delete;

    // quantize: float32 matrices become packed Q8_0 when this build has the kernels
    bool load(const char* path, Error& error, bool quantize = true);

    const PocketHparams& hparams() const
    {
        return m_hparams;
    }

    const PocketTextRules& text_rules() const
    {
        return m_rules;
    }

    const PocketTokenizer& tokenizer() const
    {
        return m_tokenizer;
    }

    // flow LM
    const Weight& embed() const
    {
        return m_embed;
    }

    const float* embed_rows() const
    {
        return m_embed_rows;
    }

    const Weight& input_linear() const
    {
        return m_input_linear;
    }

    const float* bos_before_voice() const
    {
        return m_bos_before_voice;
    }

    const float* bos_emb() const
    {
        return m_bos_emb;
    }

    const Weight& speaker_proj() const
    {
        return m_speaker_proj;
    }

    const float* emb_mean() const
    {
        return m_emb_mean;
    }

    const float* emb_std() const
    {
        return m_emb_std;
    }

    const float* out_norm_w() const
    {
        return m_out_norm_w;
    }

    const float* out_norm_b() const
    {
        return m_out_norm_b;
    }

    const float* out_eos_w() const
    {
        return m_out_eos_w;
    }

    float out_eos_b() const
    {
        return m_out_eos_b;
    }

    const PocketTfmLayer& layer(size_t i) const
    {
        return m_layers[i];
    }

    // flow head
    const Weight& cond() const
    {
        return m_cond;
    }

    const float* cond_b() const
    {
        return m_cond_b;
    }

    const Weight& head_in() const
    {
        return m_in;
    }

    const float* head_in_b() const
    {
        return m_in_b;
    }

    const PocketTimeEmbed& time_embed(size_t i) const
    {
        return m_temb[i];
    }

    const PocketResBlock& res_block(size_t i) const
    {
        return m_blocks[i];
    }

    const Weight& final_ada() const
    {
        return m_final_ada;
    }

    const float* final_ada_b() const
    {
        return m_final_ada_b;
    }

    const Weight& final_linear() const
    {
        return m_final;
    }

    const float* final_b() const
    {
        return m_final_b;
    }

    // Mimi decoder
    const Weight& quant_proj() const
    {
        return m_quant_proj;
    }

    // [outer_dim][K]
    const float* upsample_w() const
    {
        return m_upsample_w;
    }

    uint32_t upsample_k() const
    {
        return m_upsample_k;
    }

    const PocketTfmLayer& decoder_layer(size_t i) const
    {
        return m_dec_layers[i];
    }

    const PocketConv& decoder_in() const
    {
        return m_dec_in;
    }

    const PocketStage& decoder_stage(size_t i) const
    {
        return m_dec_stages[i];
    }

    const PocketConv& decoder_out() const
    {
        return m_dec_out;
    }

    // Mimi encoder
    const PocketConv& encoder_in() const
    {
        return m_enc_in;
    }

    const PocketStage& encoder_stage(size_t i) const
    {
        return m_enc_stages[i];
    }

    const PocketConv& encoder_out() const
    {
        return m_enc_out;
    }

    const PocketTfmLayer& encoder_layer(size_t i) const
    {
        return m_enc_layers[i];
    }

    const PocketConv& downsample() const
    {
        return m_downsample;
    }

private:
    bool read_hparams(const GgufFile& file, Error& error);
    bool read_rules(const GgufFile& file, Error& error);
    bool vector(const GgufFile& file, const char* name, size_t count, const float*& out, Error& error, bool required = true);
    bool matrix(const GgufFile& file, const char* name, size_t n_in, size_t n_out, Weight& out, Error& error, const float** rows = nullptr);
    bool conv(const GgufFile& file, const char* prefix, uint32_t stride, uint32_t dilation, bool bias, PocketConv& out, Error& error);
    bool convtr(const GgufFile& file, const char* prefix, uint32_t stride, PocketConv& out, Error& error);
    bool tfm_layer(const GgufFile& file, const char* prefix, bool layer_scale, PocketTfmLayer& out, Error& error);
    bool res_unit(const GgufFile& file, const char* prefix, uint32_t dilation, PocketResUnit& out, Error& error);
    float* reserve(size_t count);

    PocketHparams m_hparams;
    PocketTextRules m_rules;
    PocketTokenizer m_tokenizer;
    bool m_quantize = false;
    Array<float> m_storage;
    size_t m_used = 0;
    Array<uint8_t> m_packed;
    size_t m_packed_used = 0;

    Weight m_embed;
    const float* m_embed_rows = nullptr;
    Weight m_input_linear;
    const float* m_bos_before_voice = nullptr;
    const float* m_bos_emb = nullptr;
    Weight m_speaker_proj;
    const float* m_emb_mean = nullptr;
    const float* m_emb_std = nullptr;
    const float* m_out_norm_w = nullptr;
    const float* m_out_norm_b = nullptr;
    const float* m_out_eos_w = nullptr;
    float m_out_eos_b = 0.0f;
    Array<PocketTfmLayer> m_layers;
    Weight m_cond;
    const float* m_cond_b = nullptr;
    Weight m_in;
    const float* m_in_b = nullptr;
    PocketTimeEmbed m_temb[2];
    Array<PocketResBlock> m_blocks;
    Weight m_final_ada;
    const float* m_final_ada_b = nullptr;
    Weight m_final;
    const float* m_final_b = nullptr;
    Weight m_quant_proj;
    const float* m_upsample_w = nullptr;
    uint32_t m_upsample_k = 0;
    Array<PocketTfmLayer> m_dec_layers;
    PocketConv m_dec_in;
    Array<PocketStage> m_dec_stages;
    PocketConv m_dec_out;
    PocketConv m_enc_in;
    Array<PocketStage> m_enc_stages;
    PocketConv m_enc_out;
    Array<PocketTfmLayer> m_enc_layers;
    PocketConv m_downsample;
};

}
