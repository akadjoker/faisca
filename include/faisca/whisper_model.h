#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/mapped_file.h"
#include "faisca/model.h"
#include "faisca/string_map.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

struct WhisperHparams
{
    int32_t n_vocab = 0;
    int32_t n_audio_ctx = 0;
    int32_t n_audio_state = 0;
    int32_t n_audio_head = 0;
    int32_t n_audio_layer = 0;
    int32_t n_text_ctx = 0;
    int32_t n_text_state = 0;
    int32_t n_text_head = 0;
    int32_t n_text_layer = 0;
    int32_t n_mels = 0;
    int32_t ftype = 0;
    float eps = 1e-5f;
};

// the ids whisper.cpp derives from the vocabulary size
struct WhisperTokens
{
    int32_t eot = 50256;
    int32_t sot = 50257;
    int32_t translate = 50357;
    int32_t transcribe = 50358;
    int32_t solm = 50359;
    int32_t prev = 50360;
    int32_t nosp = 50361;
    int32_t no_timestamps = 50362;
    int32_t beg = 50363;
    int32_t n_languages = 0;
    bool multilingual = false;

    int32_t language(int32_t id) const
    {
        return sot + 1 + id;
    }
};

struct WhisperAttention
{
    const float* norm_w = nullptr;
    const float* norm_b = nullptr;
    Weight q;
    const float* q_b = nullptr;
    Weight k;
    Weight v;
    const float* v_b = nullptr;
    Weight out;
    const float* out_b = nullptr;
};

struct WhisperLayer
{
    WhisperAttention attn;
    WhisperAttention cross; // decoder layers only
    const float* mlp_norm_w = nullptr;
    const float* mlp_norm_b = nullptr;
    Weight mlp_0;
    const float* mlp_0_b = nullptr;
    Weight mlp_1;
    const float* mlp_1_b = nullptr;
};

// whisper.cpp's ggml model file (models/convert-pt-to-ggml.py): every tensor converted to
// float32 at load
class WhisperModel
{
public:
    WhisperModel() = default;
    ~WhisperModel();

    WhisperModel(const WhisperModel&) = delete;
    WhisperModel& operator=(const WhisperModel&) = delete;

    // quantize: the matrices are quantized to Q8_0 and packed for the VNNI kernels when this
    // build has them (the file's values are the float32 reference otherwise)
    bool load(const char* path, Error& error, bool quantize = true);

    bool quantized() const
    {
        return m_packed.size() != 0;
    }

    const WhisperHparams& hparams() const
    {
        return m_hparams;
    }

    const WhisperTokens& tokens() const
    {
        return m_tokens;
    }

    // n_mels x 201
    const float* filters() const
    {
        return m_filters.data();
    }

    // bytes of a text token (empty for ids the file does not list)
    const char* token(int32_t id, size_t& length) const;

    // whisper's language codes: "en" is 0, "pt" is 8; -1 when unknown
    static int32_t language_id(const char* code);
    static const char* language_code(int32_t id);

    // encoder
    const float* positional_embedding_encoder() const
    {
        return m_e_pe;
    }

    const float* conv1_w() const
    {
        return m_e_conv_1_w;
    }

    const float* conv1_b() const
    {
        return m_e_conv_1_b;
    }

    const float* conv2_w() const
    {
        return m_e_conv_2_w;
    }

    const float* conv2_b() const
    {
        return m_e_conv_2_b;
    }

    const WhisperLayer& encoder_layer(size_t index) const
    {
        return m_encoder[index];
    }

    const float* encoder_norm_w() const
    {
        return m_e_ln_w;
    }

    const float* encoder_norm_b() const
    {
        return m_e_ln_b;
    }

    // decoder
    const float* positional_embedding_decoder() const
    {
        return m_d_pe;
    }

    // n_vocab x n_text_state, the output matrix (packed when quantized)
    const Weight& token_embedding() const
    {
        return m_d_te;
    }

    // the same rows in float32 for the embedding lookup
    const float* token_embedding_rows() const
    {
        return m_d_te_rows;
    }

    const WhisperLayer& decoder_layer(size_t index) const
    {
        return m_decoder[index];
    }

    const float* decoder_norm_w() const
    {
        return m_d_ln_w;
    }

    const float* decoder_norm_b() const
    {
        return m_d_ln_b;
    }

private:
    struct Entry
    {
        const float* data = nullptr;
        int32_t n_dims = 0;
        int32_t ne[4] = {0, 0, 0, 0};
    };

    bool read_tensors(const uint8_t* data, size_t size, size_t pos, Error& error);
    bool find(const char* name, int32_t ne0, int32_t ne1, int32_t ne2, const float*& out, Error& error) const;
    bool find_weight(const char* name, int32_t n_in, int32_t n_out, Weight& out, Error& error) const;
    bool find_layer(const char* prefix, int layer, bool cross, WhisperLayer& out, Error& error) const;
    bool pack_weights(Error& error);
    void pack_weight(Weight& w, uint8_t*& cursor, Array<uint8_t>& rows) const;

    WhisperHparams m_hparams;
    WhisperTokens m_tokens;
    Array<float> m_filters;
    Array<uint8_t> m_vocab_bytes;
    Array<uint32_t> m_vocab_offsets;
    Array<float> m_storage;
    Array<uint8_t> m_packed;
    Array<Entry> m_entries;
    StringMap m_index;
    Array<WhisperLayer> m_encoder;
    Array<WhisperLayer> m_decoder;
    const float* m_e_pe = nullptr;
    const float* m_e_conv_1_w = nullptr;
    const float* m_e_conv_1_b = nullptr;
    const float* m_e_conv_2_w = nullptr;
    const float* m_e_conv_2_b = nullptr;
    const float* m_e_ln_w = nullptr;
    const float* m_e_ln_b = nullptr;
    const float* m_d_pe = nullptr;
    Weight m_d_te;
    const float* m_d_te_rows = nullptr;
    const float* m_d_ln_w = nullptr;
    const float* m_d_ln_b = nullptr;
};

}
