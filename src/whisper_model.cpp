#include "faisca/whisper_model.h"

#include "faisca/ops.h"

#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

constexpr uint32_t GGML_MAGIC = 0x67676D6C;

const char* const LANGUAGES[] = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
};
constexpr int32_t N_LANGUAGES = static_cast<int32_t>(sizeof(LANGUAGES) / sizeof(LANGUAGES[0]));

struct Reader
{
    const uint8_t* data;
    size_t size;
    size_t pos = 0;

    bool read_i32(int32_t& value)
    {
        if (pos + 4 > size)
        {
            return false;
        }
        std::memcpy(&value, data + pos, 4);
        pos += 4;
        return true;
    }

    bool skip(size_t bytes)
    {
        if (pos + bytes > size)
        {
            return false;
        }
        pos += bytes;
        return true;
    }
};

}

WhisperModel::~WhisperModel() = default;

int32_t WhisperModel::language_id(const char* code)
{
    for (int32_t i = 0; i < N_LANGUAGES; ++i)
    {
        if (std::strcmp(LANGUAGES[i], code) == 0)
        {
            return i;
        }
    }
    return -1;
}

const char* WhisperModel::language_code(int32_t id)
{
    return id >= 0 && id < N_LANGUAGES ? LANGUAGES[id] : "";
}

const char* WhisperModel::token(int32_t id, size_t& length) const
{
    if (id < 0 || static_cast<size_t>(id) + 1 >= m_vocab_offsets.size())
    {
        length = 0;
        return "";
    }
    const uint32_t start = m_vocab_offsets[static_cast<size_t>(id)];
    length = m_vocab_offsets[static_cast<size_t>(id) + 1] - start;
    return reinterpret_cast<const char*>(m_vocab_bytes.data() + start);
}

bool WhisperModel::load(const char* path, Error& error, bool quantize)
{
    MappedFile file;
    if (!file.open(path, error))
    {
        return false;
    }
    Reader reader{file.data(), file.size()};
    int32_t magic = 0;
    if (!reader.read_i32(magic) || static_cast<uint32_t>(magic) != GGML_MAGIC)
    {
        error.set("'%s' is not a whisper.cpp ggml model (bad magic)", path);
        return false;
    }
    int32_t* fields[] = {&m_hparams.n_vocab,  &m_hparams.n_audio_ctx,  &m_hparams.n_audio_state, &m_hparams.n_audio_head,
                         &m_hparams.n_audio_layer, &m_hparams.n_text_ctx, &m_hparams.n_text_state, &m_hparams.n_text_head,
                         &m_hparams.n_text_layer, &m_hparams.n_mels, &m_hparams.ftype};
    for (int32_t* field : fields)
    {
        if (!reader.read_i32(*field))
        {
            error.set("'%s': header is truncated", path);
            return false;
        }
    }
    const WhisperHparams& h = m_hparams;
    if (h.n_vocab <= 0 || h.n_audio_ctx <= 0 || h.n_audio_state <= 0 || h.n_audio_head <= 0 || h.n_audio_layer <= 0 ||
        h.n_text_ctx <= 0 || h.n_text_state <= 0 || h.n_text_head <= 0 || h.n_text_layer <= 0 || h.n_mels <= 0 ||
        h.n_audio_state != h.n_text_state || h.n_audio_state % h.n_audio_head != 0 || h.n_text_state % h.n_text_head != 0)
    {
        error.set("'%s': unsupported hparams", path);
        return false;
    }
    if (h.ftype != 0 && h.ftype != 1)
    {
        error.set("'%s': ftype %d (only float32 and f16 files are read)", path, h.ftype);
        return false;
    }

    int32_t n_mel = 0;
    int32_t n_fft = 0;
    if (!reader.read_i32(n_mel) || !reader.read_i32(n_fft) || n_mel != h.n_mels || n_fft != 201)
    {
        error.set("'%s': mel filters are %d x %d, expected %d x 201", path, n_mel, n_fft, h.n_mels);
        return false;
    }
    const size_t filter_count = static_cast<size_t>(n_mel) * static_cast<size_t>(n_fft);
    if (reader.pos + filter_count * 4 > reader.size || !m_filters.resize_uninitialized(filter_count))
    {
        error.set("'%s': mel filters are truncated", path);
        return false;
    }
    std::memcpy(m_filters.data(), reader.data + reader.pos, filter_count * 4);
    reader.pos += filter_count * 4;

    int32_t n_vocab_file = 0;
    if (!reader.read_i32(n_vocab_file) || n_vocab_file <= 0 || !m_vocab_offsets.reserve(static_cast<size_t>(n_vocab_file) + 1))
    {
        error.set("'%s': vocabulary is truncated", path);
        return false;
    }
    m_vocab_offsets.push_back_unchecked(0);
    for (int32_t i = 0; i < n_vocab_file; ++i)
    {
        int32_t length = 0;
        if (!reader.read_i32(length) || length < 0 || reader.pos + static_cast<size_t>(length) > reader.size)
        {
            error.set("'%s': vocabulary is truncated at token %d", path, i);
            return false;
        }
        if (!m_vocab_bytes.append(reader.data + reader.pos, static_cast<size_t>(length)))
        {
            error.set("out of memory");
            return false;
        }
        reader.pos += static_cast<size_t>(length);
        m_vocab_offsets.push_back_unchecked(static_cast<uint32_t>(m_vocab_bytes.size()));
    }

    m_tokens = WhisperTokens();
    m_tokens.multilingual = h.n_vocab >= 51865;
    m_tokens.n_languages = h.n_vocab - 51765 - (m_tokens.multilingual ? 1 : 0);
    if (m_tokens.multilingual)
    {
        m_tokens.eot++;
        m_tokens.sot++;
        const int32_t dt = m_tokens.n_languages - 98;
        m_tokens.translate += dt;
        m_tokens.transcribe += dt;
        m_tokens.solm += dt;
        m_tokens.prev += dt;
        m_tokens.nosp += dt;
        m_tokens.no_timestamps += dt;
        m_tokens.beg += dt;
    }

    if (!read_tensors(reader.data, reader.size, reader.pos, error))
    {
        return false;
    }

    const int32_t n_state = h.n_audio_state;
    bool ok = find("encoder.positional_embedding", n_state, h.n_audio_ctx, 0, m_e_pe, error) &&
              find("encoder.conv1.weight", 3, h.n_mels, n_state, m_e_conv_1_w, error) &&
              find("encoder.conv1.bias", n_state, 0, 0, m_e_conv_1_b, error) &&
              find("encoder.conv2.weight", 3, n_state, n_state, m_e_conv_2_w, error) &&
              find("encoder.conv2.bias", n_state, 0, 0, m_e_conv_2_b, error) &&
              find("encoder.ln_post.weight", n_state, 0, 0, m_e_ln_w, error) &&
              find("encoder.ln_post.bias", n_state, 0, 0, m_e_ln_b, error) &&
              find("decoder.positional_embedding", n_state, h.n_text_ctx, 0, m_d_pe, error) &&
              find_weight("decoder.token_embedding.weight", n_state, h.n_vocab, m_d_te, error) &&
              find("decoder.ln.weight", n_state, 0, 0, m_d_ln_w, error) &&
              find("decoder.ln.bias", n_state, 0, 0, m_d_ln_b, error);
    if (!ok)
    {
        return false;
    }
    if (!m_encoder.resize_uninitialized(static_cast<size_t>(h.n_audio_layer)) ||
        !m_decoder.resize_uninitialized(static_cast<size_t>(h.n_text_layer)))
    {
        error.set("out of memory");
        return false;
    }
    for (int32_t i = 0; i < h.n_audio_layer; ++i)
    {
        m_encoder[static_cast<size_t>(i)] = WhisperLayer();
        if (!find_layer("encoder", i, false, m_encoder[static_cast<size_t>(i)], error))
        {
            return false;
        }
    }
    for (int32_t i = 0; i < h.n_text_layer; ++i)
    {
        m_decoder[static_cast<size_t>(i)] = WhisperLayer();
        if (!find_layer("decoder", i, true, m_decoder[static_cast<size_t>(i)], error))
        {
            return false;
        }
    }
    m_d_te_rows = m_d_te.f32();
    if (quantize && q8_0_packed_supported())
    {
        return pack_weights(error);
    }
    return true;
}

void WhisperModel::pack_weight(Weight& w, uint8_t*& cursor, Array<uint8_t>& rows) const
{
    const size_t row_bytes = w.n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
    const float* source = w.f32();
    for (size_t r = 0; r < w.n_out; ++r)
    {
        quantize_q8_0(source + r * w.n_in, w.n_in, rows.data() + r * row_bytes);
    }
    pack_q8_0(rows.data(), w.n_in, w.n_out, cursor);
    w.type = GgmlType::Q8_0;
    w.packed = true;
    w.data = cursor;
    cursor += q8_0_packed_bytes(w.n_in, w.n_out);
}

bool WhisperModel::pack_weights(Error& error)
{
    Weight* weights[8 * 64 + 1];
    size_t count = 0;
    const auto collect = [&](WhisperAttention& a)
    {
        weights[count++] = &a.q;
        weights[count++] = &a.k;
        weights[count++] = &a.v;
        weights[count++] = &a.out;
    };
    for (size_t i = 0; i < m_encoder.size(); ++i)
    {
        collect(m_encoder[i].attn);
        weights[count++] = &m_encoder[i].mlp_0;
        weights[count++] = &m_encoder[i].mlp_1;
    }
    for (size_t i = 0; i < m_decoder.size(); ++i)
    {
        collect(m_decoder[i].attn);
        collect(m_decoder[i].cross);
        weights[count++] = &m_decoder[i].mlp_0;
        weights[count++] = &m_decoder[i].mlp_1;
    }
    weights[count++] = &m_d_te;

    size_t total = 0;
    size_t widest = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (weights[i]->n_in % Q8_0_BLOCK != 0)
        {
            error.set("matrix width %zu is not a multiple of 32", weights[i]->n_in);
            return false;
        }
        total += q8_0_packed_bytes(weights[i]->n_in, weights[i]->n_out);
        const size_t bytes = weights[i]->n_out * (weights[i]->n_in / Q8_0_BLOCK * Q8_0_BLOCK_BYTES);
        widest = bytes > widest ? bytes : widest;
    }
    Array<uint8_t> rows;
    if (!m_packed.resize_uninitialized(total) || !rows.resize_uninitialized(widest))
    {
        error.set("out of memory for the packed weights");
        return false;
    }
    uint8_t* cursor = m_packed.data();
    for (size_t i = 0; i < count; ++i)
    {
        pack_weight(*weights[i], cursor, rows);
    }
    return true;
}

bool WhisperModel::read_tensors(const uint8_t* data, size_t size, size_t pos, Error& error)
{
    // first pass: sizes, so one allocation holds every converted tensor
    size_t total = 0;
    size_t count = 0;
    {
        Reader reader{data, size, pos};
        while (reader.pos < reader.size)
        {
            int32_t n_dims = 0;
            int32_t name_length = 0;
            int32_t ftype = 0;
            if (!reader.read_i32(n_dims) || !reader.read_i32(name_length) || !reader.read_i32(ftype) || n_dims < 1 ||
                n_dims > 4 || name_length <= 0 || (ftype != 0 && ftype != 1))
            {
                error.set("tensor %zu: bad header", count);
                return false;
            }
            size_t n = 1;
            for (int32_t i = 0; i < n_dims; ++i)
            {
                int32_t ne = 0;
                if (!reader.read_i32(ne) || ne <= 0)
                {
                    error.set("tensor %zu: bad dimension", count);
                    return false;
                }
                n *= static_cast<size_t>(ne);
            }
            if (!reader.skip(static_cast<size_t>(name_length)) || !reader.skip(n * (ftype == 1 ? 2 : 4)))
            {
                error.set("tensor %zu: truncated", count);
                return false;
            }
            total += n;
            count++;
        }
    }
    if (!m_storage.resize_uninitialized(total) || !m_entries.reserve(count) || !m_index.reserve(count))
    {
        error.set("out of memory");
        return false;
    }

    Reader reader{data, size, pos};
    size_t used = 0;
    for (size_t t = 0; t < count; ++t)
    {
        Entry entry;
        int32_t name_length = 0;
        int32_t ftype = 0;
        if (!reader.read_i32(entry.n_dims) || !reader.read_i32(name_length) || !reader.read_i32(ftype))
        {
            error.set("tensor %zu: bad header", t);
            return false;
        }
        size_t n = 1;
        for (int32_t i = 0; i < entry.n_dims && i < 4; ++i)
        {
            if (!reader.read_i32(entry.ne[i]))
            {
                error.set("tensor %zu: bad dimension", t);
                return false;
            }
            n *= static_cast<size_t>(entry.ne[i]);
        }
        const char* name = reinterpret_cast<const char*>(reader.data + reader.pos);
        reader.pos += static_cast<size_t>(name_length);
        float* out = m_storage.data() + used;
        if (ftype == 1)
        {
            const uint8_t* half = reader.data + reader.pos;
            for (size_t i = 0; i < n; ++i)
            {
                uint16_t bits = 0;
                std::memcpy(&bits, half + 2 * i, 2);
                out[i] = f16_to_f32(bits);
            }
            reader.pos += n * 2;
        }
        else
        {
            std::memcpy(out, reader.data + reader.pos, n * 4);
            reader.pos += n * 4;
        }
        entry.data = out;
        used += n;
        const StringMap::Insert insert = m_index.insert(std::string_view(name, static_cast<size_t>(name_length)), static_cast<uint32_t>(m_entries.size()));
        if (insert == StringMap::Insert::Exists)
        {
            error.set("tensor '%.*s' appears twice", name_length, name);
            return false;
        }
        if (insert != StringMap::Insert::Added)
        {
            error.set("out of memory");
            return false;
        }
        m_entries.push_back_unchecked(entry);
    }
    return true;
}

bool WhisperModel::find(const char* name, int32_t ne0, int32_t ne1, int32_t ne2, const float*& out, Error& error) const
{
    uint32_t index = 0;
    if (!m_index.find(std::string_view(name), index))
    {
        error.set("tensor '%s' is missing", name);
        return false;
    }
    const Entry& entry = m_entries[index];
    const int32_t expected[3] = {ne0, ne1, ne2};
    int32_t n_dims = ne2 != 0 ? 3 : (ne1 != 0 ? 2 : 1);
    bool ok = entry.n_dims == n_dims;
    for (int32_t i = 0; ok && i < n_dims; ++i)
    {
        ok = entry.ne[i] == expected[i];
    }
    if (!ok && n_dims == 1)
    {
        // the conv biases are stored as 1 x n
        int64_t count = 1;
        for (int32_t i = 0; i < entry.n_dims; ++i)
        {
            count *= entry.ne[i];
        }
        ok = count == ne0;
    }
    if (!ok)
    {
        error.set("tensor '%s' has shape %d x %d x %d, expected %d x %d x %d", name, entry.ne[0], entry.ne[1], entry.ne[2], ne0, ne1, ne2);
        return false;
    }
    out = entry.data;
    return true;
}

bool WhisperModel::find_weight(const char* name, int32_t n_in, int32_t n_out, Weight& out, Error& error) const
{
    const float* data = nullptr;
    if (!find(name, n_in, n_out, 0, data, error))
    {
        return false;
    }
    out = Weight();
    out.type = GgmlType::F32;
    out.packed = false;
    out.data = reinterpret_cast<const uint8_t*>(data);
    out.n_in = static_cast<size_t>(n_in);
    out.n_out = static_cast<size_t>(n_out);
    return true;
}

bool WhisperModel::find_layer(const char* prefix, int layer, bool cross, WhisperLayer& out, Error& error) const
{
    const int32_t n_state = m_hparams.n_audio_state;
    char name[128];
    const auto vector = [&](const char* suffix, const float*& target, int32_t n) -> bool
    {
        std::snprintf(name, sizeof(name), "%s.blocks.%d.%s", prefix, layer, suffix);
        return find(name, n, 0, 0, target, error);
    };
    const auto matrix = [&](const char* suffix, Weight& target, int32_t n_in, int32_t n_out) -> bool
    {
        std::snprintf(name, sizeof(name), "%s.blocks.%d.%s", prefix, layer, suffix);
        return find_weight(name, n_in, n_out, target, error);
    };
    const auto attention = [&](const char* block, WhisperAttention& a) -> bool
    {
        char buffer[64];
        const auto with = [&](const char* tail)
        {
            std::snprintf(buffer, sizeof(buffer), "%s%s", block, tail);
            return buffer;
        };
        return vector(with("_ln.weight"), a.norm_w, n_state) && vector(with("_ln.bias"), a.norm_b, n_state) &&
               matrix(with(".query.weight"), a.q, n_state, n_state) && vector(with(".query.bias"), a.q_b, n_state) &&
               matrix(with(".key.weight"), a.k, n_state, n_state) &&
               matrix(with(".value.weight"), a.v, n_state, n_state) && vector(with(".value.bias"), a.v_b, n_state) &&
               matrix(with(".out.weight"), a.out, n_state, n_state) && vector(with(".out.bias"), a.out_b, n_state);
    };
    if (!attention("attn", out.attn))
    {
        return false;
    }
    if (cross && !attention("cross_attn", out.cross))
    {
        return false;
    }
    return vector("mlp_ln.weight", out.mlp_norm_w, n_state) && vector("mlp_ln.bias", out.mlp_norm_b, n_state) &&
           matrix("mlp.0.weight", out.mlp_0, n_state, 4 * n_state) && vector("mlp.0.bias", out.mlp_0_b, 4 * n_state) &&
           matrix("mlp.2.weight", out.mlp_1, 4 * n_state, n_state) && vector("mlp.2.bias", out.mlp_1_b, n_state);
}

}
