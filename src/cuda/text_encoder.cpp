#include "faisca/text_encoder_cuda.h"

#include "kernels_ptx.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

constexpr uint32_t W = CosmosTextCudaEncoder::WIDTH;
constexpr uint32_t L = CosmosTextCudaEncoder::TOKENS;
constexpr uint32_t HEADS = 28;
constexpr uint32_t KV_HEADS = 4;
constexpr uint32_t KV = KV_HEADS * 128;
constexpr uint32_t QKV = W + 2 * KV; // 4608
constexpr uint32_t INNER = 18944;
constexpr float EPS = 1e-6f;
// one layer in a slot, bf16 rows: q | k | v, o, gate | up, down
constexpr uint64_t QKV_BYTES = uint64_t(QKV) * W * 2;
constexpr uint64_t O_BYTES = uint64_t(W) * W * 2;
constexpr uint64_t GU_BYTES = uint64_t(2 * INNER) * W * 2;
constexpr uint64_t DOWN_BYTES = uint64_t(W) * INNER * 2;
constexpr uint64_t O_AT = QKV_BYTES;
constexpr uint64_t GU_AT = O_AT + O_BYTES;
constexpr uint64_t DOWN_AT = GU_AT + GU_BYTES;
constexpr uint64_t SLOT_BYTES = DOWN_AT + DOWN_BYTES;
// CosmosReason1TextEncoder._encode's conversation through Qwen2.5-VL's chat template
constexpr const char* SYSTEM = "You are a helpful assistant who will provide prompts to an image generator.";

CudaBuffer range(const CudaBuffer& buffer, uint64_t offset, uint64_t bytes)
{
    CudaBuffer out = buffer;
    out.address += offset;
    out.bytes = bytes;
    return out;
}

float bf16_to_float(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

}

CosmosTextCudaEncoder::~CosmosTextCudaEncoder()
{
    release();
}

void CosmosTextCudaEncoder::release()
{
    if (m_device == nullptr)
    {
        return;
    }
    Error ignored;
    m_device->synchronize(ignored);
    const auto drop = [&](CudaBuffer& b) {
        if (b.bytes != 0)
        {
            m_device->destroy_buffer(b, ignored);
        }
        b = CudaBuffer();
    };
    for (Layer& layer : m_layers)
    {
        drop(layer.bias);
        drop(layer.norm1);
        drop(layer.norm2);
    }
    CudaBuffer* buffers[] = {&m_final_norm, &m_ones,    &m_slots[0], &m_slots[1], &m_x,      &m_h,
                             &m_qkv,        &m_attn,    &m_gate_up,  &m_mid,      &m_angles, &m_out};
    for (CudaBuffer* b : buffers)
    {
        drop(*b);
    }
    for (int i = 0; i < 2; ++i)
    {
        if (m_staging[i].bytes != 0)
        {
            m_device->destroy_host_buffer(m_staging[i], ignored);
        }
        m_staging[i] = CudaHostBuffer();
        m_device->destroy_event(m_uploaded[i], ignored);
        m_device->destroy_event(m_used[i], ignored);
    }
    m_device->destroy_module(m_module, ignored);
    m_device->destroy_stream(m_stream, ignored);
    m_device->destroy_stream(m_copy_stream, ignored);
    m_kernels.release();
    m_device = nullptr;
}

bool CosmosTextCudaEncoder::init(CudaDevice& device, const char* model_dir, Error& error)
{
    release();
    m_device = &device;
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
    if (!m_tokenizer.load_hf(path, error))
    {
        release();
        return false;
    }
    m_pad = m_tokenizer.find_token("<|endoftext|>");
    std::snprintf(path, sizeof(path), "%s/model.safetensors.index.json", model_dir);
    if (m_pad < 0 || !m_weights.open(path, error))
    {
        if (m_pad < 0)
        {
            error.set("tokenizer.json has no <|endoftext|> pad token");
        }
        release();
        return false;
    }
    const auto tensor = [&](const char* name, uint64_t rows, uint64_t columns) -> const uint8_t* {
        const SafetensorsTensor* t = m_weights.find(name);
        const bool shape = columns == 0 ? t != nullptr && t->n_dims == 1 && t->shape[0] == rows
                                        : t != nullptr && t->n_dims == 2 && t->shape[0] == rows && t->shape[1] == columns;
        if (!shape || t->dtype != SafetensorsDtype::BF16)
        {
            error.set("Cosmos-Reason1: %s missing or not bf16 of the expected shape", name);
            return nullptr;
        }
        return t->data;
    };
    m_embedding = reinterpret_cast<const uint16_t*>(tensor("model.embed_tokens.weight", 152064, W));
    const uint8_t* final_norm = tensor("model.norm.weight", W, 0);
    bool ok = m_embedding != nullptr && final_norm != nullptr && m_kernels.init(device, error) &&
              device.create_stream(m_stream, error) && device.create_stream(m_copy_stream, error) &&
              device.load_module(ptx_qwen, m_module, error) &&
              device.find_kernel(m_module, "qwen_rms_norm", m_rms_norm, error) &&
              device.find_kernel(m_module, "qwen_swiglu", m_swiglu, error) &&
              device.find_kernel(m_module, "qwen_hidden_out", m_hidden_out, error) &&
              device.create_buffer(W * 2, m_final_norm, error) && device.upload(m_final_norm, final_norm, W * 2, error);
    Array<float> values;
    ok = ok && values.resize_uninitialized(QKV > W ? QKV : W);
    for (uint32_t i = 0; ok && i < W; ++i)
    {
        values[i] = 1.0f;
    }
    ok = ok && device.create_buffer(W * 4, m_ones, error) && device.upload(m_ones, values.data(), W * 4, error);
    char name[128];
    for (uint32_t l = 0; ok && l < LAYERS; ++l)
    {
        Layer& layer = m_layers[l];
        const auto get = [&](const char* part, uint64_t rows, uint64_t columns) {
            std::snprintf(name, sizeof(name), "model.layers.%u.%s", l, part);
            return tensor(name, rows, columns);
        };
        layer.q = get("self_attn.q_proj.weight", W, W);
        layer.k = get("self_attn.k_proj.weight", KV, W);
        layer.v = get("self_attn.v_proj.weight", KV, W);
        layer.o = get("self_attn.o_proj.weight", W, W);
        layer.gate = get("mlp.gate_proj.weight", INNER, W);
        layer.up = get("mlp.up_proj.weight", INNER, W);
        layer.down = get("mlp.down_proj.weight", W, INNER);
        const uint8_t* q_bias = get("self_attn.q_proj.bias", W, 0);
        const uint8_t* k_bias = get("self_attn.k_proj.bias", KV, 0);
        const uint8_t* v_bias = get("self_attn.v_proj.bias", KV, 0);
        const uint8_t* norm1 = get("input_layernorm.weight", W, 0);
        const uint8_t* norm2 = get("post_attention_layernorm.weight", W, 0);
        ok = layer.q && layer.k && layer.v && layer.o && layer.gate && layer.up && layer.down && q_bias && k_bias &&
             v_bias && norm1 && norm2;
        if (!ok)
        {
            break;
        }
        const uint16_t* biases[3] = {reinterpret_cast<const uint16_t*>(q_bias),
                                     reinterpret_cast<const uint16_t*>(k_bias),
                                     reinterpret_cast<const uint16_t*>(v_bias)};
        const uint32_t sizes[3] = {W, KV, KV};
        uint32_t at = 0;
        for (int b = 0; b < 3; ++b)
        {
            for (uint32_t i = 0; i < sizes[b]; ++i)
            {
                values[at++] = bf16_to_float(biases[b][i]);
            }
        }
        ok = device.create_buffer(QKV * 4, layer.bias, error) && device.upload(layer.bias, values.data(), QKV * 4, error) &&
             device.create_buffer(W * 2, layer.norm1, error) && device.upload(layer.norm1, norm1, W * 2, error) &&
             device.create_buffer(W * 2, layer.norm2, error) && device.upload(layer.norm2, norm2, W * 2, error);
    }
    for (int i = 0; ok && i < 2; ++i)
    {
        ok = device.create_host_buffer(SLOT_BYTES, m_staging[i], error) &&
             device.create_buffer(SLOT_BYTES, m_slots[i], error) && device.create_event(m_uploaded[i], error) &&
             device.create_event(m_used[i], error);
    }
    ok = ok && device.create_buffer(size_t(L) * W * 4, m_x, error) && device.create_buffer(size_t(L) * W * 2, m_h, error) &&
         device.create_buffer(size_t(L) * QKV * 2, m_qkv, error) && device.create_buffer(size_t(L) * W * 2, m_attn, error) &&
         device.create_buffer(size_t(L) * 2 * INNER * 2, m_gate_up, error) &&
         device.create_buffer(size_t(L) * INNER * 2, m_mid, error) &&
         device.create_buffer(size_t(L) * 64 * 4, m_angles, error) &&
         device.create_buffer(size_t(L) * OUT_WIDTH * 2, m_out, error) && m_host_x.resize_uninitialized(size_t(L) * W);
    if (!ok)
    {
        release();
        return false;
    }
    return true;
}

// layer `index` into staging and device slot index % 2: the host copy waits until the slot's last
// upload is done, the upload until the compute that read the slot is done
bool CosmosTextCudaEncoder::stage(uint32_t index, Error& error)
{
    const uint32_t s = index % 2;
    const Layer& layer = m_layers[index];
    if (index >= 2 && !m_device->synchronize(m_uploaded[s], error))
    {
        return false;
    }
    uint8_t* host = static_cast<uint8_t*>(m_staging[s].data);
    std::memcpy(host, layer.q, uint64_t(W) * W * 2);
    std::memcpy(host + uint64_t(W) * W * 2, layer.k, uint64_t(KV) * W * 2);
    std::memcpy(host + uint64_t(W + KV) * W * 2, layer.v, uint64_t(KV) * W * 2);
    std::memcpy(host + O_AT, layer.o, O_BYTES);
    std::memcpy(host + GU_AT, layer.gate, GU_BYTES / 2);
    std::memcpy(host + GU_AT + GU_BYTES / 2, layer.up, GU_BYTES / 2);
    std::memcpy(host + DOWN_AT, layer.down, DOWN_BYTES);
    return (index < 2 || m_device->wait_event(m_copy_stream, m_used[s], error)) &&
           m_device->upload_async(m_slots[s], m_staging[s], SLOT_BYTES, m_copy_stream, error) &&
           m_device->record_event(m_uploaded[s], m_copy_stream, error);
}

bool CosmosTextCudaEncoder::encode(const char* prompt, Array<uint16_t>& out, Error& error)
{
    using Out = CudaKernels::GemmOutput;
    if (m_device == nullptr)
    {
        error.set("text encoder used before init");
        return false;
    }
    // <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n, then the pad token
    Array<char> text;
    const auto add = [&](const char* s) { return text.append(s, std::strlen(s)); };
    m_ids.clear();
    if (!add("<|im_start|>system\n") || !add(SYSTEM) || !add("<|im_end|>\n<|im_start|>user\n") || !add(prompt) ||
        !add("<|im_end|>\n") || !m_tokenizer.encode(std::string_view(text.data(), text.size()), false, true, m_ids))
    {
        error.set("out of memory while tokenizing");
        return false;
    }
    if (m_ids.size() > L)
    {
        m_ids.resize_uninitialized(L);
    }
    while (m_ids.size() < L)
    {
        if (!m_ids.push_back(m_pad))
        {
            error.set("out of memory");
            return false;
        }
    }
    // embed_tokens rows to f32 on the host
    for (uint32_t t = 0; t < L; ++t)
    {
        const uint16_t* row = m_embedding + uint64_t(static_cast<uint32_t>(m_ids[t])) * W;
        for (uint32_t c = 0; c < W; ++c)
        {
            m_host_x[size_t(t) * W + c] = bf16_to_float(row[c]);
        }
    }
    // Qwen2_5_VLRotaryEmbedding: inv_freq = 1 / 1e6 ** (arange(0, 128, 2) / 128) in float32, angle
    // position * inv_freq; for text the three M-RoPE sections share the position
    Array<float> angles;
    if (!angles.resize_uninitialized(size_t(L) * 64) || !out.resize_uninitialized(size_t(L) * OUT_WIDTH))
    {
        error.set("out of memory");
        return false;
    }
    for (uint32_t i = 0; i < 64; ++i)
    {
        const float exponent = static_cast<float>(2 * i) / 128.0f;
        const float inv_freq = 1.0f / std::pow(1000000.0f, exponent);
        for (uint32_t p = 0; p < L; ++p)
        {
            angles[size_t(p) * 64 + i] = static_cast<float>(p) * inv_freq;
        }
    }
    if (!m_device->upload(m_x, m_host_x.data(), size_t(L) * W * 4, error) ||
        !m_device->upload(m_angles, angles.data(), size_t(L) * 64 * 4, error) || !stage(0, error) ||
        !stage(1, error))
    {
        return false;
    }
    uint32_t width = W;
    float eps = EPS;
    const auto norm = [&](const CudaBuffer& gamma) {
        uint64_t a[] = {m_x.address, gamma.address, m_h.address};
        void* args[] = {&a[0], &a[1], &a[2], &width, &eps};
        CudaLaunch launch;
        launch.grid_x = L;
        launch.block_x = 256;
        return m_device->launch(m_rms_norm, m_stream, launch, args, error);
    };
    const float scale = 1.0f / std::sqrt(128.0f);
    for (uint32_t l = 0; l < LAYERS; ++l)
    {
        const uint32_t s = l % 2;
        const Layer& layer = m_layers[l];
        uint32_t inner = INNER;
        unsigned long long count = uint64_t(L) * INNER;
        uint64_t sw[] = {m_gate_up.address, m_mid.address};
        void* sw_args[] = {&sw[0], &sw[1], &inner, &count};
        CudaLaunch sw_launch;
        sw_launch.grid_x = static_cast<uint32_t>((count + 255) / 256);
        sw_launch.block_x = 256;
        unsigned long long out_stride = OUT_WIDTH;
        uint64_t ho[] = {m_x.address, l + 1 == LAYERS ? m_final_norm.address : 0, m_out.address + uint64_t(l) * W * 2};
        void* ho_args[] = {&ho[0], &ho[1], &ho[2], &width, &out_stride, &eps};
        CudaLaunch ho_launch;
        ho_launch.grid_x = L;
        ho_launch.block_x = 256;
        const bool ok =
            m_device->wait_event(m_stream, m_uploaded[s], error) && norm(layer.norm1) &&
            m_kernels.gemm_f16(m_h, range(m_slots[s], 0, QKV_BYTES), m_qkv, L, QKV, W, true, m_stream, error, Out::Half,
                               &layer.bias) &&
            m_kernels.dit_qk_norm_rope(m_qkv, QKV, 0, nullptr, &m_angles, L, HEADS, true, m_stream, error) &&
            m_kernels.dit_qk_norm_rope(m_qkv, QKV, W, nullptr, &m_angles, L, KV_HEADS, true, m_stream, error) &&
            m_kernels.attention_f16(m_qkv, QKV, range(m_qkv, W * 2, m_qkv.bytes - W * 2),
                                    range(m_qkv, (W + KV) * 2, m_qkv.bytes - (W + KV) * 2), QKV, m_attn, W, L, L,
                                    HEADS, scale, true, m_stream, error, KV_HEADS, true) &&
            m_kernels.gemm_f16(m_attn, range(m_slots[s], O_AT, O_BYTES), m_x, L, W, W, true, m_stream, error,
                               Out::Residual, &m_ones) &&
            norm(layer.norm2) &&
            m_kernels.gemm_f16(m_h, range(m_slots[s], GU_AT, GU_BYTES), m_gate_up, L, 2 * INNER, W, true, m_stream,
                               error, Out::Half) &&
            m_device->launch(m_swiglu, m_stream, sw_launch, sw_args, error) &&
            m_kernels.gemm_f16(m_mid, range(m_slots[s], DOWN_AT, DOWN_BYTES), m_x, L, W, INNER, true, m_stream, error,
                               Out::Residual, &m_ones) &&
            m_device->launch(m_hidden_out, m_stream, ho_launch, ho_args, error) &&
            m_device->record_event(m_used[s], m_stream, error) && (l + 2 >= LAYERS || stage(l + 2, error));
        if (!ok)
        {
            return false;
        }
    }
    return m_device->synchronize(m_stream, error) &&
           m_device->download(m_out, out.data(), size_t(L) * OUT_WIDTH * 2, error);
}

}
