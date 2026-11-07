#include "faisca/taehv_cuda.h"

#include "faisca/array.h"
#include "faisca/torch_ckpt.h"
#include "kernels_ptx.h"

#include <cstdio>
#include <cstring>
#include <utility>

namespace faisca
{

namespace
{

// TaehvVAEDecoder._LIGHTTAE_MEAN / _LIGHTTAE_STD
const float LIGHTTAE_MEAN[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                 0.4134f,  -0.0715f, 0.5517f,  -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
const float LIGHTTAE_STD[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                                3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
constexpr uint32_t LATENT = 16;
constexpr uint32_t WIDTHS[3] = {256, 128, 64}; // channels of the three MemBlock stages

}

TaehvCudaDecoder::~TaehvCudaDecoder()
{
    release();
}

void TaehvCudaDecoder::release()
{
    if (m_device == nullptr)
    {
        return;
    }
    Error ignored;
    const auto drop = [&](CudaBuffer& buffer) {
        if (buffer.bytes != 0)
        {
            m_device->destroy_buffer(buffer, ignored);
        }
        buffer = CudaBuffer();
    };
    for (int i = 0; i < m_conv_count; ++i)
    {
        drop(m_convs[i].weight);
        drop(m_convs[i].bias);
    }
    for (auto& stage : m_pool)
    {
        for (CudaBuffer& buffer : stage)
        {
            drop(buffer);
        }
    }
    drop(m_latent);
    drop(m_mean);
    drop(m_std);
    drop(m_grow);
    drop(m_wide);
    drop(m_rgb);
    m_device->destroy_module(m_module, ignored);
    m_device->destroy_stream(m_stream, ignored);
    m_conv_count = 0;
    m_device = nullptr;
}

bool TaehvCudaDecoder::init(CudaDevice& device, const char* checkpoint, uint32_t latent_height, uint32_t latent_width,
                            Error& error)
{
    release();
    m_device = &device;
    m_h = latent_height;
    m_w = latent_width;
    CkptFile file;
    if (!file.open(checkpoint, error) || !device.create_stream(m_stream, error) ||
        !device.load_module(ptx_taehv, m_module, error) || !device.find_kernel(m_module, "taehv_conv3", m_conv3, error) ||
        !device.find_kernel(m_module, "taehv_conv1", m_conv1, error) ||
        !device.find_kernel(m_module, "taehv_input", m_input, error))
    {
        release();
        return false;
    }
    // weights [cout][cin][k][k] -> [cin][k * k][cout padded to 64]; rows [first, first + cout) of
    // the tensor (the halves of a stride 2 TGrow)
    Array<float> packed;
    const auto add = [&](const char* name, uint32_t cout, uint32_t first_row, bool bias, int& index) {
        char key[64];
        std::snprintf(key, sizeof(key), "decoder.%s.weight", name);
        const CkptTensor* t = file.find(key);
        if (t == nullptr || t->dtype != CkptDtype::F32 || t->n_dims != 4 || t->shape[0] < first_row + cout ||
            t->shape[2] != t->shape[3])
        {
            error.set("TAEHV checkpoint: %s missing or not f32 [cout][cin][k][k]", key);
            return false;
        }
        Conv& c = m_convs[m_conv_count];
        c.cin = static_cast<uint32_t>(t->shape[1]);
        c.k = static_cast<uint32_t>(t->shape[2]);
        c.cout = cout;
        c.cout_padded = (cout + 63) / 64 * 64;
        const size_t kk = size_t(c.k) * c.k;
        if (!packed.resize_uninitialized(size_t(c.cin) * kk * c.cout_padded))
        {
            error.set("out of memory");
            return false;
        }
        std::memset(packed.data(), 0, packed.size() * 4);
        const float* w = reinterpret_cast<const float*>(t->data);
        for (uint32_t co = 0; co < cout; ++co)
        {
            for (uint32_t ci = 0; ci < c.cin; ++ci)
            {
                for (size_t k = 0; k < kk; ++k)
                {
                    packed.data()[(ci * kk + k) * c.cout_padded + co] = w[((first_row + co) * size_t(c.cin) + ci) * kk + k];
                }
            }
        }
        if (!device.create_buffer(packed.size() * 4, c.weight, error) ||
            !device.upload(c.weight, packed.data(), packed.size() * 4, error))
        {
            return false;
        }
        if (bias)
        {
            std::snprintf(key, sizeof(key), "decoder.%s.bias", name);
            const CkptTensor* b = file.find(key);
            if (b == nullptr || b->dtype != CkptDtype::F32 || b->n_elements != cout)
            {
                error.set("TAEHV checkpoint: %s missing", key);
                return false;
            }
            if (!device.create_buffer(cout * 4, c.bias, error) || !device.upload(c.bias, b->data, cout * 4, error))
            {
                return false;
            }
        }
        index = m_conv_count++;
        return true;
    };
    // impl.py Decoder.blocks: 0 Clamp, 1 conv, 2 ReLU, 3-5 MemBlock, 6 Upsample, 7 TGrow(1), 8 conv,
    // 9-11 MemBlock, 12 Upsample, 13 TGrow(2), 14 conv, 15-17 MemBlock, 18 Upsample, 19 TGrow(2),
    // 20 conv, 21 ReLU, 22 conv
    const int mem_index[MEMBLOCKS] = {3, 4, 5, 9, 10, 11, 15, 16, 17};
    bool ok = add("1", 256, 0, true, m_first);
    for (int m = 0; ok && m < MEMBLOCKS; ++m)
    {
        const uint32_t c = WIDTHS[m / 3];
        char name[16];
        int ignored_index = 0;
        std::snprintf(name, sizeof(name), "%d.conv.0", mem_index[m]);
        ok = add(name, c, 0, true, m_mem[m]);
        std::snprintf(name, sizeof(name), "%d.conv.2", mem_index[m]);
        ok = ok && add(name, c, 0, true, ignored_index);
        std::snprintf(name, sizeof(name), "%d.conv.4", mem_index[m]);
        ok = ok && add(name, c, 0, true, ignored_index);
    }
    ok = ok && add("7.conv", 256, 0, false, m_tgrow7) && add("8", 128, 0, false, m_conv8) &&
         add("13.conv", 128, 0, false, m_tgrow13[0]) && add("13.conv", 128, 128, false, m_tgrow13[1]) &&
         add("14", 64, 0, false, m_conv14) && add("19.conv", 64, 0, false, m_tgrow19[0]) &&
         add("19.conv", 64, 64, false, m_tgrow19[1]) && add("20", 64, 0, false, m_conv20) &&
         add("22", 3, 0, true, m_conv22);
    const size_t hw = size_t(m_h) * m_w;
    for (int s = 0; ok && s < 3; ++s)
    {
        const size_t bytes = size_t(WIDTHS[s]) * hw * (size_t(1) << (2 * s)) * 4;
        for (int i = 0; ok && i < 6; ++i)
        {
            ok = device.create_buffer(bytes, m_pool[s][i], error);
        }
    }
    ok = ok && device.create_buffer(1024 * hw * 4, m_grow, error) &&
         device.create_buffer(64 * hw * 64 * 4, m_wide, error) && device.create_buffer(3 * hw * 64 * 4, m_rgb, error) &&
         device.create_buffer(4 * LATENT * hw * 4, m_latent, error) && device.create_buffer(64, m_mean, error) &&
         device.create_buffer(64, m_std, error) && device.upload(m_mean, LIGHTTAE_MEAN, 64, error) &&
         device.upload(m_std, LIGHTTAE_STD, 64, error);
    if (!ok)
    {
        release();
        return false;
    }
    reset();
    return true;
}

void TaehvCudaDecoder::reset()
{
    Error ignored;
    Array<float> zeros;
    for (int s = 0; s < 3; ++s)
    {
        for (int i = 3; i < 6; ++i)
        {
            const size_t count = m_pool[s][i].bytes / 4;
            if (zeros.size() < count)
            {
                zeros.resize_uninitialized(count);
                std::memset(zeros.data(), 0, count * 4);
            }
            m_device->upload(m_pool[s][i], zeros.data(), count * 4, ignored);
        }
    }
    // impl.py TAEHV.frames_to_trim: 2 ** sum(decoder_time_upscale) - 1
    m_trim = 3;
}

bool TaehvCudaDecoder::conv(const Conv& c, const CudaBuffer& a, const CudaBuffer* b, const CudaBuffer* residual,
                            const CudaBuffer& out, uint32_t height, uint32_t width, bool relu, bool clamp01, bool up,
                            Error& error)
{
    uint64_t addresses[] = {a.address, b != nullptr ? b->address : 0, c.weight.address,
                            c.bias.bytes != 0 ? c.bias.address : 0, residual != nullptr ? residual->address : 0,
                            out.address};
    uint32_t ca = b != nullptr ? c.cin / 2 : c.cin;
    uint32_t cb = b != nullptr ? c.cin / 2 : 0;
    uint32_t cout_padded = c.cout_padded;
    uint32_t cout = c.cout;
    uint32_t h = height;
    uint32_t w = width;
    uint32_t r = relu ? 1u : 0u;
    uint32_t cl = clamp01 ? 1u : 0u;
    uint32_t u = up ? 1u : 0u;
    void* arguments[] = {&addresses[0], &ca, &addresses[1], &cb, &addresses[2], &cout_padded, &addresses[3],
                         &addresses[4], &addresses[5], &cout, &h, &w, &r, &cl, &u};
    CudaLaunch launch;
    launch.grid_x = ((width + 15) / 16) * ((height + 7) / 8);
    launch.grid_y = c.cout_padded / 64;
    launch.block_x = 256;
    return m_device->launch(c.k == 3 ? m_conv3 : m_conv1, m_stream, launch, arguments, error);
}

// impl.py MemBlock: relu(conv(cat([x, past])) + x), conv = conv3 relu conv3 relu conv3; then the
// input becomes the memory (cache_step) and the result the next x
bool TaehvCudaDecoder::memblock(int index, CudaBuffer* pool, uint32_t height, uint32_t width, Error& error)
{
    const int c = m_mem[index];
    CudaBuffer& memory = pool[3 + index % 3];
    if (!conv(m_convs[c], pool[0], &memory, nullptr, pool[1], height, width, true, false, false, error) ||
        !conv(m_convs[c + 1], pool[1], nullptr, nullptr, pool[2], height, width, true, false, false, error) ||
        !conv(m_convs[c + 2], pool[2], nullptr, &pool[0], pool[1], height, width, true, false, false, error))
    {
        return false;
    }
    std::swap(memory, pool[0]);
    std::swap(pool[0], pool[1]);
    return true;
}

// the last MemBlock stage's frame x: the two TGrow halves, each through the 8x convolutions
bool TaehvCudaDecoder::frame_stage2(const CudaBuffer& x, float*& out, uint32_t& count, Error& error)
{
    const uint32_t h4 = m_h * 4, w4 = m_w * 4;
    const size_t frame = size_t(3) * m_h * 8 * m_w * 8;
    for (int s = 0; s < 2; ++s)
    {
        if (!conv(m_convs[m_tgrow19[s]], x, nullptr, nullptr, m_grow, h4, w4, false, false, false, error) ||
            !conv(m_convs[m_conv20], m_grow, nullptr, nullptr, m_wide, h4 * 2, w4 * 2, true, false, true, error) ||
            !conv(m_convs[m_conv22], m_wide, nullptr, nullptr, m_rgb, h4 * 2, w4 * 2, false, true, false, error))
        {
            return false;
        }
        if (m_trim > 0)
        {
            --m_trim;
            continue;
        }
        if (!m_device->synchronize(m_stream, error) || !m_device->download(m_rgb, out, frame * 4, error))
        {
            return false;
        }
        ++count;
        out += frame;
    }
    return true;
}

bool TaehvCudaDecoder::decode(const float* latents, uint32_t frames, float* out, uint32_t& out_frames, Error& error)
{
    out_frames = 0;
    if (m_device == nullptr || frames == 0 || frames > 4)
    {
        error.set("TAEHV decode: 1 to 4 latent frames at a time");
        return false;
    }
    const size_t hw = size_t(m_h) * m_w;
    if (!m_device->upload(m_latent, latents, frames * LATENT * hw * 4, error))
    {
        return false;
    }
    float* o = out;
    for (uint32_t f = 0; f < frames; ++f)
    {
        // TaehvVAEDecoder.forward z * std + mean, then Decoder.blocks[0] Clamp
        uint64_t addresses[] = {m_latent.address + f * LATENT * hw * 4, m_mean.address, m_std.address,
                                m_pool[0][1].address};
        uint32_t channels = LATENT;
        uint32_t plane = static_cast<uint32_t>(hw);
        unsigned long long count = LATENT * hw;
        void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &channels, &plane, &count};
        CudaLaunch launch;
        launch.grid_x = static_cast<uint32_t>((count + 255) / 256);
        launch.block_x = 256;
        if (!m_device->launch(m_input, m_stream, launch, arguments, error) ||
            !conv(m_convs[m_first], m_pool[0][1], nullptr, nullptr, m_pool[0][0], m_h, m_w, true, false, false,
                  error))
        {
            return false;
        }
        for (int m = 0; m < 3; ++m)
        {
            if (!memblock(m, m_pool[0], m_h, m_w, error))
            {
                return false;
            }
        }
        // TGrow(1) before the upsample: a 1x1 convolution commutes with nearest neighbour copies
        if (!conv(m_convs[m_tgrow7], m_pool[0][0], nullptr, nullptr, m_pool[0][1], m_h, m_w, false, false, false,
                  error) ||
            !conv(m_convs[m_conv8], m_pool[0][1], nullptr, nullptr, m_pool[1][0], m_h * 2, m_w * 2, false, false,
                  true, error))
        {
            return false;
        }
        for (int m = 3; m < 6; ++m)
        {
            if (!memblock(m, m_pool[1], m_h * 2, m_w * 2, error))
            {
                return false;
            }
        }
        for (int s = 0; s < 2; ++s)
        {
            if (!conv(m_convs[m_tgrow13[s]], m_pool[1][0], nullptr, nullptr, m_grow, m_h * 2, m_w * 2, false, false,
                      false, error) ||
                !conv(m_convs[m_conv14], m_grow, nullptr, nullptr, m_pool[2][0], m_h * 4, m_w * 4, false, false, true,
                      error))
            {
                return false;
            }
            for (int m = 6; m < 9; ++m)
            {
                if (!memblock(m, m_pool[2], m_h * 4, m_w * 4, error))
                {
                    return false;
                }
            }
            if (!frame_stage2(m_pool[2][0], o, out_frames, error))
            {
                return false;
            }
        }
    }
    return m_device->synchronize(m_stream, error);
}

}
