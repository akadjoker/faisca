#include "faisca/wanvae_cuda.h"

#include "faisca/array.h"
#include "faisca/cuda_ops.h"
#include "faisca/torch_ckpt.h"
#include "kernels_ptx.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace faisca
{

namespace
{

// vae.py _WAN21_LATENT_MEAN / _WAN21_LATENT_STD
const float WAN21_MEAN[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                              0.4134f,  -0.0715f, 0.5517f,  -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
const float WAN21_STD[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                             3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
// lightvae: Encoder3d(dim=96, pruning_rate=0.75): dims 24, 24, 48, 96, 96 over levels 0..3
const uint32_t LEVEL_CHANNELS[4] = {24, 48, 96, 96};

CudaBuffer view(const CudaBuffer& buffer, uint64_t offset)
{
    CudaBuffer out = buffer;
    out.address += offset;
    out.bytes -= offset;
    return out;
}

}

WanVaeCudaEncoder::~WanVaeCudaEncoder()
{
    release();
}

void WanVaeCudaEncoder::release()
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
    const auto drop_conv = [&](Conv& c) {
        drop(c.weight);
        drop(c.bias);
    };
    const auto drop_ring = [&](Ring& r) {
        for (CudaBuffer& s : r.slot)
        {
            drop(s);
        }
    };
    drop_conv(m_conv_in);
    drop_conv(m_head);
    drop_conv(m_out);
    drop_conv(m_qkv);
    drop_conv(m_proj);
    drop_ring(m_ring_in);
    drop_ring(m_ring_head);
    for (Block& b : m_blocks)
    {
        drop(b.gamma0);
        drop(b.gamma3);
        drop_conv(b.a);
        drop_conv(b.b);
        drop_conv(b.shortcut);
        drop_ring(b.ring_a);
        drop_ring(b.ring_b);
    }
    for (Down& d : m_down)
    {
        drop_conv(d.spatial);
        drop_conv(d.time);
        drop_ring(d.ring);
    }
    for (int l = 0; l < 4; ++l)
    {
        drop(m_act[l][0]);
        drop(m_act[l][1]);
        drop(m_mid[l]);
        drop(m_short[l]);
    }
    CudaBuffer* buffers[] = {&m_head_gamma, &m_attn_gamma, &m_mean,       &m_inv_std,  &m_qkv_planes, &m_rows,
                             &m_attn_rows,  &m_attn_out,   &m_head_out,   &m_mu,       &m_latent_out};
    for (CudaBuffer* b : buffers)
    {
        drop(*b);
    }
    m_kernels.release();
    m_device->destroy_module(m_module, ignored);
    m_device->destroy_stream(m_stream, ignored);
    m_device = nullptr;
}

// weight [cout][cin][kt][k][k] (or [cout][cin][k][k]) -> [kt * cin][k * k][cout padded to 64],
// time tap t of input channel ci at row t * cin + ci; only the first `rows` outputs when not 0
bool WanVaeCudaEncoder::load_conv(const void* file_pointer, const char* name, uint32_t taps, uint32_t rows, Conv& c,
                                  Error& error)
{
    const CkptFile& file = *static_cast<const CkptFile*>(file_pointer);
    char key[96];
    std::snprintf(key, sizeof(key), "%s.weight", name);
    const CkptTensor* t = file.find(key);
    const bool five = t != nullptr && t->n_dims == 5;
    if (t == nullptr || t->dtype != CkptDtype::F32 || (t->n_dims != 4 && !five) ||
        (five ? t->shape[2] : 1) != taps || t->shape[t->n_dims - 1] != t->shape[t->n_dims - 2])
    {
        error.set("Wan VAE checkpoint: %s missing or not the expected f32 convolution", key);
        return false;
    }
    c.cout = rows != 0 ? rows : static_cast<uint32_t>(t->shape[0]);
    c.cin = static_cast<uint32_t>(t->shape[1]);
    c.taps = taps;
    c.k = static_cast<uint32_t>(t->shape[t->n_dims - 1]);
    c.cout_padded = (c.cout + 63) / 64 * 64;
    const size_t kk = size_t(c.k) * c.k;
    Array<float> packed;
    if (!packed.resize_uninitialized(size_t(taps) * c.cin * kk * c.cout_padded))
    {
        error.set("out of memory");
        return false;
    }
    std::memset(packed.data(), 0, packed.size() * 4);
    const float* w = reinterpret_cast<const float*>(t->data);
    for (uint32_t co = 0; co < c.cout; ++co)
    {
        for (uint32_t ci = 0; ci < c.cin; ++ci)
        {
            for (uint32_t tap = 0; tap < taps; ++tap)
            {
                for (size_t k = 0; k < kk; ++k)
                {
                    packed.data()[((size_t(tap) * c.cin + ci) * kk + k) * c.cout_padded + co] =
                        w[((size_t(co) * c.cin + ci) * taps + tap) * kk + k];
                }
            }
        }
    }
    std::snprintf(key, sizeof(key), "%s.bias", name);
    const CkptTensor* b = file.find(key);
    if (b == nullptr || b->dtype != CkptDtype::F32 || b->n_elements < c.cout)
    {
        error.set("Wan VAE checkpoint: %s missing", key);
        return false;
    }
    return m_device->create_buffer(packed.size() * 4, c.weight, error) &&
           m_device->upload(c.weight, packed.data(), packed.size() * 4, error) &&
           m_device->create_buffer(c.cout * 4, c.bias, error) && m_device->upload(c.bias, b->data, c.cout * 4, error);
}

bool WanVaeCudaEncoder::init(CudaDevice& device, const char* checkpoint, uint32_t height, uint32_t width,
                             Error& error)
{
    release();
    if (height % 16 != 0 || width % 16 != 0 || height == 0 || width == 0)
    {
        error.set("Wan VAE encoder: the frame size must be a multiple of 16");
        return false;
    }
    m_device = &device;
    m_height = height;
    m_width = width;
    CkptFile file;
    bool ok = file.open(checkpoint, error) && device.create_stream(m_stream, error) && m_kernels.init(device, error) &&
              device.load_module(ptx_wanvae, m_module, error) &&
              device.find_kernel(m_module, "wan_conv3", m_conv3, error) &&
              device.find_kernel(m_module, "wan_conv3_down", m_conv3_down, error) &&
              device.find_kernel(m_module, "wan_conv1", m_conv1, error) &&
              device.find_kernel(m_module, "wan_norm", m_norm, error) &&
              device.find_kernel(m_module, "wan_attention_pack", m_pack, error) &&
              device.find_kernel(m_module, "wan_attention_unpack", m_unpack, error) &&
              device.find_kernel(m_module, "wan_latent", m_latent, error);
    const auto gamma = [&](const char* name, uint32_t count, CudaBuffer& out) {
        const CkptTensor* t = file.find(name);
        if (t == nullptr || t->dtype != CkptDtype::F32 || t->n_elements != count)
        {
            error.set("Wan VAE checkpoint: %s missing", name);
            return false;
        }
        return device.create_buffer(count * 4, out, error) && device.upload(out, t->data, count * 4, error);
    };
    const auto plane = [&](uint32_t level) { return size_t(height >> level) * (width >> level); };
    const auto ring = [&](Ring& r, uint32_t channels, uint32_t level) {
        for (CudaBuffer& s : r.slot)
        {
            if (!device.create_buffer(size_t(channels) * plane(level) * 4, s, error))
            {
                return false;
            }
        }
        return true;
    };
    ok = ok && load_conv(&file, "encoder.conv1", 3, 0, m_conv_in, error) && ring(m_ring_in, 3, 0);
    // downsamples.0-10 and middle.0, middle.2: (checkpoint name, level)
    struct Layout
    {
        const char* name;
        uint32_t level;
    };
    const Layout blocks[BLOCKS] = {{"encoder.downsamples.0", 0}, {"encoder.downsamples.1", 0},
                                   {"encoder.downsamples.3", 1}, {"encoder.downsamples.4", 1},
                                   {"encoder.downsamples.6", 2}, {"encoder.downsamples.7", 2},
                                   {"encoder.downsamples.9", 3}, {"encoder.downsamples.10", 3},
                                   {"encoder.middle.0", 3},      {"encoder.middle.2", 3}};
    char name[96];
    for (int i = 0; ok && i < BLOCKS; ++i)
    {
        Block& b = m_blocks[i];
        b.level = blocks[i].level;
        std::snprintf(name, sizeof(name), "%s.residual.2", blocks[i].name);
        ok = load_conv(&file, name, 3, 0, b.a, error);
        std::snprintf(name, sizeof(name), "%s.residual.6", blocks[i].name);
        ok = ok && load_conv(&file, name, 3, 0, b.b, error);
        b.cin = b.a.cin;
        b.cout = b.a.cout;
        b.has_shortcut = b.cin != b.cout;
        if (ok && b.has_shortcut)
        {
            std::snprintf(name, sizeof(name), "%s.shortcut", blocks[i].name);
            ok = load_conv(&file, name, 1, 0, b.shortcut, error);
        }
        std::snprintf(name, sizeof(name), "%s.residual.0.gamma", blocks[i].name);
        ok = ok && gamma(name, b.cin, b.gamma0);
        std::snprintf(name, sizeof(name), "%s.residual.3.gamma", blocks[i].name);
        ok = ok && gamma(name, b.cout, b.gamma3) && ring(b.ring_a, b.cin, b.level) && ring(b.ring_b, b.cout, b.level);
    }
    const char* downs[3] = {"encoder.downsamples.2", "encoder.downsamples.5", "encoder.downsamples.8"};
    for (int i = 0; ok && i < 3; ++i)
    {
        Down& d = m_down[i];
        d.level = static_cast<uint32_t>(i);
        d.temporal = i > 0; // temperal_downsample (False, True, True)
        std::snprintf(name, sizeof(name), "%s.resample.1", downs[i]);
        ok = load_conv(&file, name, 1, 0, d.spatial, error);
        d.channels = d.spatial.cout;
        if (ok && d.temporal)
        {
            std::snprintf(name, sizeof(name), "%s.time_conv", downs[i]);
            ok = load_conv(&file, name, 3, 0, d.time, error) && ring(d.ring, d.channels, d.level + 1);
        }
    }
    const uint32_t p3 = static_cast<uint32_t>(plane(3));
    ok = ok && gamma("encoder.middle.1.norm.gamma", 96, m_attn_gamma) &&
         load_conv(&file, "encoder.middle.1.to_qkv", 1, 0, m_qkv, error) &&
         load_conv(&file, "encoder.middle.1.proj", 1, 0, m_proj, error) &&
         gamma("encoder.head.0.gamma", 96, m_head_gamma) && load_conv(&file, "encoder.head.2", 3, 0, m_head, error) &&
         ring(m_ring_head, 96, 3) && load_conv(&file, "conv1", 1, 16, m_out, error);
    for (uint32_t l = 0; ok && l < 4; ++l)
    {
        const size_t bytes = size_t(LEVEL_CHANNELS[l]) * plane(l) * 4;
        ok = device.create_buffer(bytes, m_act[l][0], error) && device.create_buffer(bytes, m_act[l][1], error) &&
             device.create_buffer(bytes, m_mid[l], error) && device.create_buffer(bytes, m_short[l], error);
    }
    float inv_std[16];
    for (int i = 0; i < 16; ++i)
    {
        inv_std[i] = 1.0f / WAN21_STD[i];
    }
    ok = ok && device.create_buffer(size_t(288) * p3 * 4, m_qkv_planes, error) &&
         device.create_buffer(size_t(384) * p3 * 2, m_rows, error) &&
         device.create_buffer(size_t(128) * p3 * 2, m_attn_rows, error) &&
         device.create_buffer(size_t(96) * p3 * 4, m_attn_out, error) &&
         device.create_buffer(size_t(32) * p3 * 4, m_head_out, error) &&
         device.create_buffer(size_t(16) * p3 * 4, m_mu, error) &&
         device.create_buffer(size_t(16) * p3 * 4, m_latent_out, error) && device.create_buffer(64, m_mean, error) &&
         device.create_buffer(64, m_inv_std, error) && device.upload(m_mean, WAN21_MEAN, 64, error) &&
         device.upload(m_inv_std, inv_std, 64, error);
    if (!ok)
    {
        release();
        return false;
    }
    reset();
    return true;
}

void WanVaeCudaEncoder::reset()
{
    Error ignored;
    Array<float> zeros;
    const auto clear = [&](Ring& r) {
        for (CudaBuffer& s : r.slot)
        {
            if (s.bytes == 0)
            {
                continue;
            }
            if (zeros.size() * 4 < s.bytes)
            {
                zeros.resize_uninitialized(s.bytes / 4);
                std::memset(zeros.data(), 0, s.bytes);
            }
            m_device->upload(s, zeros.data(), s.bytes, ignored);
        }
        r.next = 0;
    };
    clear(m_ring_in);
    clear(m_ring_head);
    for (Block& b : m_blocks)
    {
        clear(b.ring_a);
        clear(b.ring_b);
    }
    for (Down& d : m_down)
    {
        clear(d.ring);
        d.seen = 0;
    }
}

bool WanVaeCudaEncoder::run_conv(const Conv& c, const CudaBuffer* in0, const CudaBuffer* in1, const CudaBuffer* in2,
                                 const CudaBuffer* residual, const CudaBuffer& out, uint32_t height, uint32_t width,
                                 bool down, Error& error)
{
    uint64_t addresses[] = {in0 != nullptr ? in0->address : 0, in1 != nullptr ? in1->address : 0,
                            in2 != nullptr ? in2->address : 0, c.weight.address, c.bias.address,
                            residual != nullptr ? residual->address : 0, out.address};
    uint32_t taps = c.taps, cin = c.cin, cout_padded = c.cout_padded, cout = c.cout, h = height, w = width;
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &taps,         &cin, &addresses[3], &cout_padded,
                         &addresses[4], &addresses[5], &addresses[6], &cout,         &h,   &w};
    const uint32_t s = down ? 2 : 1;
    CudaLaunch launch;
    launch.grid_x = ((width / s + 15) / 16) * ((height / s + 7) / 8);
    launch.grid_y = c.cout_padded / 64;
    launch.block_x = 256;
    const CudaKernel& kernel = down ? m_conv3_down : c.k == 3 ? m_conv3 : m_conv1;
    return m_device->launch(kernel, m_stream, launch, arguments, error);
}

bool WanVaeCudaEncoder::norm(const CudaBuffer& x, const CudaBuffer& gamma, const CudaBuffer& out, uint32_t channels,
                             uint32_t level, bool silu, Error& error)
{
    uint64_t addresses[] = {x.address, gamma.address, out.address};
    uint32_t c = channels;
    uint32_t p = height(level) * width(level);
    uint32_t s = silu ? 1u : 0u;
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &c, &p, &s};
    CudaLaunch launch;
    launch.grid_x = (p + 255) / 256;
    launch.block_x = 256;
    return m_device->launch(m_norm, m_stream, launch, arguments, error);
}

// vae.py ResidualBlock: x + (or shortcut(x) +) conv_b(silu(norm_3(conv_a(silu(norm_0(x)))))), the
// causal convolutions reading their rings of the two inputs before and this one
bool WanVaeCudaEncoder::block(Block& b, CudaBuffer& x, Error& error)
{
    const uint32_t l = b.level;
    const uint32_t h = height(l), w = width(l);
    Ring& ra = b.ring_a;
    Ring& rb = b.ring_b;
    CudaBuffer& out = x.address == m_act[l][0].address ? m_act[l][1] : m_act[l][0];
    const CudaBuffer* residual = &x;
    bool ok = norm(x, b.gamma0, ra.slot[(ra.next + 2) % 3], b.cin, l, true, error) &&
              run_conv(b.a, &ra.slot[ra.next % 3], &ra.slot[(ra.next + 1) % 3], &ra.slot[(ra.next + 2) % 3], nullptr,
                       m_mid[l], h, w, false, error) &&
              norm(m_mid[l], b.gamma3, rb.slot[(rb.next + 2) % 3], b.cout, l, true, error);
    if (ok && b.has_shortcut)
    {
        ok = run_conv(b.shortcut, &x, nullptr, nullptr, nullptr, m_short[l], h, w, false, error);
        residual = &m_short[l];
    }
    ok = ok && run_conv(b.b, &rb.slot[rb.next % 3], &rb.slot[(rb.next + 1) % 3], &rb.slot[(rb.next + 2) % 3], residual,
                        out, h, w, false, error);
    ra.next++;
    rb.next++;
    x = out;
    return ok;
}

// vae.py AttentionBlock on one frame: x + proj(attention(to_qkv(norm(x)))), one head of 96 over
// the frame's pixels (FlashAttention-2 f16, 96 padded to 128 with zeros, scale 1 / sqrt(96))
bool WanVaeCudaEncoder::attention(CudaBuffer& x, Error& error)
{
    const uint32_t h = height(3), w = width(3);
    const uint32_t p = h * w;
    uint64_t pack_addresses[] = {m_qkv_planes.address, m_rows.address};
    uint32_t channels = 96;
    uint32_t plane = p;
    void* pack_arguments[] = {&pack_addresses[0], &pack_addresses[1], &channels, &plane};
    CudaLaunch pack;
    pack.grid_x = p;
    pack.block_x = 256;
    uint64_t unpack_addresses[] = {m_attn_rows.address, m_attn_out.address};
    void* unpack_arguments[] = {&unpack_addresses[0], &unpack_addresses[1], &channels, &plane};
    CudaLaunch unpack;
    unpack.grid_x = (96 * p + 255) / 256;
    unpack.block_x = 256;
    CudaBuffer& out = x.address == m_act[3][0].address ? m_act[3][1] : m_act[3][0];
    const bool ok =
        norm(x, m_attn_gamma, m_mid[3], 96, 3, false, error) &&
        run_conv(m_qkv, &m_mid[3], nullptr, nullptr, nullptr, m_qkv_planes, h, w, false, error) &&
        m_device->launch(m_pack, m_stream, pack, pack_arguments, error) &&
        m_kernels.attention_f16(m_rows, 384, view(m_rows, 256), view(m_rows, 512), 384, m_attn_rows, 128, p, p, 1,
                              1.0f / std::sqrt(96.0f), false, m_stream, error) &&
        m_device->launch(m_unpack, m_stream, unpack, unpack_arguments, error) &&
        run_conv(m_proj, &m_attn_out, nullptr, nullptr, &x, out, h, w, false, error);
    x = out;
    return ok;
}

// one frame through Encoder3d and WanVAE.conv1; produced when a latent frame came out
bool WanVaeCudaEncoder::frame(const CudaBuffer& input, float* latent, bool& produced, Error& error)
{
    (void)input;
    produced = false;
    Ring& ri = m_ring_in;
    CudaBuffer x = m_act[0][0];
    if (!run_conv(m_conv_in, &ri.slot[ri.next % 3], &ri.slot[(ri.next + 1) % 3], &ri.slot[(ri.next + 2) % 3],
                  nullptr, x, m_height, m_width, false, error))
    {
        return false;
    }
    ri.next++;
    int next_block = 0;
    for (uint32_t level = 0; level < 3; ++level)
    {
        for (int i = 0; i < 2; ++i)
        {
            if (!block(m_blocks[next_block++], x, error))
            {
                return false;
            }
        }
        Down& d = m_down[level];
        if (!d.temporal)
        {
            // Resample downsample2d into the next level's first buffer
            if (!run_conv(d.spatial, &x, nullptr, nullptr, nullptr, m_act[level + 1][0], height(level), width(level),
                          true, error))
            {
                return false;
            }
            x = m_act[level + 1][0];
            continue;
        }
        // Resample downsample3d (_downsample3d_step): the spatial convolution of every frame, the
        // first passed on as it is, then (frame i - 2, i - 1, i) through time_conv every second one
        const uint32_t i = d.seen++;
        CudaBuffer& slot = d.ring.slot[i % 3];
        if (!run_conv(d.spatial, &x, nullptr, nullptr, nullptr, slot, height(level), width(level), true, error))
        {
            return false;
        }
        if (i == 0)
        {
            x = slot;
            continue;
        }
        if (i % 2 == 1)
        {
            return true;
        }
        const CudaBuffer& out = m_act[level + 1][0];
        if (!run_conv(d.time, &d.ring.slot[(i - 2) % 3], &d.ring.slot[(i - 1) % 3], &slot, nullptr, out,
                      height(level + 1), width(level + 1), false, error))
        {
            return false;
        }
        x = out;
    }
    // level 3: downsamples.9, 10, then middle: block, attention, block
    if (!block(m_blocks[6], x, error) || !block(m_blocks[7], x, error) || !block(m_blocks[8], x, error) ||
        !attention(x, error) || !block(m_blocks[9], x, error))
    {
        return false;
    }
    Ring& rh = m_ring_head;
    const uint32_t h3 = height(3), w3 = width(3);
    uint64_t addresses[] = {m_mu.address, m_mean.address, m_inv_std.address, m_latent_out.address};
    uint32_t plane = h3 * w3;
    void* arguments[] = {&addresses[0], &addresses[1], &addresses[2], &addresses[3], &plane};
    CudaLaunch launch;
    launch.grid_x = (16 * plane + 255) / 256;
    launch.block_x = 256;
    const bool ok =
        norm(x, m_head_gamma, rh.slot[(rh.next + 2) % 3], 96, 3, true, error) &&
        run_conv(m_head, &rh.slot[rh.next % 3], &rh.slot[(rh.next + 1) % 3], &rh.slot[(rh.next + 2) % 3], nullptr,
                 m_head_out, h3, w3, false, error) &&
        run_conv(m_out, &m_head_out, nullptr, nullptr, nullptr, m_mu, h3, w3, false, error) &&
        m_device->launch(m_latent, m_stream, launch, arguments, error) && m_device->synchronize(m_stream, error) &&
        m_device->download(m_latent_out, latent, size_t(16) * plane * 4, error);
    rh.next++;
    produced = ok;
    return ok;
}

bool WanVaeCudaEncoder::encode(const float* frames, uint32_t count, float* latents, uint32_t& latent_frames,
                               Error& error)
{
    latent_frames = 0;
    if (m_device == nullptr)
    {
        error.set("Wan VAE encoder used before init");
        return false;
    }
    const size_t frame_floats = size_t(3) * m_height * m_width;
    const size_t latent_floats = size_t(16) * height(3) * width(3);
    for (uint32_t f = 0; f < count; ++f)
    {
        // the frame goes straight into conv1's ring, as its newest input
        Ring& ri = m_ring_in;
        bool produced = false;
        if (!m_device->synchronize(m_stream, error) ||
            !m_device->upload(ri.slot[(ri.next + 2) % 3], frames + f * frame_floats, frame_floats * 4, error) ||
            !frame(ri.slot[(ri.next + 2) % 3], latents + latent_frames * latent_floats, produced, error))
        {
            return false;
        }
        if (produced)
        {
            ++latent_frames;
        }
    }
    return m_device->synchronize(m_stream, error);
}

}
