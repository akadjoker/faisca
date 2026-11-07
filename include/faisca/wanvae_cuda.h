#pragma once

#include "faisca/cuda.h"
#include "faisca/cuda_ops.h"
#include "faisca/error.h"

namespace faisca
{

// The Wan 2.1 VAE encoder OmniDreams uses for the first frame and the HD map (lightvaew2_1.pth,
// huggingface.co/lightx2v/Autoencoders, the 75% pruned Wan VAE; FlashDreams
// flashdreams/recipes/wan/autoencoder/vae.py Encoder3d and WanVAE.encode) in f32 on the GPU,
// streaming: RGB frames in [-1, 1] in, 16-channel latents 8x smaller out, normalised with the
// Wan 2.1 latent mean and std. Frames go in one at a time: the first gives a latent frame, then
// every fourth (the causal convolutions keep their last two inputs, the temporal downsamples
// pair frames up), as the seed + 4-frame chunks of WanVAE.encode.
class WanVaeCudaEncoder
{
public:
    WanVaeCudaEncoder() = default;
    ~WanVaeCudaEncoder();
    WanVaeCudaEncoder(const WanVaeCudaEncoder&) = delete;
    WanVaeCudaEncoder& operator=(const WanVaeCudaEncoder&) = delete;

    // height x width: the frame size, multiples of 16, fixed for the encoder's life
    bool init(CudaDevice& device, const char* checkpoint, uint32_t height, uint32_t width, Error& error);
    void release();
    // a new rollout: the memories back to zero, the next frame is a first frame again
    void reset();

    // frames x 3 x height x width floats in [-1, 1]; latents gets latent_frames x 16 x
    // (height / 8) x (width / 8) floats
    bool encode(const float* frames, uint32_t count, float* latents, uint32_t& latent_frames, Error& error);

private:
    struct Conv
    {
        CudaBuffer weight, bias;
        uint32_t cin = 0, taps = 1, cout = 0, cout_padded = 0, k = 1;
    };
    struct Ring // the inputs of a causal convolution or a temporal downsample: three frames
    {
        CudaBuffer slot[3];
        uint32_t next = 0;
    };
    struct Block
    {
        CudaBuffer gamma0, gamma3;
        Conv a, b, shortcut;
        Ring ring_a, ring_b;
        uint32_t cin = 0, cout = 0, level = 0;
        bool has_shortcut = false;
    };
    struct Down
    {
        Conv spatial, time;
        Ring ring;
        uint32_t channels = 0, level = 0, seen = 0;
        bool temporal = false;
    };
    static constexpr int BLOCKS = 10;

    bool load_conv(const void* file, const char* name, uint32_t taps, uint32_t rows, Conv& c, Error& error);
    bool run_conv(const Conv& c, const CudaBuffer* in0, const CudaBuffer* in1, const CudaBuffer* in2,
                  const CudaBuffer* residual, const CudaBuffer& out, uint32_t height, uint32_t width, bool down,
                  Error& error);
    bool norm(const CudaBuffer& x, const CudaBuffer& gamma, const CudaBuffer& out, uint32_t channels,
              uint32_t level, bool silu, Error& error);
    bool block(Block& b, CudaBuffer& x, Error& error);
    bool attention(CudaBuffer& x, Error& error);
    bool frame(const CudaBuffer& input, float* latent, bool& produced, Error& error);
    uint32_t height(uint32_t level) const
    {
        return m_height >> level;
    }
    uint32_t width(uint32_t level) const
    {
        return m_width >> level;
    }

    CudaDevice* m_device = nullptr;
    CudaStream m_stream;
    CudaKernels m_kernels;
    CudaModule m_module;
    CudaKernel m_conv3, m_conv3_down, m_conv1, m_norm, m_pack, m_unpack, m_latent;
    uint32_t m_height = 0, m_width = 0;
    Conv m_conv_in, m_head, m_out, m_qkv, m_proj;
    Ring m_ring_in, m_ring_head;
    Block m_blocks[BLOCKS]; // 8 in the stages, the 2 of the middle
    Down m_down[3];
    CudaBuffer m_head_gamma, m_attn_gamma, m_mean, m_inv_std;
    // per level: two activation buffers, the first convolution's output, the shortcut's output
    CudaBuffer m_act[4][2], m_mid[4], m_short[4];
    CudaBuffer m_qkv_planes, m_rows, m_attn_rows, m_attn_out, m_head_out, m_mu, m_latent_out;
};

}
