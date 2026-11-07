#pragma once

#include "faisca/cuda.h"
#include "faisca/error.h"

namespace faisca
{

// The TAEHV video decoder OmniDreams uses (lighttaew2_1.pth, huggingface.co/lightx2v/Autoencoders;
// FlashDreams flashdreams/recipes/taehv: impl.py Decoder, __init__.py TaehvVAEDecoder) in f32 on
// the GPU, streaming: 16-channel latent frames in, RGB frames 8x larger in [0, 1] out, four per
// latent frame, the first three of a rollout dropped. Frames go through one at a time (each
// MemBlock only looks one frame back), so the buffers stay those of one frame.
class TaehvCudaDecoder
{
public:
    TaehvCudaDecoder() = default;
    ~TaehvCudaDecoder();
    TaehvCudaDecoder(const TaehvCudaDecoder&) = delete;
    TaehvCudaDecoder& operator=(const TaehvCudaDecoder&) = delete;

    // latent_height x latent_width: the latent size, fixed for the decoder's life
    bool init(CudaDevice& device, const char* checkpoint, uint32_t latent_height, uint32_t latent_width,
              Error& error);
    void release();
    // a new rollout: the MemBlock memories back to zero, the next frames trimmed again
    void reset();

    // latents: frames x 16 x latent_height x latent_width floats, as the DiT's sampler leaves them
    // (before the lighttae latent scaling); out gets out_frames x 3 x (8 h) x (8 w) floats
    bool decode(const float* latents, uint32_t frames, float* out, uint32_t& out_frames, Error& error);

    uint32_t max_out_frames(uint32_t frames) const
    {
        return frames * 4;
    }

private:
    struct Conv
    {
        CudaBuffer weight;
        CudaBuffer bias;
        uint32_t cin = 0;
        uint32_t cout = 0;
        uint32_t cout_padded = 0;
        uint32_t k = 0;
    };
    static constexpr int MEMBLOCKS = 9;
    static constexpr int MAX_CONVS = 1 + 3 * MEMBLOCKS + 1 + 1 + 2 + 1 + 2 + 1 + 1;

    bool conv(const Conv& c, const CudaBuffer& a, const CudaBuffer* b, const CudaBuffer* residual,
              const CudaBuffer& out, uint32_t height, uint32_t width, bool relu, bool clamp01, bool up, Error& error);
    bool memblock(int index, CudaBuffer* pool, uint32_t height, uint32_t width, Error& error);
    bool frame_stage2(const CudaBuffer& x, float*& out, uint32_t& count, Error& error);

    CudaDevice* m_device = nullptr;
    CudaStream m_stream;
    CudaModule m_module;
    CudaKernel m_conv3, m_conv1, m_input;
    Conv m_convs[MAX_CONVS];
    int m_conv_count = 0;
    int m_first = 0;                  // decoder.1
    int m_mem[MEMBLOCKS] = {};        // first of the three convs of each MemBlock
    int m_tgrow7 = 0, m_conv8 = 0, m_tgrow13[2] = {}, m_conv14 = 0, m_tgrow19[2] = {}, m_conv20 = 0, m_conv22 = 0;
    uint32_t m_h = 0, m_w = 0;
    // per stage: x, h1, h2 and the three MemBlock memories, all of one frame
    CudaBuffer m_pool[3][6];
    CudaBuffer m_latent, m_mean, m_std, m_grow, m_wide, m_rgb;
    uint32_t m_trim = 0;
};

}
