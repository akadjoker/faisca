#include <cuda_fp16.h>

// Wan 2.1 VAE encoder pieces (FlashDreams flashdreams/recipes/wan/autoencoder/vae.py Encoder3d,
// the pruned "lightvae" OmniDreams uses): f32 NCHW frames, one frame at a time.

namespace
{

constexpr unsigned int TILE_W = 16;
constexpr unsigned int TILE_H = 8;
constexpr unsigned int TILE_C = 64; // output channels per block
constexpr unsigned int STEP_C = 8;  // input channels per shared memory step

}

// out[co][y][x] = bias[co] + sum over tap, ci, ky, kx of w[tap * cin + ci][ky * K + kx][co] *
// in_tap[ci][y * S + ky - pad][x * S + kx - pad] (+ residual), zeros outside. in0..in2 are the
// time taps of a causal (3, k, k) convolution (oldest first; null reads as a zero frame), or only
// in0 for a 2D one. pad = K / 2 for stride 1; 0 for stride 2, the Resample downsample's
// ZeroPad2d((0, 1, 0, 1)) then Conv2d(3, stride 2). Input height x width, output height / S x
// width / S. Weights packed [taps * cin][K * K][cout_padded]. Block: 64 output channels by 16 x 8
// output pixels; thread: 4 channels by 8 pixels of one row.
template <int K, int S>
__device__ __forceinline__ void conv(const float* in0, const float* in1, const float* in2, unsigned int taps,
                                     unsigned int cin, const float* w, unsigned int cout_padded, const float* bias,
                                     const float* residual, float* out, unsigned int cout, unsigned int height,
                                     unsigned int width)
{
    constexpr int PAD = S == 1 ? K / 2 : 0;
    constexpr int PW = (TILE_W - 1) * S + K;
    constexpr int PH = (TILE_H - 1) * S + K;
    constexpr int SPAN = 7 * S + K; // inputs one thread reads along a row
    __shared__ float patch[STEP_C][PH][PW + 1];
    __shared__ __align__(16) float wt[STEP_C][K * K][TILE_C];
    const unsigned int out_h = height / S;
    const unsigned int out_w = width / S;
    const unsigned int tiles_x = (out_w + TILE_W - 1) / TILE_W;
    const unsigned int x0 = (blockIdx.x % tiles_x) * TILE_W;
    const unsigned int y0 = (blockIdx.x / tiles_x) * TILE_H;
    const unsigned int c0 = blockIdx.y * TILE_C;
    const unsigned int plane = height * width;
    const unsigned int total = taps * cin;
    const float* inputs[3] = {in0, in1, in2};
    const unsigned int tc = threadIdx.x % 16;
    const unsigned int tp = threadIdx.x / 16;
    const unsigned int row = tp % TILE_H;
    const unsigned int half = tp / TILE_H;
    float acc[4][8] = {};
    for (unsigned int c = 0; c < total; c += STEP_C)
    {
        for (unsigned int i = threadIdx.x; i < STEP_C * PH * PW; i += 256)
        {
            const unsigned int ci = i / (PH * PW);
            const unsigned int rest = i % (PH * PW);
            const int py = static_cast<int>(rest / PW);
            const int px = static_cast<int>(rest % PW);
            const int y = static_cast<int>(y0 * S) + py - PAD;
            const int x = static_cast<int>(x0 * S) + px - PAD;
            const unsigned int channel = c + ci;
            float v = 0.0f;
            if (channel < total && y >= 0 && x >= 0 && y < static_cast<int>(height) && x < static_cast<int>(width))
            {
                const float* source = inputs[channel / cin];
                if (source != nullptr)
                {
                    v = source[static_cast<unsigned long long>(channel % cin) * plane +
                               static_cast<unsigned int>(y) * width + static_cast<unsigned int>(x)];
                }
            }
            patch[ci][py][px] = v;
        }
        for (unsigned int i = threadIdx.x; i < STEP_C * K * K * (TILE_C / 4); i += 256)
        {
            const unsigned int ci = i / (K * K * (TILE_C / 4));
            const unsigned int rest = i % (K * K * (TILE_C / 4));
            const unsigned int k = rest / (TILE_C / 4);
            const unsigned int q = rest % (TILE_C / 4);
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (c + ci < total)
            {
                v = *reinterpret_cast<const float4*>(
                    w + (static_cast<unsigned long long>(c + ci) * K * K + k) * cout_padded + c0 + q * 4);
            }
            *reinterpret_cast<float4*>(&wt[ci][k][q * 4]) = v;
        }
        __syncthreads();
#pragma unroll
        for (unsigned int ci = 0; ci < STEP_C; ++ci)
        {
#pragma unroll
            for (int ky = 0; ky < K; ++ky)
            {
                float in[SPAN];
#pragma unroll
                for (int i = 0; i < SPAN; ++i)
                {
                    in[i] = patch[ci][row * S + ky][half * 8 * S + i];
                }
#pragma unroll
                for (int kx = 0; kx < K; ++kx)
                {
                    const float4 wv = *reinterpret_cast<const float4*>(&wt[ci][ky * K + kx][tc * 4]);
                    const float wc[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                    for (int o = 0; o < 4; ++o)
                    {
#pragma unroll
                        for (int p = 0; p < 8; ++p)
                        {
                            acc[o][p] = fmaf(wc[o], in[p * S + kx], acc[o][p]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    const unsigned int y = y0 + row;
    if (y >= out_h)
    {
        return;
    }
    const unsigned int out_plane = out_h * out_w;
    for (int o = 0; o < 4; ++o)
    {
        const unsigned int co = c0 + tc * 4 + o;
        if (co >= cout)
        {
            continue;
        }
        const unsigned long long base = static_cast<unsigned long long>(co) * out_plane + y * out_w;
        for (int p = 0; p < 8; ++p)
        {
            const unsigned int x = x0 + half * 8 + p;
            if (x >= out_w)
            {
                continue;
            }
            float v = acc[o][p] + bias[co];
            if (residual != nullptr)
            {
                v += residual[base + x];
            }
            out[base + x] = v;
        }
    }
}

#define FAISCA_WAN_CONV(name, K, S)                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                            \
        name(const float* in0, const float* in1, const float* in2, unsigned int taps, unsigned int cin,          \
             const float* w, unsigned int cout_padded, const float* bias, const float* residual, float* out,     \
             unsigned int cout, unsigned int height, unsigned int width)                                         \
    {                                                                                                            \
        conv<K, S>(in0, in1, in2, taps, cin, w, cout_padded, bias, residual, out, cout, height, width);          \
    }

FAISCA_WAN_CONV(wan_conv3, 3, 1)
FAISCA_WAN_CONV(wan_conv3_down, 3, 2)
FAISCA_WAN_CONV(wan_conv1, 1, 1)

// RMS_norm over the channels (F.normalize(x, dim=1) * sqrt(C) * gamma: x / max(|x|, 1e-12)),
// then SiLU when silu; one thread per pixel of a channels x plane frame
extern "C" __global__ void __launch_bounds__(256)
    wan_norm(const float* x, const float* gamma, float* out, unsigned int channels, unsigned int plane,
             unsigned int silu)
{
    const unsigned int p = blockIdx.x * 256 + threadIdx.x;
    if (p >= plane)
    {
        return;
    }
    float squares = 0.0f;
    for (unsigned int c = 0; c < channels; ++c)
    {
        const float v = x[static_cast<unsigned long long>(c) * plane + p];
        squares = fmaf(v, v, squares);
    }
    const float scale = sqrtf(static_cast<float>(channels)) / fmaxf(sqrtf(squares), 1e-12f);
    for (unsigned int c = 0; c < channels; ++c)
    {
        const unsigned long long i = static_cast<unsigned long long>(c) * plane + p;
        float v = x[i] * scale * gamma[c];
        if (silu != 0)
        {
            v = v / (1.0f + expf(-v));
        }
        out[i] = v;
    }
}

// q, k, v planes (3 x channels x plane f32, to_qkv's output) to rows of f16 for attention_f16:
// row p = [q (128) | k (128) | v (128)], channels past `channels` zero
extern "C" __global__ void __launch_bounds__(256)
    wan_attention_pack(const float* qkv, unsigned short* rows, unsigned int channels, unsigned int plane)
{
    const unsigned int p = blockIdx.x;
    for (unsigned int i = threadIdx.x; i < 384; i += 256)
    {
        const unsigned int part = i / 128;
        const unsigned int c = i % 128;
        const float v = c < channels ? qkv[(static_cast<unsigned long long>(part) * channels + c) * plane + p] : 0.0f;
        const __half h = __float2half_rn(v);
        rows[static_cast<unsigned long long>(p) * 384 + i] = *reinterpret_cast<const unsigned short*>(&h);
    }
}

// attention rows (f16, 128 wide) back to channels x plane f32
extern "C" __global__ void __launch_bounds__(256)
    wan_attention_unpack(const unsigned short* rows, float* out, unsigned int channels, unsigned int plane)
{
    const unsigned long long count = static_cast<unsigned long long>(channels) * plane;
    for (unsigned long long i = blockIdx.x * 256ull + threadIdx.x; i < count; i += gridDim.x * 256ull)
    {
        const unsigned int c = static_cast<unsigned int>(i / plane);
        const unsigned int p = static_cast<unsigned int>(i % plane);
        out[i] = __half2float(*reinterpret_cast<const __half*>(&rows[static_cast<unsigned long long>(p) * 128 + c]));
    }
}

// WanVAE.encode: (mu - mean) * inv_std over 16 channels
extern "C" __global__ void __launch_bounds__(256)
    wan_latent(const float* mu, const float* mean, const float* inv_std, float* out, unsigned int plane)
{
    const unsigned long long count = 16ull * plane;
    for (unsigned long long i = blockIdx.x * 256ull + threadIdx.x; i < count; i += gridDim.x * 256ull)
    {
        const unsigned int c = static_cast<unsigned int>(i / plane);
        out[i] = (mu[i] - mean[c]) * inv_std[c];
    }
}
