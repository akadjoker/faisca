// TAEHV video decoder pieces (FlashDreams flashdreams/recipes/taehv/impl.py Decoder): f32 NCHW
// frames, one frame after the other in memory.

namespace
{

constexpr unsigned int TILE_W = 16;
constexpr unsigned int TILE_H = 8;
constexpr unsigned int TILE_C = 64; // output channels per block
constexpr unsigned int STEP_C = 8;  // input channels per shared memory step

}

// out[f][co][y][x] = epilogue(bias[co] + sum over ci, ky, kx of w[ci][ky * K + kx][co] *
// in[f][ci][y + ky - K / 2][x + kx - K / 2]) with zero padding; the input channels are those of a
// (ca) followed by those of b (cb, b may be null): torch.cat([a, b], 1) without the copy.
// Weights packed [ci][K * K][cout_padded], cout_padded a multiple of 64. Epilogue: + residual
// (same shape as out) when given, then ReLU when relu, then clamp to [0, 1] when clamp01.
// With up, the inputs are height / 2 x width / 2 and read at (y / 2, x / 2): the nearest x 2
// upsample (nn.Upsample(scale_factor=2)) in front of the convolution, never stored.
// Block: 64 output channels by 16 x 8 pixels; thread: 4 channels by 8 pixels of one row.
template <int K>
__device__ __forceinline__ void conv(const float* a, unsigned int ca, const float* b, unsigned int cb, const float* w,
                                     unsigned int cout_padded, const float* bias, const float* residual, float* out,
                                     unsigned int cout, unsigned int height, unsigned int width, unsigned int relu,
                                     unsigned int clamp01, unsigned int up)
{
    constexpr int R = K / 2;
    constexpr int PW = TILE_W + 2 * R;
    constexpr int PH = TILE_H + 2 * R;
    __shared__ float patch[STEP_C][PH][PW + 1];
    __shared__ __align__(16) float wt[STEP_C][K * K][TILE_C];
    const unsigned int tiles_x = (width + TILE_W - 1) / TILE_W;
    const unsigned int x0 = (blockIdx.x % tiles_x) * TILE_W;
    const unsigned int y0 = (blockIdx.x / tiles_x) * TILE_H;
    const unsigned int c0 = blockIdx.y * TILE_C;
    const unsigned int frame = blockIdx.z;
    const unsigned int plane = height * width;
    const unsigned int shift = up != 0 ? 1 : 0;
    const unsigned int source_width = width >> shift;
    const unsigned int source_plane = (height >> shift) * source_width;
    const unsigned int cin = ca + cb;
    const float* fa = a + static_cast<unsigned long long>(frame) * ca * source_plane;
    const float* fb = b != nullptr ? b + static_cast<unsigned long long>(frame) * cb * source_plane : nullptr;
    const unsigned int tc = threadIdx.x % 16;
    const unsigned int tp = threadIdx.x / 16;
    const unsigned int row = tp % TILE_H;
    const unsigned int half = tp / TILE_H;
    float acc[4][8] = {};
    for (unsigned int c = 0; c < cin; c += STEP_C)
    {
        for (unsigned int i = threadIdx.x; i < STEP_C * PH * PW; i += 256)
        {
            const unsigned int ci = i / (PH * PW);
            const unsigned int rest = i % (PH * PW);
            const int py = static_cast<int>(rest / PW);
            const int px = static_cast<int>(rest % PW);
            const int y = static_cast<int>(y0) + py - R;
            const int x = static_cast<int>(x0) + px - R;
            const unsigned int channel = c + ci;
            float v = 0.0f;
            if (channel < cin && y >= 0 && x >= 0 && y < static_cast<int>(height) && x < static_cast<int>(width))
            {
                const unsigned int offset =
                    (static_cast<unsigned int>(y) >> shift) * source_width + (static_cast<unsigned int>(x) >> shift);
                v = channel < ca ? fa[static_cast<unsigned long long>(channel) * source_plane + offset]
                                 : fb[static_cast<unsigned long long>(channel - ca) * source_plane + offset];
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
            if (c + ci < cin)
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
                float in[8 + 2 * R];
#pragma unroll
                for (int i = 0; i < 8 + 2 * R; ++i)
                {
                    in[i] = patch[ci][row + ky][half * 8 + i];
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
                            acc[o][p] = fmaf(wc[o], in[p + kx], acc[o][p]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    const unsigned int y = y0 + row;
    if (y >= height)
    {
        return;
    }
    for (int o = 0; o < 4; ++o)
    {
        const unsigned int co = c0 + tc * 4 + o;
        if (co >= cout)
        {
            continue;
        }
        const unsigned long long base =
            (static_cast<unsigned long long>(frame) * cout + co) * plane + static_cast<unsigned long long>(y) * width;
        const float bv = bias != nullptr ? bias[co] : 0.0f;
        for (int p = 0; p < 8; ++p)
        {
            const unsigned int x = x0 + half * 8 + p;
            if (x >= width)
            {
                continue;
            }
            float v = acc[o][p] + bv;
            if (residual != nullptr)
            {
                v += residual[base + x];
            }
            if (relu != 0)
            {
                v = fmaxf(v, 0.0f);
            }
            if (clamp01 != 0)
            {
                v = fminf(fmaxf(v, 0.0f), 1.0f);
            }
            out[base + x] = v;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256)
    taehv_conv3(const float* a, unsigned int ca, const float* b, unsigned int cb, const float* w,
                unsigned int cout_padded, const float* bias, const float* residual, float* out, unsigned int cout,
                unsigned int height, unsigned int width, unsigned int relu, unsigned int clamp01, unsigned int up)
{
    conv<3>(a, ca, b, cb, w, cout_padded, bias, residual, out, cout, height, width, relu, clamp01, up);
}

extern "C" __global__ void __launch_bounds__(256)
    taehv_conv1(const float* a, unsigned int ca, const float* b, unsigned int cb, const float* w,
                unsigned int cout_padded, const float* bias, const float* residual, float* out, unsigned int cout,
                unsigned int height, unsigned int width, unsigned int relu, unsigned int clamp01, unsigned int up)
{
    conv<1>(a, ca, b, cb, w, cout_padded, bias, residual, out, cout, height, width, relu, clamp01, up);
}

// out = tanh(x / 3) * 3 (Clamp), or, with scale, x * std[c] + mean[c] first (the lighttae
// latent scaling of TaehvVAEDecoder.forward): x holds frames of channels planes of plane values
extern "C" __global__ void __launch_bounds__(256)
    taehv_input(const float* x, const float* mean, const float* std, float* out, unsigned int channels,
                unsigned int plane, unsigned long long count)
{
    for (unsigned long long i = blockIdx.x * 256ull + threadIdx.x; i < count; i += gridDim.x * 256ull)
    {
        const unsigned int c = static_cast<unsigned int>((i / plane) % channels);
        const float v = fmaf(x[i], std[c], mean[c]);
        out[i] = tanhf(v / 3.0f) * 3.0f;
    }
}
