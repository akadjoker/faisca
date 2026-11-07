// TripoSR on the GPU in float32. Built with --fmad=false: every fused multiply-add is an
// explicit fmaf, so the products of tripo_gemm follow the CPU gemm_f32_packed order.

#define PANEL 16

// out[t][o] for t < n_tokens, o < n_out: acc = fmaf chain over k from 0 in k order (as
// gemm_f32_packed), then acc + bias[o] if bias, then out[t][o] + that if accumulate.
// panels: rows packed as pack_f32_panels does ([panel][k][16]). 256 threads, a tile of
// 64 tokens x 64 outputs, 4 x 4 outputs per thread.
extern "C" __global__ void tripo_gemm(const float* x, unsigned int x_stride, const float* panels, unsigned int n_in,
                                      unsigned int n_out, unsigned int n_tokens, const float* bias, float* out,
                                      unsigned int out_stride, unsigned int accumulate)
{
    __shared__ float as[16][64];
    __shared__ float bs[16][64];
    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid % 16;
    const unsigned int ty = tid / 16;
    const unsigned int t0 = blockIdx.y * 64;
    const unsigned int panel0 = blockIdx.x * 4;
    const unsigned int n_panels = n_out / PANEL;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            acc[i][j] = 0.0f;
        }
    }
    for (unsigned int k0 = 0; k0 < n_in; k0 += 16)
    {
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            const unsigned int e = tid + 256 * r;
            const unsigned int tl = e / 16;
            const unsigned int kl = e % 16;
            const unsigned int t = t0 + tl;
            const unsigned int k = k0 + kl;
            as[kl][tl] = t < n_tokens && k < n_in ? x[(size_t)t * x_stride + k] : 0.0f;
            const unsigned int p = panel0 + r;
            const unsigned int kw = k0 + tid / 16;
            bs[tid / 16][r * 16 + tid % 16] =
                p < n_panels && kw < n_in ? panels[((size_t)p * n_in + kw) * PANEL + tid % 16] : 0.0f;
        }
        __syncthreads();
        const unsigned int steps = n_in - k0 < 16 ? n_in - k0 : 16;
        for (unsigned int kk = 0; kk < steps; ++kk)
        {
            float a[4];
            float b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                a[i] = as[kk][ty + 16 * i];
                b[i] = bs[kk][tx + 16 * i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const unsigned int t = t0 + ty + 16 * i;
        if (t >= n_tokens)
        {
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const unsigned int o = panel0 * PANEL + tx + 16 * j;
            if (o >= n_out)
            {
                continue;
            }
            float v = acc[i][j];
            if (bias != 0)
            {
                v = v + bias[o];
            }
            float* dst = out + (size_t)t * out_stride + o;
            *dst = accumulate ? *dst + v : v;
        }
    }
}

// layer_norm (src/ops_whisper.cpp) of one row per block of 256 threads: double sums of the
// values and of the squared centred values, reduced by a fixed tree
extern "C" __global__ void tripo_layer_norm(const float* x, const float* weight, const float* bias, float* out,
                                            unsigned int width, float eps)
{
    __shared__ double partial[256];
    const float* row = x + (size_t)blockIdx.x * width;
    float* dst = out + (size_t)blockIdx.x * width;
    double sum = 0.0;
    for (unsigned int i = threadIdx.x; i < width; i += 256)
    {
        sum += row[i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
        {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float mean = (float)(partial[0] / (double)width);
    __syncthreads();
    double squares = 0.0;
    for (unsigned int i = threadIdx.x; i < width; i += 256)
    {
        const float centered = row[i] - mean;
        squares += (double)centered * (double)centered;
    }
    partial[threadIdx.x] = squares;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
        {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float variance = (float)(partial[0] / (double)width);
    const float scale = 1.0f / sqrtf(variance + eps);
    for (unsigned int i = threadIdx.x; i < width; i += 256)
    {
        const float centered = row[i] - mean;
        dst[i] = centered * scale * weight[i] + bias[i];
    }
}

__device__ float gelu_erf(float x)
{
    return x * 0.5f * (1.0f + erff(x * 0.70710678118654752440f));
}

extern "C" __global__ void tripo_gelu(float* x, unsigned int count)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        x[i] = gelu_erf(x[i]);
    }
}

// rows of 2 ff: the first half times the GELU of the second, in place in the first half
extern "C" __global__ void tripo_geglu(float* x, unsigned int ff, unsigned int count)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        float* row = x + (size_t)(i / ff) * 2 * ff;
        const unsigned int c = i % ff;
        row[c] = row[c] * gelu_erf(row[ff + c]);
    }
}

extern "C" __global__ void tripo_add(float* x, const float* y, unsigned int count)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        x[i] = x[i] + y[i];
    }
}

// softmax(q k^T * scale) v for head_dim 64: one query per thread, 128 queries of head
// blockIdx.y per block, keys and values in tiles of 64 through shared memory, online softmax
extern "C" __global__ void tripo_attention(const float* q, const float* k, const float* v, float* out,
                                           unsigned int n_q, unsigned int n_kv, unsigned int stride, float scale)
{
    __shared__ float ks[64][64];
    __shared__ float vs[64][64];
    const unsigned int head = blockIdx.y;
    const unsigned int query = blockIdx.x * 128 + threadIdx.x;
    const bool active = query < n_q;
    float qr[64];
    float o[64];
#pragma unroll
    for (int d = 0; d < 64; ++d)
    {
        qr[d] = active ? q[(size_t)query * stride + head * 64 + d] : 0.0f;
        o[d] = 0.0f;
    }
    float m = -3.402823466e38f;
    float l = 0.0f;
    for (unsigned int k0 = 0; k0 < n_kv; k0 += 64)
    {
        for (unsigned int e = threadIdx.x; e < 64 * 64; e += 128)
        {
            const unsigned int j = e / 64;
            const unsigned int d = e % 64;
            const bool in = k0 + j < n_kv;
            ks[j][d] = in ? k[(size_t)(k0 + j) * stride + head * 64 + d] : 0.0f;
            vs[j][d] = in ? v[(size_t)(k0 + j) * stride + head * 64 + d] : 0.0f;
        }
        __syncthreads();
        const unsigned int count = n_kv - k0 < 64 ? n_kv - k0 : 64;
        for (unsigned int j = 0; j < count; ++j)
        {
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < 64; ++d)
            {
                s = fmaf(qr[d], ks[j][d], s);
            }
            s = s * scale;
            if (s > m)
            {
                const float alpha = expf(m - s);
                l = l * alpha;
#pragma unroll
                for (int d = 0; d < 64; ++d)
                {
                    o[d] = o[d] * alpha;
                }
                m = s;
            }
            const float p = expf(s - m);
            l = l + p;
#pragma unroll
            for (int d = 0; d < 64; ++d)
            {
                o[d] = fmaf(p, vs[j][d], o[d]);
            }
        }
        __syncthreads();
    }
    if (active)
    {
        const float inverse = 1.0f / l;
#pragma unroll
        for (int d = 0; d < 64; ++d)
        {
            out[(size_t)query * stride + head * 64 + d] = o[d] * inverse;
        }
    }
}

// the transposed convolution's rows (t, (co * 2 + ky) * 2 + kx) plus bias[co] into the
// scene code [3][40][64][64] and the channel last planes [3][64][64][40]
extern "C" __global__ void tripo_upsample(const float* rows, const float* bias, float* code, float* planes,
                                          unsigned int count)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    const unsigned int t = i / 160;
    const unsigned int r = i % 160;
    const unsigned int co = r / 4;
    const unsigned int ky = (r / 2) % 2;
    const unsigned int kx = r % 2;
    const unsigned int np = t / 1024;
    const unsigned int y = 2 * ((t % 1024) / 32) + ky;
    const unsigned int x = 2 * (t % 32) + kx;
    const float value = rows[i] + bias[co];
    code[((np * 40 + co) * 64 + y) * 64 + x] = value;
    planes[((np * 64 + y) * 64 + x) * 40 + co] = value;
}

// F.grid_sample bilinear, zeros, align_corners False (GridSamplerKernel.cpp ApplyGridSample)
// of channel c at a point whose corners and weights are prepared
struct Corners
{
    const float* nw;
    const float* ne;
    const float* sw;
    const float* se;
    float w_nw, w_ne, w_sw, w_se;
    bool m_nw, m_ne, m_sw, m_se;
};

__device__ Corners corners(const float* plane, float gx, float gy)
{
    const float x = (gx + 1.0f) * 32.0f - 0.5f;
    const float y = (gy + 1.0f) * 32.0f - 0.5f;
    const float x_w = floorf(x);
    const float y_n = floorf(y);
    const float w = x - x_w;
    const float e = 1.0f - w;
    const float n = y - y_n;
    const float s = 1.0f - n;
    Corners c;
    c.w_nw = s * e;
    c.w_ne = s * w;
    c.w_sw = n * e;
    c.w_se = n * w;
    const int ix = (int)x_w;
    const int iy = (int)y_n;
    const bool west = ix > -1 && ix < 64;
    const bool east = ix + 1 > -1 && ix + 1 < 64;
    const bool north = iy > -1 && iy < 64;
    const bool south = iy + 1 > -1 && iy + 1 < 64;
    c.m_nw = west && north;
    c.m_ne = east && north;
    c.m_sw = west && south;
    c.m_se = east && south;
    const int cx = ix < 0 ? 0 : (ix > 62 ? 62 : ix);
    const int cy = iy < 0 ? 0 : (iy > 62 ? 62 : iy);
    c.nw = plane + ((size_t)iy * 64 + ix) * 40;
    c.ne = c.nw + 40;
    c.sw = c.nw + 64 * 40;
    c.se = c.sw + 40;
    if (!(c.m_nw && c.m_ne && c.m_sw && c.m_se))
    {
        const float* safe = plane + ((size_t)cy * 64 + cx) * 40;
        c.nw = c.m_nw ? c.nw : safe;
        c.ne = c.m_ne ? c.ne : safe;
        c.sw = c.m_sw ? c.sw : safe;
        c.se = c.m_se ? c.se : safe;
    }
    return c;
}

__device__ float sample(const Corners& c, unsigned int ch)
{
    const float v_nw = c.m_nw ? c.nw[ch] : 0.0f;
    const float v_ne = c.m_ne ? c.ne[ch] : 0.0f;
    const float v_sw = c.m_sw ? c.sw[ch] : 0.0f;
    const float v_se = c.m_se ? c.se[ch] : 0.0f;
    return v_nw * c.w_nw + v_ne * c.w_ne + v_sw * c.w_sw + v_se * c.w_se;
}

__device__ void stage(float* dst, const float* src, unsigned int count)
{
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x)
    {
        dst[i] = src[i];
    }
    __syncthreads();
}

// TriplaneNeRFRenderer.query_triplane fused, one point per thread (128 per block): the
// points come from points (3 floats each) or, when points is null, from the grid
// (lin[i / s^2], lin[i / s % s], lin[i % s]); weights holds the MLP panels in layer order
// (64 x 120, 8 x 64 x 64, the last padded to 16 x 64), biases 9 x 64 then 16
extern "C" __global__ void tripo_query(const float* planes, const float* points, const float* lin,
                                       unsigned int grid_size, unsigned int n, const float* weights,
                                       const float* biases, float low, float span, float density_bias, float* density,
                                       float* color)
{
    __shared__ float ws[64 * 120];
    const unsigned int index = blockIdx.x * 128 + threadIdx.x;
    const bool active = index < n;
    float p[3] = {0.0f, 0.0f, 0.0f};
    if (active)
    {
        if (points != 0)
        {
            p[0] = points[3 * (size_t)index];
            p[1] = points[3 * (size_t)index + 1];
            p[2] = points[3 * (size_t)index + 2];
        }
        else
        {
            p[0] = lin[index / (grid_size * grid_size)];
            p[1] = lin[(index / grid_size) % grid_size];
            p[2] = lin[index % grid_size];
        }
    }
    float q[3];
#pragma unroll
    for (int d = 0; d < 3; ++d)
    {
        q[d] = (p[d] - low) / span * 2.0f + -1.0f;
    }

    float acc[64];
#pragma unroll
    for (int j = 0; j < 64; ++j)
    {
        acc[j] = 0.0f;
    }
    stage(ws, weights, 64 * 120);
    const unsigned int plane_size = 64 * 64 * 40;
    const float gxs[3] = {q[0], q[0], q[1]};
    const float gys[3] = {q[1], q[2], q[2]};
    for (unsigned int np = 0; np < 3; ++np)
    {
        const Corners c = corners(planes + np * plane_size, gxs[np], gys[np]);
        for (unsigned int ch = 0; ch < 40; ++ch)
        {
            const float f = sample(c, ch);
            const unsigned int k = np * 40 + ch;
#pragma unroll
            for (int j = 0; j < 64; ++j)
            {
                acc[j] = fmaf(f, ws[(j / 16) * 120 * 16 + k * 16 + j % 16], acc[j]);
            }
        }
    }
    const float* w = weights + 64 * 120;
    const float* b = biases;
    for (int layer = 0; layer < 9; ++layer)
    {
        float h[64];
#pragma unroll
        for (int j = 0; j < 64; ++j)
        {
            const float x = acc[j] + b[j];
            h[j] = x / (1.0f + expf(-x));
            acc[j] = 0.0f;
        }
        b += 64;
        if (layer == 8)
        {
            stage(ws, w, 16 * 64);
            float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < 64; ++k)
            {
#pragma unroll
                for (int r = 0; r < 4; ++r)
                {
                    out[r] = fmaf(h[k], ws[k * 16 + r], out[r]);
                }
            }
            if (active)
            {
                density[index] = expf(out[0] + b[0] + density_bias);
                if (color != 0)
                {
#pragma unroll
                    for (int r = 0; r < 3; ++r)
                    {
                        color[3 * (size_t)index + r] = 1.0f / (1.0f + expf(-(out[1 + r] + b[1 + r])));
                    }
                }
            }
            return;
        }
        stage(ws, w, 64 * 64);
        w += 64 * 64;
#pragma unroll
        for (int k = 0; k < 64; ++k)
        {
#pragma unroll
            for (int j = 0; j < 64; ++j)
            {
                acc[j] = fmaf(h[k], ws[(j / 16) * 64 * 16 + k * 16 + j % 16], acc[j]);
            }
        }
    }
}
