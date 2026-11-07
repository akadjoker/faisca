// TripoSR on the GPU in float32. Built with --fmad=false: every fused multiply-add is an
// explicit fmaf, so the products of tripo_gemm follow the CPU gemm_f32_packed order.

#define PANEL 16

// out[t][o] for t < n_tokens, o < n_out: acc = fmaf chain over k from 0 in k order (as
// gemm_f32_packed), then acc + bias[o] if bias, then out[t][o] + that if accumulate.
// panels: rows packed as pack_f32_panels does ([panel][k][16]). n_in, x_stride and
// out_stride multiples of 4 (n_in of 8), 16 byte aligned rows. 256 threads, a tile of
// 128 tokens x 128 outputs (8 panels), 8 x 8 outputs per thread, k in steps of 8 through
// two shared buffers (the SIMT tiling of CUTLASS's sgemm and cuBLAS's ampere_sgemm_128x128).
extern "C" __global__ void __launch_bounds__(256, 2)
    tripo_gemm(const float* x, unsigned int x_stride, const float* panels, unsigned int n_in, unsigned int n_out,
               unsigned int n_tokens, const float* bias, float* out, unsigned int out_stride, unsigned int accumulate)
{
    __shared__ __align__(16) float as[2][8][132];
    __shared__ __align__(16) float bs[2][8][128];
    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid % 16;
    const unsigned int ty = tid / 16;
    const unsigned int t0 = blockIdx.y * 128;
    const unsigned int panel0 = blockIdx.x * 8;
    const unsigned int n_panels = n_out / PANEL;

    const unsigned int a_t = tid / 2;
    const unsigned int a_k = (tid % 2) * 4;
    const unsigned int b_p = tid / 32;
    const unsigned int b_k = (tid % 32) / 4;
    const unsigned int b_c = (tid % 4) * 4;
    const bool a_ok = t0 + a_t < n_tokens;
    const bool b_ok = panel0 + b_p < n_panels;
    const float* a_src = x + (size_t)(a_ok ? t0 + a_t : 0) * x_stride + a_k;
    const float* b_src = panels + ((size_t)(b_ok ? panel0 + b_p : 0) * n_in + b_k) * PANEL + b_c;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 a_next = a_ok ? *(const float4*)a_src : zero;
    float4 b_next = b_ok ? *(const float4*)b_src : zero;
    as[0][a_k + 0][a_t] = a_next.x;
    as[0][a_k + 1][a_t] = a_next.y;
    as[0][a_k + 2][a_t] = a_next.z;
    as[0][a_k + 3][a_t] = a_next.w;
    *(float4*)&bs[0][b_k][b_p * 16 + b_c] = b_next;
    __syncthreads();

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            acc[i][j] = 0.0f;
        }
    }
    unsigned int cur = 0;
    for (unsigned int k0 = 0; k0 < n_in; k0 += 8)
    {
        const bool more = k0 + 8 < n_in;
        if (more)
        {
            a_next = a_ok ? *(const float4*)(a_src + k0 + 8) : zero;
            b_next = b_ok ? *(const float4*)(b_src + (size_t)(k0 + 8) * PANEL) : zero;
        }
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
        {
            const float4 a0 = *(const float4*)&as[cur][kk][ty * 4];
            const float4 a1 = *(const float4*)&as[cur][kk][64 + ty * 4];
            const float4 b0 = *(const float4*)&bs[cur][kk][tx * 4];
            const float4 b1 = *(const float4*)&bs[cur][kk][64 + tx * 4];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        if (more)
        {
            const unsigned int next = cur ^ 1u;
            as[next][a_k + 0][a_t] = a_next.x;
            as[next][a_k + 1][a_t] = a_next.y;
            as[next][a_k + 2][a_t] = a_next.z;
            as[next][a_k + 3][a_t] = a_next.w;
            *(float4*)&bs[next][b_k][b_p * 16 + b_c] = b_next;
        }
        __syncthreads();
        cur ^= 1u;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const unsigned int t = t0 + (i / 4) * 64 + ty * 4 + i % 4;
        if (t >= n_tokens)
        {
            continue;
        }
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const unsigned int o = panel0 * PANEL + h * 64 + tx * 4;
            if (o >= n_out)
            {
                continue;
            }
            float4 v = make_float4(acc[i][h * 4], acc[i][h * 4 + 1], acc[i][h * 4 + 2], acc[i][h * 4 + 3]);
            if (bias != 0)
            {
                const float4 bv = *(const float4*)(bias + o);
                v.x = v.x + bv.x;
                v.y = v.y + bv.y;
                v.z = v.z + bv.z;
                v.w = v.w + bv.w;
            }
            float4* dst = (float4*)(out + (size_t)t * out_stride + o);
            if (accumulate)
            {
                const float4 d = *dst;
                v.x = d.x + v.x;
                v.y = d.y + v.y;
                v.z = d.z + v.z;
                v.w = d.w + v.w;
            }
            *dst = v;
        }
    }
}

// Knuth's TwoSum: s + e == a + b exactly
__device__ void two_sum(float a, float b, float& s, float& e)
{
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

__device__ double warp_float_float_sum(float hi, float lo)
{
    for (int o = 16; o > 0; o /= 2)
    {
        const float h2 = __shfl_xor_sync(0xffffffffu, hi, o);
        const float l2 = __shfl_xor_sync(0xffffffffu, lo, o);
        float s, e;
        two_sum(hi, h2, s, e);
        e = e + (lo + l2);
        hi = s + e;
        lo = e - (hi - s);
    }
    return (double)hi + (double)lo;
}

// layer_norm (src/ops_whisper.cpp) of one row per warp, 8 rows per block of 256 threads,
// width a multiple of 128 up to 1024 held in registers. The CPU's double sums cost 1/64
// rate FP64 here (563 us for 3072 x 1024); compensated float-float sums (TwoSum, exact
// squares by fmaf) keep about 48 bits and gave the same outputs on random and model rows.
extern "C" __global__ void tripo_layer_norm(const float* x, const float* weight, const float* bias, float* out,
                                            unsigned int rows, unsigned int width, float eps)
{
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int r = blockIdx.x * 8 + threadIdx.x / 32;
    if (r >= rows)
    {
        return;
    }
    const unsigned int chunks = width / 128;
    const float* row = x + (size_t)r * width;
    float v[32];
    float hi = 0.0f;
    float lo = 0.0f;
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j)
    {
        if (j < chunks)
        {
            const float4 f = *(const float4*)(row + j * 128 + lane * 4);
            v[4 * j] = f.x;
            v[4 * j + 1] = f.y;
            v[4 * j + 2] = f.z;
            v[4 * j + 3] = f.w;
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                float s, e;
                two_sum(hi, v[4 * j + c], s, e);
                hi = s;
                lo = lo + e;
            }
        }
    }
    const float mean = (float)(warp_float_float_sum(hi, lo) / (double)width);
    hi = 0.0f;
    lo = 0.0f;
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j)
    {
        if (j < chunks)
        {
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const float centered = v[4 * j + c] - mean;
                const float p = centered * centered;
                float s, e;
                two_sum(hi, p, s, e);
                hi = s;
                lo = lo + e + fmaf(centered, centered, -p);
            }
        }
    }
    const float variance = (float)(warp_float_float_sum(hi, lo) / (double)width);
    const float scale = 1.0f / sqrtf(variance + eps);
    float* dst = out + (size_t)r * width;
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j)
    {
        if (j < chunks)
        {
            const unsigned int i = j * 128 + lane * 4;
            const float4 w = *(const float4*)(weight + i);
            const float4 b = *(const float4*)(bias + i);
            float4 o;
            o.x = (v[4 * j] - mean) * scale * w.x + b.x;
            o.y = (v[4 * j + 1] - mean) * scale * w.y + b.y;
            o.z = (v[4 * j + 2] - mean) * scale * w.z + b.z;
            o.w = (v[4 * j + 3] - mean) * scale * w.w + b.w;
            *(float4*)(dst + i) = o;
        }
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

// tile[64][64] of a head (rows r0.., columns head * 64..) transposed into dst[d][row],
// rows from limit on as zeros; a warp covers 32 rows of one float4 column
__device__ void load_transposed(float (*dst)[64], const float* src, unsigned int r0, unsigned int limit,
                                unsigned int stride, unsigned int head)
{
    for (unsigned int e = threadIdx.x; e < 1024; e += 128)
    {
        const unsigned int w = e / 32;
        const unsigned int row = (w % 2) * 32 + e % 32;
        const unsigned int c = (w / 2) * 4;
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r0 + row < limit)
        {
            f = *(const float4*)(src + (size_t)(r0 + row) * stride + head * 64 + c);
        }
        dst[c][row] = f.x;
        dst[c + 1][row] = f.y;
        dst[c + 2][row] = f.z;
        dst[c + 3][row] = f.w;
    }
}

// softmax(q k^T * scale) v for head_dim 64 in the FlashAttention-2 order (Dao 2023, as
// PyTorch's mem-efficient cutlass kernel): 64 queries per block of 128 threads, keys in
// steps of 64, the online softmax rescaled once per step. Thread (tq, tk) holds S and O
// for queries 4 tq.. and keys / dims 4 tk.. and 32 + 4 tk..
extern "C" __global__ void __launch_bounds__(128)
    tripo_flash_attention(const float* q, const float* k, const float* v, float* out, unsigned int n_q,
                          unsigned int n_kv, unsigned int stride, float scale)
{
    __shared__ __align__(16) float qt[64][64];
    __shared__ __align__(16) float kt[64][64];
    __shared__ __align__(16) float vs[64][64];
    const unsigned int tid = threadIdx.x;
    const unsigned int tq = tid / 8;
    const unsigned int tk = tid % 8;
    const unsigned int head = blockIdx.y;
    const unsigned int q0 = blockIdx.x * 64;
    const float minus_infinity = __int_as_float(0xff800000);
    load_transposed(qt, q, q0, n_q, stride, head);

    float o[4][8];
    float m[4];
    float l[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        m[i] = minus_infinity;
        l[i] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            o[i][j] = 0.0f;
        }
    }
    for (unsigned int k0 = 0; k0 < n_kv; k0 += 64)
    {
        __syncthreads();
        load_transposed(kt, k, k0, n_kv, stride, head);
        for (unsigned int e = tid; e < 1024; e += 128)
        {
            const unsigned int row = e / 16;
            const unsigned int c = (e % 16) * 4;
            float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (k0 + row < n_kv)
            {
                f = *(const float4*)(v + (size_t)(k0 + row) * stride + head * 64 + c);
            }
            *(float4*)&vs[row][c] = f;
        }
        __syncthreads();

        float s[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                s[i][j] = 0.0f;
            }
        }
#pragma unroll 8
        for (int d = 0; d < 64; ++d)
        {
            const float4 a = *(const float4*)&qt[d][tq * 4];
            const float4 b0 = *(const float4*)&kt[d][tk * 4];
            const float4 b1 = *(const float4*)&kt[d][32 + tk * 4];
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    s[i][j] = fmaf(av[i], bv[j], s[i][j]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float top = minus_infinity;
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                const unsigned int key = k0 + (j / 4) * 32 + tk * 4 + j % 4;
                s[i][j] = key < n_kv ? s[i][j] * scale : minus_infinity;
                top = fmaxf(top, s[i][j]);
            }
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 4));
            const float m_new = fmaxf(m[i], top);
            const float alpha = expf(m[i] - m_new);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                s[i][j] = expf(s[i][j] - m_new);
                sum = sum + s[i][j];
            }
            sum = sum + __shfl_xor_sync(0xffffffffu, sum, 1);
            sum = sum + __shfl_xor_sync(0xffffffffu, sum, 2);
            sum = sum + __shfl_xor_sync(0xffffffffu, sum, 4);
            l[i] = l[i] * alpha + sum;
            m[i] = m_new;
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                o[i][j] = o[i][j] * alpha;
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            const unsigned int key = (j / 4) * 32 + tk * 4 + j % 4;
            *(float4*)&kt[key][tq * 4] = make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
        }
        __syncthreads();
#pragma unroll 8
        for (int key = 0; key < 64; ++key)
        {
            const float4 p = *(const float4*)&kt[key][tq * 4];
            const float4 v0 = *(const float4*)&vs[key][tk * 4];
            const float4 v1 = *(const float4*)&vs[key][32 + tk * 4];
            const float pv[4] = {p.x, p.y, p.z, p.w};
            const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    o[i][j] = fmaf(pv[i], vv[j], o[i][j]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const unsigned int query = q0 + tq * 4 + i;
        if (query >= n_q)
        {
            continue;
        }
        const float inverse = 1.0f / l[i];
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            float4 r;
            r.x = o[i][h * 4] * inverse;
            r.y = o[i][h * 4 + 1] * inverse;
            r.z = o[i][h * 4 + 2] * inverse;
            r.w = o[i][h * 4 + 3] * inverse;
            *(float4*)(out + (size_t)query * stride + head * 64 + h * 32 + tk * 4) = r;
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
