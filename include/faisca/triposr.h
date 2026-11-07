#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/image.h"
#include "faisca/runner.h"
#include "faisca/thread.h"
#include "faisca/torch_ckpt.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// a float32 linear layer packed for gemm_f32_packed
struct TripoLinear
{
    const float* panels;
    size_t n_in;
    size_t n_out;
};

struct TripoVitLayer
{
    const float* ln1_w;
    const float* ln1_b;
    TripoLinear q;
    const float* q_b;
    TripoLinear k;
    const float* k_b;
    TripoLinear v;
    const float* v_b;
    TripoLinear out;
    const float* out_b;
    const float* ln2_w;
    const float* ln2_b;
    TripoLinear up;
    const float* up_b;
    TripoLinear down;
    const float* down_b;
};

struct TripoBlock
{
    const float* ln1_w;
    const float* ln1_b;
    TripoLinear self_q;
    TripoLinear self_k;
    TripoLinear self_v;
    TripoLinear self_out;
    const float* self_out_b;
    const float* ln2_w;
    const float* ln2_b;
    TripoLinear cross_q;
    TripoLinear cross_k;
    TripoLinear cross_v;
    TripoLinear cross_out;
    const float* cross_out_b;
    const float* ln3_w;
    const float* ln3_b;
    TripoLinear ff_in;
    const float* ff_in_b;
    TripoLinear ff_out;
    const float* ff_out_b;
};

// stabilityai/TripoSR model.ckpt: DINO ViT-B/16 image encoder, triplane transformer,
// upsampler and NeRF MLP (config.yaml of that release)
class TripoModel
{
public:
    static constexpr uint32_t VIT_DIM = 768;
    static constexpr uint32_t VIT_HEADS = 12;
    static constexpr uint32_t VIT_LAYERS = 12;
    static constexpr uint32_t VIT_FF = 3072;
    static constexpr uint32_t VIT_PATCH = 16;
    static constexpr uint32_t VIT_POS_SIDE = 14;
    static constexpr float VIT_EPS = 1e-12f;
    static constexpr uint32_t IMAGE_SIZE = 512;

    static constexpr uint32_t PLANE_SIZE = 32;
    static constexpr uint32_t PLANE_TOKENS = 3 * PLANE_SIZE * PLANE_SIZE;
    static constexpr uint32_t DIM = 1024;
    static constexpr uint32_t HEADS = 16;
    static constexpr uint32_t BLOCKS = 16;
    static constexpr uint32_t FF = 4096;
    static constexpr uint32_t GROUPS = 32;
    static constexpr float GROUP_EPS = 1e-6f;
    static constexpr float LN_EPS = 1e-5f;

    static constexpr uint32_t CODE_CHANNELS = 40;
    static constexpr uint32_t CODE_SIZE = 2 * PLANE_SIZE;

    static constexpr uint32_t MLP_IN = 3 * CODE_CHANNELS;
    static constexpr uint32_t MLP_WIDTH = 64;
    static constexpr uint32_t MLP_LAYERS = 10;
    static constexpr float RADIUS = 0.87f;
    static constexpr float DENSITY_BIAS = -1.0f;

    TripoModel() = default;

    TripoModel(const TripoModel&) = delete;
    TripoModel& operator=(const TripoModel&) = delete;

    bool open(const char* path, Error& error);

    const TripoVitLayer& vit_layer(uint32_t index) const
    {
        return m_vit_layers[index];
    }

    const float* vit_cls() const
    {
        return m_vit_cls;
    }

    const float* vit_pos() const
    {
        return m_vit_pos;
    }

    const TripoLinear& vit_patch() const
    {
        return m_vit_patch;
    }

    const float* vit_patch_b() const
    {
        return m_vit_patch_b;
    }

    const float* vit_norm_w() const
    {
        return m_vit_norm_w;
    }

    const float* vit_norm_b() const
    {
        return m_vit_norm_b;
    }

    // [3][DIM][PLANE_SIZE][PLANE_SIZE]
    // every packed linear layer lives in this one block
    const float* panel_data() const
    {
        return m_panels.data();
    }

    size_t panel_floats() const
    {
        return m_panels.size();
    }

    const float* plane_embeddings() const
    {
        return m_planes;
    }

    const float* group_norm_w() const
    {
        return m_group_w;
    }

    const float* group_norm_b() const
    {
        return m_group_b;
    }

    const TripoLinear& proj_in() const
    {
        return m_proj_in;
    }

    const float* proj_in_b() const
    {
        return m_proj_in_b;
    }

    const TripoBlock& block(uint32_t index) const
    {
        return m_blocks[index];
    }

    const TripoLinear& proj_out() const
    {
        return m_proj_out;
    }

    const float* proj_out_b() const
    {
        return m_proj_out_b;
    }

    // the transposed convolution as CODE_CHANNELS * 4 rows of DIM: row (co * 2 + ky) * 2 + kx
    const TripoLinear& upsample() const
    {
        return m_upsample;
    }

    const float* upsample_b() const
    {
        return m_upsample_b;
    }

    // NeRFMLP: MLP_LAYERS linear layers, SiLU between them; the last one padded from 4 to
    // F32_PANEL outputs with zero rows (density, then the three color features)
    const TripoLinear& mlp(uint32_t index) const
    {
        return m_mlp[index];
    }

    const float* mlp_b(uint32_t index) const
    {
        return m_mlp_b[index];
    }

private:
    const float* f32(const char* name, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3, Error& error);
    bool linear(const char* name, uint64_t n_out, uint64_t n_in, TripoLinear& w, const float** bias, Error& error);
    void add_linear(const float* rows, size_t n_out, size_t n_in, TripoLinear& w);
    bool pack_linears(Error& error);

    static constexpr size_t MAX_LINEARS = 256;
    struct Pending
    {
        TripoLinear* linear;
        const float* rows;
    };
    Pending m_pending[MAX_LINEARS] = {};
    size_t m_n_pending = 0;
    Array<float> m_panels;

    CkptFile m_file;
    TripoVitLayer m_vit_layers[VIT_LAYERS] = {};
    const float* m_vit_cls = nullptr;
    const float* m_vit_pos = nullptr;
    TripoLinear m_vit_patch;
    const float* m_vit_patch_b = nullptr;
    const float* m_vit_norm_w = nullptr;
    const float* m_vit_norm_b = nullptr;

    const float* m_planes = nullptr;
    const float* m_group_w = nullptr;
    const float* m_group_b = nullptr;
    TripoLinear m_proj_in;
    const float* m_proj_in_b = nullptr;
    TripoBlock m_blocks[BLOCKS] = {};
    TripoLinear m_proj_out;
    const float* m_proj_out_b = nullptr;
    Array<float> m_upsample_rows;
    TripoLinear m_upsample;
    TripoLinear m_mlp[MLP_LAYERS] = {};
    const float* m_mlp_b[MLP_LAYERS] = {};
    Array<float> m_mlp_last;
    const float* m_upsample_b = nullptr;
};

// pieces shared by the CPU and CUDA runners: the CLS and resampled position embeddings
// ((IMAGE_SIZE / VIT_PATCH)^2 + 1 rows), the normalized patches p in [first, last) of an
// rgb image (3 x 16 x 16 values each), the plane embeddings as PLANE_TOKENS rows of DIM,
// and their GroupNorm for the groups in [first, last)
void tripo_vit_positions(const TripoModel& model, float* pos);
void tripo_patches(const float* rgb, size_t first, size_t last, float* patches);
void tripo_plane_tokens(const TripoModel& model, float* tokens);
void tripo_group_norm(const TripoModel& model, const float* tokens, size_t first, size_t last, float* out);

// pos is side x side rows of dim values; out gets out_side x out_side rows, resampled as
// torch.nn.functional.interpolate(mode="bicubic", align_corners=False, scale_factor=scale)
// does on the CPU in float32
void tripo_bicubic_rows(const float* pos, uint32_t side, uint32_t dim, double scale, uint32_t out_side, float* out);

// TripoSR run.py and ImagePreprocessor without rembg: an image with alpha is cut to its
// foreground (resize_foreground with ratio, the box of alpha > 0 as numpy slices it),
// blended on grey 0.5 and rounded to 8 bits; then every image goes to rgb as
// IMAGE_SIZE x IMAGE_SIZE x 3 floats through F.interpolate(bilinear, antialias=True)
bool tripo_prepare_image(const Image& image, float foreground_ratio, Array<float>& rgb, Error& error);

// torchmcubes mcubes_cpu (commit a9ac95a) of vol, resolution^3 values indexed
// [z][y][x], at level 0: points gets x, y, z in grid units per unique vertex (first seen
// order of the serial loop, merged by exact position), faces three indices per triangle
bool tripo_marching_cubes(ThreadPool& pool, const float* vol, uint32_t resolution, Array<float>& points,
                          Array<uint32_t>& faces, Error& error);

struct TripoMesh
{
    Array<float> positions; // x, y, z per vertex, in (-RADIUS, RADIUS)
    Array<float> colors;    // r, g, b per vertex in [0, 1]
    Array<uint32_t> faces;  // three vertex indices per triangle
};

// the grid axis of TSR.extract_mesh (resolution values in (-RADIUS, RADIUS)) and the
// marching cubes points (grid units, x y z) back to positions in (-RADIUS, RADIUS)
void tripo_grid_axis(uint32_t resolution, float* lin);
void tripo_mesh_positions(const float* grid_points, size_t n, uint32_t resolution, float* positions);

// Wavefront OBJ with a colour per vertex ("v x y z r g b", as trimesh writes them)
bool tripo_write_obj(const char* path, const TripoMesh& mesh, Error& error);

class TripoRunner
{
public:
    TripoRunner() = default;

    TripoRunner(const TripoRunner&) = delete;
    TripoRunner& operator=(const TripoRunner&) = delete;

    bool init(const TripoModel& model, ThreadPool& pool, Error& error);

    void set_trace(RunnerTrace callback, void* user)
    {
        m_trace = callback;
        m_trace_user = user;
    }

    // rgb: IMAGE_SIZE x IMAGE_SIZE pixels of R, G, B in [0, 1], as TripoSR's
    // ImagePreprocessor gives them; fills image_tokens()
    bool encode_image(const float* rgb, Error& error);

    // image_token_count() rows of VIT_DIM values: the CLS token, then the patches
    const float* image_tokens() const
    {
        return m_image.data();
    }

    size_t image_token_count() const
    {
        return m_n_image;
    }

    // the triplane transformer and upsampler over image_tokens(); fills scene_code()
    bool build_triplane(Error& error);

    // [3][CODE_CHANNELS][CODE_SIZE][CODE_SIZE], TripoSR's scene_codes[0]
    const float* scene_code() const
    {
        return m_code.data();
    }

    // TriplaneNeRFRenderer.query_triplane over scene_code(): density_act (exp of the
    // density plus DENSITY_BIAS) and, if color is not null, the sigmoid colors (3 per
    // point) at n points (3 floats each) in (-RADIUS, RADIUS)
    bool query(const float* points, size_t n, float* density, float* color, Error& error);

    // TSR.extract_mesh: density on a resolution^3 grid, torchmcubes marching cubes of
    // density - threshold at 0, vertices back in (-RADIUS, RADIUS), colors at the vertices
    bool extract_mesh(uint32_t resolution, float threshold, TripoMesh& mesh, Error& error);

private:
    static constexpr size_t ATTENTION_QUERIES = 48;
    static constexpr size_t QUERY_CHUNK = 1024;

    typedef void (*PointSource)(const void* user, size_t index, float out[3]);
    bool query_points(PointSource source, const void* user, size_t n, float* density, float* color, Error& error);

    void trace(const char* name, int layer, const float* data, size_t count) const;
    // x_stride 0 means w.n_in
    void matmul(const TripoLinear& w, const float* bias, const float* x, size_t n_tokens, float* out, size_t x_stride = 0);
    void norm_rows(const float* x, const float* w, const float* b, float eps, size_t rows, size_t width, float* out);
    void attention(const float* q, const float* k, const float* v, size_t n_q, size_t n_kv, size_t heads,
                   size_t head_dim, float* out);

    const TripoModel* m_model = nullptr;
    ThreadPool* m_pool = nullptr;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;
    size_t m_n_image = 0;

    Array<float> m_pos;
    Array<float> m_patches;
    Array<float> m_image;
    Array<float> m_residual;
    Array<float> m_x;
    Array<float> m_h;
    Array<float> m_q;
    Array<float> m_k;
    Array<float> m_v;
    Array<float> m_attn;
    Array<float> m_up;
    Array<float> m_scores;
    Array<float> m_k_tiles;
    Array<float> m_v_heads;
    Array<float> m_gemm_scratch;
    Array<float> m_planes_last; // scene_code() as [3][CODE_SIZE][CODE_SIZE][CODE_CHANNELS]
    Array<float> m_query;       // per thread: QUERY_CHUNK x (MLP_IN + 2 MLP_WIDTH + F32_PANEL)
    Array<float> m_code;
};

}
