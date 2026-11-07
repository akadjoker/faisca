#include "faisca/triposr_cuda.h"

#include "faisca/ops.h"
#include "kernels_ptx.h"

#include <cmath>
#include <cstring>

namespace faisca
{

namespace
{

size_t aligned(size_t floats)
{
    return (floats + 63) / 64 * 64;
}

}

TripoCudaRunner::~TripoCudaRunner()
{
    release();
}

void TripoCudaRunner::release()
{
    if (m_device == nullptr)
    {
        return;
    }
    Error ignored;
    if (m_grid.address != 0)
    {
        m_device->destroy_buffer(m_grid, ignored);
    }
    if (m_work.address != 0)
    {
        m_device->destroy_buffer(m_work, ignored);
    }
    if (m_params.address != 0)
    {
        m_device->destroy_buffer(m_params, ignored);
    }
    if (m_weights.address != 0)
    {
        m_device->destroy_buffer(m_weights, ignored);
    }
    if (m_module.handle != nullptr)
    {
        m_device->destroy_module(m_module, ignored);
    }
    if (m_stream.handle != nullptr)
    {
        m_device->destroy_stream(m_stream, ignored);
    }
    m_device = nullptr;
}

uint64_t TripoCudaRunner::panels(const TripoLinear& w) const
{
    return m_weights.address + static_cast<uint64_t>(w.panels - m_model->panel_data()) * sizeof(float);
}

bool TripoCudaRunner::init(const TripoModel& model, CudaDevice& device, ThreadPool& pool, Error& error)
{
    release();
    m_model = &model;
    m_device = &device;
    m_pool = &pool;
    if (!device.create_stream(m_stream, error) || !device.load_module(ptx_triposr, m_module, error) ||
        !device.find_kernel(m_module, "tripo_gemm", m_gemm, error) ||
        !device.find_kernel(m_module, "tripo_layer_norm", m_layer_norm, error) ||
        !device.find_kernel(m_module, "tripo_gelu", m_gelu, error) ||
        !device.find_kernel(m_module, "tripo_geglu", m_geglu, error) ||
        !device.find_kernel(m_module, "tripo_add", m_add, error) ||
        !device.find_kernel(m_module, "tripo_attention", m_attention, error) ||
        !device.find_kernel(m_module, "tripo_upsample", m_upsample, error) ||
        !device.find_kernel(m_module, "tripo_query", m_query, error))
    {
        return false;
    }
    const size_t weight_bytes = model.panel_floats() * sizeof(float);
    if (!device.create_buffer(weight_bytes, m_weights, error) ||
        !device.upload(m_weights, model.panel_data(), weight_bytes, error))
    {
        return false;
    }

    // every small tensor into one buffer: offsets first, addresses once it exists
    struct Pending
    {
        uint64_t* field;
        size_t offset;
    };
    Array<Pending> pending;
    Array<float> staging;
    bool ok = true;
    auto put = [&](uint64_t& field, const float* data, size_t count)
    {
        const size_t offset = staging.size();
        ok = ok && staging.resize_uninitialized(offset + aligned(count)) && pending.push_back(Pending{&field, offset});
        if (ok)
        {
            std::memcpy(staging.data() + offset, data, count * sizeof(float));
        }
    };
    const size_t vit_dim = TripoModel::VIT_DIM;
    const size_t dim = TripoModel::DIM;
    const size_t n_image = (TripoModel::IMAGE_SIZE / TripoModel::VIT_PATCH) * (TripoModel::IMAGE_SIZE / TripoModel::VIT_PATCH) + 1;
    const size_t n_plane = TripoModel::PLANE_TOKENS;

    Array<float> host;
    if (!host.resize_uninitialized(2 * n_plane * dim))
    {
        error.set("out of memory preparing the TripoSR GPU parameters");
        return false;
    }
    tripo_vit_positions(model, host.data());
    put(m_vit_pos, host.data(), n_image * vit_dim);
    put(m_vit_cls, model.vit_cls(), vit_dim);
    put(m_vit_patch_b, model.vit_patch_b(), vit_dim);
    put(m_vit_norm_w, model.vit_norm_w(), vit_dim);
    put(m_vit_norm_b, model.vit_norm_b(), vit_dim);
    m_vit_patch = panels(model.vit_patch());
    for (uint32_t il = 0; il < TripoModel::VIT_LAYERS; ++il)
    {
        const TripoVitLayer& l = model.vit_layer(il);
        Vit& g = m_vit[il];
        put(g.ln1_w, l.ln1_w, vit_dim);
        put(g.ln1_b, l.ln1_b, vit_dim);
        put(g.ln2_w, l.ln2_w, vit_dim);
        put(g.ln2_b, l.ln2_b, vit_dim);
        put(g.q_b, l.q_b, vit_dim);
        put(g.k_b, l.k_b, vit_dim);
        put(g.v_b, l.v_b, vit_dim);
        put(g.out_b, l.out_b, vit_dim);
        put(g.up_b, l.up_b, TripoModel::VIT_FF);
        put(g.down_b, l.down_b, vit_dim);
        g.q = panels(l.q);
        g.k = panels(l.k);
        g.v = panels(l.v);
        g.out = panels(l.out);
        g.up = panels(l.up);
        g.down = panels(l.down);
    }
    float* tokens = host.data();
    float* normalized = host.data() + n_plane * dim;
    tripo_plane_tokens(model, tokens);
    tripo_group_norm(model, tokens, 0, TripoModel::GROUPS, normalized);
    put(m_plane_tokens, tokens, n_plane * dim);
    put(m_plane_norm, normalized, n_plane * dim);
    put(m_proj_in_b, model.proj_in_b(), dim);
    put(m_proj_out_b, model.proj_out_b(), dim);
    m_proj_in = panels(model.proj_in());
    m_proj_out = panels(model.proj_out());
    for (uint32_t ib = 0; ib < TripoModel::BLOCKS; ++ib)
    {
        const TripoBlock& b = model.block(ib);
        Block& g = m_blocks[ib];
        put(g.ln1_w, b.ln1_w, dim);
        put(g.ln1_b, b.ln1_b, dim);
        put(g.ln2_w, b.ln2_w, dim);
        put(g.ln2_b, b.ln2_b, dim);
        put(g.ln3_w, b.ln3_w, dim);
        put(g.ln3_b, b.ln3_b, dim);
        put(g.self_out_b, b.self_out_b, dim);
        put(g.cross_out_b, b.cross_out_b, dim);
        put(g.ff_in_b, b.ff_in_b, 2 * TripoModel::FF);
        put(g.ff_out_b, b.ff_out_b, dim);
        g.self_q = panels(b.self_q);
        g.self_k = panels(b.self_k);
        g.self_v = panels(b.self_v);
        g.self_out = panels(b.self_out);
        g.cross_q = panels(b.cross_q);
        g.cross_k = panels(b.cross_k);
        g.cross_v = panels(b.cross_v);
        g.cross_out = panels(b.cross_out);
        g.ff_in = panels(b.ff_in);
        g.ff_out = panels(b.ff_out);
    }
    put(m_upsample_b, model.upsample_b(), TripoModel::CODE_CHANNELS);
    m_upsample_w = panels(model.upsample());

    // tripo_query reads the MLP panels as one run in layer order
    const float* run = model.mlp(0).panels;
    for (uint32_t l = 0; l < TripoModel::MLP_LAYERS; ++l)
    {
        const TripoLinear& w = model.mlp(l);
        if (w.panels != run)
        {
            error.set("TripoSR MLP panels are not contiguous");
            return false;
        }
        run += w.n_in * w.n_out;
    }
    m_mlp_w = panels(model.mlp(0));
    float biases[9 * TripoModel::MLP_WIDTH + F32_PANEL];
    for (uint32_t l = 0; l + 1 < TripoModel::MLP_LAYERS; ++l)
    {
        std::memcpy(biases + l * TripoModel::MLP_WIDTH, model.mlp_b(l), TripoModel::MLP_WIDTH * sizeof(float));
    }
    std::memcpy(biases + 9 * TripoModel::MLP_WIDTH, model.mlp_b(TripoModel::MLP_LAYERS - 1), F32_PANEL * sizeof(float));
    put(m_mlp_b, biases, 9 * TripoModel::MLP_WIDTH + F32_PANEL);
    if (!ok || !device.create_buffer(staging.size() * sizeof(float), m_params, error) ||
        !device.upload(m_params, staging.data(), staging.size() * sizeof(float), error))
    {
        if (!ok)
        {
            error.set("out of memory preparing the TripoSR GPU parameters");
        }
        return false;
    }
    for (const Pending& p : pending)
    {
        *p.field = m_params.address + p.offset * sizeof(float);
    }

    const size_t wide = n_plane * dim;
    const size_t up = n_plane * 2 * TripoModel::FF;
    const size_t code = 3 * TripoModel::CODE_CHANNELS * TripoModel::CODE_SIZE * TripoModel::CODE_SIZE;
    const size_t sizes[] = {aligned((n_image - 1) * 3 * TripoModel::VIT_PATCH * TripoModel::VIT_PATCH),
                            aligned(n_image * vit_dim), aligned(wide), aligned(wide), aligned(wide), aligned(wide),
                            aligned(wide), aligned(wide), aligned(up), aligned(code), aligned(code)};
    uint64_t* fields[] = {&m_patches, &m_image, &m_x, &m_h, &m_q, &m_k, &m_v, &m_attn, &m_up, &m_code, &m_planes};
    size_t total = 0;
    for (size_t s : sizes)
    {
        total += s;
    }
    if (!device.create_buffer(total * sizeof(float), m_work, error))
    {
        return false;
    }
    size_t offset = 0;
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i)
    {
        *fields[i] = m_work.address + offset * sizeof(float);
        offset += sizes[i];
    }
    return true;
}

bool TripoCudaRunner::trace(const char* name, int layer, uint64_t address, size_t count, Error& error)
{
    if (m_trace == nullptr)
    {
        return true;
    }
    if (!m_host.resize_uninitialized(count) || !m_device->synchronize(m_stream, error))
    {
        return false;
    }
    CudaBuffer view = m_work.address <= address && address < m_work.address + m_work.bytes ? m_work : m_grid;
    if (!m_device->download(view, m_host.data(), count * sizeof(float), error, address - view.address))
    {
        return false;
    }
    m_trace(m_trace_user, name, layer, m_host.data(), count);
    return true;
}

bool TripoCudaRunner::gemm(uint64_t x, uint32_t x_stride, uint64_t w, uint32_t n_in, uint32_t n_out, uint32_t n_tokens,
                           uint64_t bias, uint64_t out, bool accumulate, Error& error)
{
    uint32_t out_stride = n_out;
    uint32_t acc = accumulate ? 1u : 0u;
    void* arguments[] = {&x, &x_stride, &w, &n_in, &n_out, &n_tokens, &bias, &out, &out_stride, &acc};
    CudaLaunch launch;
    launch.grid_x = (n_out / static_cast<uint32_t>(F32_PANEL) + 3) / 4;
    launch.grid_y = (n_tokens + 63) / 64;
    launch.block_x = 256;
    return m_device->launch(m_gemm, m_stream, launch, arguments, error);
}

bool TripoCudaRunner::layer_norm(uint64_t x, uint64_t w, uint64_t b, uint64_t out, uint32_t rows, uint32_t width,
                                 float eps, Error& error)
{
    void* arguments[] = {&x, &w, &b, &out, &width, &eps};
    CudaLaunch launch;
    launch.grid_x = rows;
    launch.block_x = 256;
    return m_device->launch(m_layer_norm, m_stream, launch, arguments, error);
}

bool TripoCudaRunner::attention(uint64_t q, uint64_t k, uint64_t v, uint64_t out, uint32_t n_q, uint32_t n_kv,
                                uint32_t heads, Error& error)
{
    uint32_t stride = heads * 64;
    float scale = 1.0f / std::sqrt(64.0f);
    void* arguments[] = {&q, &k, &v, &out, &n_q, &n_kv, &stride, &scale};
    CudaLaunch launch;
    launch.grid_x = (n_q + 127) / 128;
    launch.grid_y = heads;
    launch.block_x = 128;
    return m_device->launch(m_attention, m_stream, launch, arguments, error);
}

bool TripoCudaRunner::elementwise(const CudaKernel& kernel, void** arguments, uint32_t count, Error& error)
{
    CudaLaunch launch;
    launch.grid_x = (count + 255) / 256;
    launch.block_x = 256;
    return m_device->launch(kernel, m_stream, launch, arguments, error);
}

bool TripoCudaRunner::encode_image(const float* rgb, Error& error)
{
    if (m_device == nullptr)
    {
        error.set("TripoCudaRunner used before init");
        return false;
    }
    const uint32_t side = TripoModel::IMAGE_SIZE / TripoModel::VIT_PATCH;
    const uint32_t n_patches = side * side;
    const uint32_t n = n_patches + 1;
    const uint32_t dim = TripoModel::VIT_DIM;
    const uint32_t taps = 3 * TripoModel::VIT_PATCH * TripoModel::VIT_PATCH;
    if (!m_host.resize_uninitialized(size_t(n_patches) * taps))
    {
        error.set("out of memory for the patches");
        return false;
    }
    tripo_patches(rgb, 0, n_patches, m_host.data());
    if (!m_device->synchronize(m_stream, error) ||
        !m_device->upload(m_work, m_host.data(), size_t(n_patches) * taps * sizeof(float), error,
                          m_patches - m_work.address) ||
        !m_device->upload(m_work, m_model->vit_cls(), dim * sizeof(float), error, m_image - m_work.address) ||
        !gemm(m_patches, taps, m_vit_patch, taps, dim, n_patches, m_vit_patch_b, m_image + dim * sizeof(float), false,
              error))
    {
        return false;
    }
    uint32_t count = n * dim;
    void* add_args[] = {&m_image, &m_vit_pos, &count};
    if (!elementwise(m_add, add_args, count, error) || !trace("vit_embed", -1, m_image, count, error))
    {
        return false;
    }
    for (uint32_t il = 0; il < TripoModel::VIT_LAYERS; ++il)
    {
        const Vit& l = m_vit[il];
        uint32_t ff_count = n * TripoModel::VIT_FF;
        void* gelu_args[] = {&m_up, &ff_count};
        if (!layer_norm(m_image, l.ln1_w, l.ln1_b, m_h, n, dim, TripoModel::VIT_EPS, error) ||
            !gemm(m_h, dim, l.q, dim, dim, n, l.q_b, m_q, false, error) ||
            !gemm(m_h, dim, l.k, dim, dim, n, l.k_b, m_k, false, error) ||
            !gemm(m_h, dim, l.v, dim, dim, n, l.v_b, m_v, false, error) ||
            !attention(m_q, m_k, m_v, m_attn, n, n, TripoModel::VIT_HEADS, error) ||
            !gemm(m_attn, dim, l.out, dim, dim, n, l.out_b, m_image, true, error) ||
            !layer_norm(m_image, l.ln2_w, l.ln2_b, m_h, n, dim, TripoModel::VIT_EPS, error) ||
            !gemm(m_h, dim, l.up, dim, TripoModel::VIT_FF, n, l.up_b, m_up, false, error) ||
            !elementwise(m_gelu, gelu_args, ff_count, error) ||
            !gemm(m_up, TripoModel::VIT_FF, l.down, TripoModel::VIT_FF, dim, n, l.down_b, m_image, true, error) ||
            !trace("vit_layer", static_cast<int>(il), m_image, count, error))
        {
            return false;
        }
    }
    return layer_norm(m_image, m_vit_norm_w, m_vit_norm_b, m_image, n, dim, TripoModel::VIT_EPS, error) &&
           trace("vit_tokens", -1, m_image, count, error);
}

bool TripoCudaRunner::build_triplane(Error& error)
{
    if (m_device == nullptr)
    {
        error.set("TripoCudaRunner used before init");
        return false;
    }
    const uint32_t n = TripoModel::PLANE_TOKENS;
    const uint32_t dim = TripoModel::DIM;
    const uint32_t ff = TripoModel::FF;
    const uint32_t n_image =
        (TripoModel::IMAGE_SIZE / TripoModel::VIT_PATCH) * (TripoModel::IMAGE_SIZE / TripoModel::VIT_PATCH) + 1;
    uint32_t count = n * dim;
    if (!gemm(m_plane_norm, dim, m_proj_in, dim, dim, n, m_proj_in_b, m_x, false, error) ||
        !trace("bb_proj_in", -1, m_x, count, error))
    {
        return false;
    }
    for (uint32_t ib = 0; ib < TripoModel::BLOCKS; ++ib)
    {
        const Block& b = m_blocks[ib];
        uint32_t geglu_count = n * ff;
        uint32_t ff_arg = ff;
        void* geglu_args[] = {&m_up, &ff_arg, &geglu_count};
        if (!layer_norm(m_x, b.ln1_w, b.ln1_b, m_h, n, dim, TripoModel::LN_EPS, error) ||
            !gemm(m_h, dim, b.self_q, dim, dim, n, 0, m_q, false, error) ||
            !gemm(m_h, dim, b.self_k, dim, dim, n, 0, m_k, false, error) ||
            !gemm(m_h, dim, b.self_v, dim, dim, n, 0, m_v, false, error) ||
            !attention(m_q, m_k, m_v, m_attn, n, n, TripoModel::HEADS, error) ||
            !gemm(m_attn, dim, b.self_out, dim, dim, n, b.self_out_b, m_x, true, error) ||
            !layer_norm(m_x, b.ln2_w, b.ln2_b, m_h, n, dim, TripoModel::LN_EPS, error) ||
            !gemm(m_h, dim, b.cross_q, dim, dim, n, 0, m_q, false, error) ||
            !gemm(m_image, TripoModel::VIT_DIM, b.cross_k, TripoModel::VIT_DIM, dim, n_image, 0, m_k, false, error) ||
            !gemm(m_image, TripoModel::VIT_DIM, b.cross_v, TripoModel::VIT_DIM, dim, n_image, 0, m_v, false, error) ||
            !attention(m_q, m_k, m_v, m_attn, n, n_image, TripoModel::HEADS, error) ||
            !gemm(m_attn, dim, b.cross_out, dim, dim, n, b.cross_out_b, m_x, true, error) ||
            !layer_norm(m_x, b.ln3_w, b.ln3_b, m_h, n, dim, TripoModel::LN_EPS, error) ||
            !gemm(m_h, dim, b.ff_in, dim, 2 * ff, n, b.ff_in_b, m_up, false, error) ||
            !elementwise(m_geglu, geglu_args, geglu_count, error) ||
            !gemm(m_up, 2 * ff, b.ff_out, ff, dim, n, b.ff_out_b, m_x, true, error) ||
            !trace("bb_block", static_cast<int>(ib), m_x, count, error))
        {
            return false;
        }
    }
    void* add_args[] = {&m_h, &m_plane_tokens, &count};
    if (!gemm(m_x, dim, m_proj_out, dim, dim, n, m_proj_out_b, m_h, false, error) ||
        !elementwise(m_add, add_args, count, error) || !trace("bb_out", -1, m_h, count, error))
    {
        return false;
    }
    const uint32_t rows = TripoModel::CODE_CHANNELS * 4;
    uint32_t up_count = n * rows;
    void* up_args[] = {&m_attn, &m_upsample_b, &m_code, &m_planes, &up_count};
    const uint32_t code = 3 * TripoModel::CODE_CHANNELS * TripoModel::CODE_SIZE * TripoModel::CODE_SIZE;
    return gemm(m_h, dim, m_upsample_w, dim, rows, n, 0, m_attn, false, error) &&
           elementwise(m_upsample, up_args, up_count, error) && trace("scene_codes", -1, m_code, code, error);
}

bool TripoCudaRunner::query(uint64_t points, uint64_t lin, uint32_t grid_size, uint32_t n, uint64_t density,
                            uint64_t color, Error& error)
{
    float low = -TripoModel::RADIUS;
    float span = TripoModel::RADIUS - low;
    float density_bias = TripoModel::DENSITY_BIAS;
    void* arguments[] = {&m_planes, &points, &lin,  &grid_size,    &n,       &m_mlp_w,
                         &m_mlp_b,  &low,    &span, &density_bias, &density, &color};
    CudaLaunch launch;
    launch.grid_x = (n + 127) / 128;
    launch.block_x = 128;
    return m_device->launch(m_query, m_stream, launch, arguments, error);
}

bool TripoCudaRunner::extract_mesh(uint32_t resolution, float threshold, TripoMesh& mesh, Error& error)
{
    if (m_device == nullptr || resolution < 2)
    {
        error.set("TripoCudaRunner::extract_mesh: no model or resolution below 2");
        return false;
    }
    const size_t cells = size_t(resolution) * resolution * resolution;
    const size_t need = aligned(cells) + aligned(resolution);
    if (m_grid.bytes < need * sizeof(float))
    {
        if (m_grid.address != 0 && !m_device->destroy_buffer(m_grid, error))
        {
            return false;
        }
        if (!m_device->create_buffer(need * sizeof(float), m_grid, error))
        {
            return false;
        }
    }
    const uint64_t density = m_grid.address;
    const uint64_t lin = m_grid.address + aligned(cells) * sizeof(float);
    Array<float> axis;
    Array<float> vol;
    if (!axis.resize_uninitialized(resolution) || !vol.resize_uninitialized(cells))
    {
        error.set("out of memory for a %u^3 density grid", resolution);
        return false;
    }
    tripo_grid_axis(resolution, axis.data());
    if (!m_device->upload(m_grid, axis.data(), resolution * sizeof(float), error, lin - m_grid.address) ||
        !query(0, lin, resolution, static_cast<uint32_t>(cells), density, 0, error) ||
        !m_device->synchronize(m_stream, error) || !m_device->download(m_grid, vol.data(), cells * sizeof(float), error))
    {
        return false;
    }
    if (m_trace != nullptr)
    {
        m_trace(m_trace_user, "density_act", -1, vol.data(), cells);
    }
    for (size_t i = 0; i < cells; ++i)
    {
        vol[i] = vol[i] - threshold;
    }
    Array<float> grid_points;
    if (!tripo_marching_cubes(*m_pool, vol.data(), resolution, grid_points, mesh.faces, error))
    {
        return false;
    }
    const size_t n_vertices = grid_points.size() / 3;
    if (!mesh.positions.resize_uninitialized(3 * n_vertices) || !mesh.colors.resize_uninitialized(3 * n_vertices))
    {
        error.set("out of memory for %zu vertices", n_vertices);
        return false;
    }
    tripo_mesh_positions(grid_points.data(), n_vertices, resolution, mesh.positions.data());
    if (n_vertices == 0)
    {
        return true;
    }
    // points, their density and colors where the grid was
    const uint64_t points = m_grid.address;
    const uint64_t colors = points + aligned(3 * n_vertices) * sizeof(float);
    const uint64_t vertex_density = colors + aligned(3 * n_vertices) * sizeof(float);
    if (aligned(3 * n_vertices) * 2 + aligned(n_vertices) > need)
    {
        error.set("too many vertices for the grid buffer");
        return false;
    }
    return m_device->upload(m_grid, mesh.positions.data(), 3 * n_vertices * sizeof(float), error) &&
           query(points, 0, 0, static_cast<uint32_t>(n_vertices), vertex_density, colors, error) &&
           m_device->synchronize(m_stream, error) &&
           m_device->download(m_grid, mesh.colors.data(), 3 * n_vertices * sizeof(float), error,
                              colors - m_grid.address);
}

}
