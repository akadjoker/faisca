#pragma once

#include "faisca/cuda.h"
#include "faisca/triposr.h"

namespace faisca
{

// TripoRunner on an NVIDIA GPU in float32: the packed weights uploaded once, one stream,
// the marching cubes on the CPU threads of pool
class TripoCudaRunner
{
public:
    TripoCudaRunner() = default;
    ~TripoCudaRunner();

    TripoCudaRunner(const TripoCudaRunner&) = delete;
    TripoCudaRunner& operator=(const TripoCudaRunner&) = delete;

    bool init(const TripoModel& model, CudaDevice& device, ThreadPool& pool, Error& error);
    void release();

    // traces download every stage (slow); for the comparison with the reference only
    void set_trace(RunnerTrace callback, void* user)
    {
        m_trace = callback;
        m_trace_user = user;
    }

    bool encode_image(const float* rgb, Error& error);
    bool build_triplane(Error& error);
    bool extract_mesh(uint32_t resolution, float threshold, TripoMesh& mesh, Error& error);

    // waits for the queued work, for timing
    bool synchronize(Error& error)
    {
        return m_device->synchronize(m_stream, error);
    }

private:
    struct Vit
    {
        uint64_t ln1_w, ln1_b, q, q_b, k, k_b, v, v_b, out, out_b, ln2_w, ln2_b, up, up_b, down, down_b;
    };
    struct Block
    {
        uint64_t ln1_w, ln1_b, self_q, self_k, self_v, self_out, self_out_b, ln2_w, ln2_b, cross_q, cross_k, cross_v,
            cross_out, cross_out_b, ln3_w, ln3_b, ff_in, ff_in_b, ff_out, ff_out_b;
    };

    bool trace(const char* name, int layer, uint64_t address, size_t count, Error& error);
    uint64_t panels(const TripoLinear& w) const;
    bool gemm(uint64_t x, uint32_t x_stride, uint64_t w, uint32_t n_in, uint32_t n_out, uint32_t n_tokens,
              uint64_t bias, uint64_t out, bool accumulate, Error& error);
    bool layer_norm(uint64_t x, uint64_t w, uint64_t b, uint64_t out, uint32_t rows, uint32_t width, float eps,
                    Error& error);
    bool attention(uint64_t q, uint32_t q_stride, uint64_t k, uint64_t v, uint32_t kv_stride, uint64_t out,
                   uint32_t n_q, uint32_t n_kv, uint32_t heads, Error& error);
    bool elementwise(const CudaKernel& kernel, void** arguments, uint32_t count, Error& error);
    bool query(uint64_t points, uint64_t lin, uint32_t grid_size, uint32_t n, uint64_t density, uint64_t color,
               Error& error);

    const TripoModel* m_model = nullptr;
    CudaDevice* m_device = nullptr;
    ThreadPool* m_pool = nullptr;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;
    CudaStream m_stream;
    CudaModule m_module;
    CudaKernel m_gemm, m_layer_norm, m_gelu, m_geglu, m_add, m_flash, m_upsample, m_query;

    CudaBuffer m_weights; // TripoModel::panel_data()
    CudaBuffer m_params;  // biases, norms, positions, plane inputs, MLP biases
    CudaBuffer m_work;    // activations
    CudaBuffer m_grid;    // density grid and mesh points, grown on demand
    Vit m_vit[TripoModel::VIT_LAYERS] = {};
    Block m_blocks[TripoModel::BLOCKS] = {};
    uint64_t m_vit_patch = 0, m_vit_patch_b = 0, m_vit_cls = 0, m_vit_pos = 0, m_vit_norm_w = 0, m_vit_norm_b = 0;
    uint64_t m_plane_tokens = 0, m_plane_norm = 0, m_proj_in = 0, m_proj_in_b = 0, m_proj_out = 0, m_proj_out_b = 0;
    uint64_t m_upsample_w = 0, m_upsample_b = 0, m_mlp_w = 0, m_mlp_b = 0;
    uint64_t m_patches = 0, m_image = 0, m_x = 0, m_h = 0, m_q = 0, m_k = 0, m_v = 0, m_attn = 0, m_up = 0;
    uint64_t m_code = 0, m_planes = 0;
    Array<float> m_host;
};

}
