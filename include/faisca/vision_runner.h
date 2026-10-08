#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/image.h"
#include "faisca/runner.h"
#include "faisca/thread.h"
#include "faisca/vision_model.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// the image of mtmd's fixed size preprocessor: scaled to fit image_size x image_size (ceil,
// Pillow bilinear), centred on black, normalized with the mean and std, as planes per
// channel (R, G, B) of image_size x image_size floats
bool preprocess_image(const Image& image, const VisionHparams& h, Array<float>& planes, Error& error);

// an image through the SigLIP encoder and the gemma3 projector: n_tokens() rows of n_proj
// values, the embeddings the language model takes in place of the image tokens
class VisionRunner
{
public:
    VisionRunner() = default;

    VisionRunner(const VisionRunner&) = delete;
    VisionRunner& operator=(const VisionRunner&) = delete;

    bool init(const VisionModel& model, ThreadPool& pool, Error& error);

    void set_trace(RunnerTrace callback, void* user)
    {
        m_trace = callback;
        m_trace_user = user;
    }

    bool encode(const Image& image, Array<float>& embeddings, Error& error);

    // planes as preprocess_image gives them
    bool encode_planes(const float* planes, Array<float>& embeddings, Error& error);

private:
    struct RowsJob;
    struct MatmulJob;
    struct AttentionJob;
    struct TileJob;
    struct PatchJob;

    static void rows_task(void* user, int thread_index, int thread_count);
    static void matmul_task(void* user, int thread_index, int thread_count);
    static void quantize_task(void* user, int thread_index, int thread_count);
    static void attention_task(void* user, int thread_index, int thread_count);
    static void tile_task(void* user, int thread_index, int thread_count);
    static void patch_task(void* user, int thread_index, int thread_count);

    void trace(const char* name, int layer, const float* data, size_t count) const;
    void matmul(const Weight& w, const float* x, size_t n_tokens, float* out);
    void quantize_inputs(const float* x, size_t n_tokens, size_t n);
    void norm_rows(const float* x, const float* w, const float* b, size_t rows, size_t width, float* out);
    void gelu_rows(float* x, size_t rows, size_t width);
    void bias_rows(float* x, const float* b, size_t rows, size_t width);
    void add_rows(float* x, const float* y, size_t rows, size_t width);
    void tile_keys(const float* k, size_t n_kv, float* tiles);
    size_t tiles_per_head() const;

    const VisionModel* m_model = nullptr;
    ThreadPool* m_pool = nullptr;
    RunnerTrace m_trace = nullptr;
    void* m_trace_user = nullptr;
    const float* m_quantize_x = nullptr;
    size_t m_quantize_tokens = 0;
    size_t m_quantize_n = 0;

    Array<float> m_planes;
    Array<float> m_patches;
    Array<float> m_x;
    Array<float> m_h;
    Array<float> m_q;
    Array<float> m_k;
    Array<float> m_v;
    Array<float> m_attn;
    Array<float> m_up;
    Array<float> m_k_tiles;
    Array<float> m_scores;
    Array<float> m_pooled;
    Array<uint8_t> m_xq;
    Array<int32_t> m_sums;
    size_t m_xq_stride = 0;
    size_t m_sums_stride = 0;
};

}
