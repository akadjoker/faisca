#include "faisca/omnidreams_cuda.h"

#include <cmath>
#include <cstring>

namespace faisca
{

namespace
{

constexpr uint32_t D = DIT_WIDTH;
constexpr uint32_t EMBED_IN = 160; // patches, HD map patches, zeros up to a multiple of 32
constexpr uint64_t V_SIN = 0;      // offsets in the vector buffer, floats
constexpr uint64_t V_TEMB = D;
constexpr uint64_t V_H1 = 2 * D;
constexpr uint64_t V_LORA = 3 * D;
constexpr uint64_t V_TMP = 6 * D;
constexpr uint64_t V_COUNT = 6 * D + DIT_LORA;
// streamed block weights in a slot: qkv, out, cross q, cross out, MLP up, MLP down
constexpr uint64_t SQUARE = uint64_t(D) * D * 2;
constexpr uint64_t SLOT_OFFSETS[6] = {0, 3 * SQUARE, 4 * SQUARE, 5 * SQUARE, 6 * SQUARE, 10 * SQUARE};
constexpr uint64_t SLOT_SIZES[6] = {3 * SQUARE, SQUARE, SQUARE, SQUARE, 4 * SQUARE, 4 * SQUARE};
constexpr uint64_t SLOT_BYTES = 14 * SQUARE;

CudaBuffer range(const CudaBuffer& buffer, uint64_t offset, uint64_t bytes)
{
    CudaBuffer out = buffer;
    out.address += offset;
    out.bytes = bytes;
    return out;
}

CudaBuffer view(const CudaBuffer& buffer, uint64_t offset)
{
    CudaBuffer out = buffer;
    out.address += offset;
    out.bytes -= offset;
    return out;
}

float bf16_to_float(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// rope.py _compute_freqs in float32 as PyTorch does it: arange(0, dim, 2) / dim, theta a float
// scalar, 1 / theta ** range
bool compute_freqs(uint32_t dim, float ratio, Array<float>& out)
{
    if (!out.resize_uninitialized(dim / 2))
    {
        return false;
    }
    const double theta = 10000.0 * std::pow(static_cast<double>(ratio), static_cast<double>(dim) / (dim - 2.0));
    for (uint32_t i = 0; i < dim / 2; ++i)
    {
        const float range = static_cast<float>(2 * i) / static_cast<float>(dim);
        out.data()[i] = 1.0f / std::pow(static_cast<float>(theta), range);
    }
    return true;
}

}

void DitKvCache::init(uint32_t chunk, uint32_t window, uint32_t sink)
{
    m_chunk = chunk;
    m_window = window;
    m_sink = sink;
    reset();
}

void DitKvCache::reset()
{
    m_previous = -1;
    m_current = -1;
    m_updating = false;
    m_cached = 0;
}

bool DitKvCache::overlaps_sink() const
{
    return m_sink > 0 && m_current * static_cast<int64_t>(m_chunk) < static_cast<int64_t>(m_sink);
}

bool DitKvCache::before_update(int64_t chunk_index, uint32_t& shift, uint32_t& keep, Error& error)
{
    shift = 0;
    keep = 0;
    if (m_updating)
    {
        error.set("DiT cache: before_update twice");
        return false;
    }
    m_current = chunk_index;
    m_updating = true;
    if (chunk_index == m_previous)
    {
        return true;
    }
    if (chunk_index != m_previous + 1)
    {
        error.set("DiT cache: chunk %lld after %lld", static_cast<long long>(chunk_index),
                  static_cast<long long>(m_previous));
        m_updating = false;
        return false;
    }
    if (!overlaps_sink())
    {
        const int64_t overflow = static_cast<int64_t>(m_cached) + m_chunk - total();
        if (overflow > 0)
        {
            // _roll_local_window_left
            shift = static_cast<uint32_t>(overflow);
            const uint32_t valid_end = m_cached < total() ? m_cached : total();
            const uint32_t valid_local = valid_end > m_sink ? valid_end - m_sink : 0;
            keep = valid_local > shift ? valid_local - shift : 0;
            m_cached = total();
        }
    }
    return true;
}

void DitKvCache::write_bounds(uint32_t& write_start, uint32_t& write_end, uint32_t& read_start,
                              uint32_t& read_end) const
{
    if (m_current == m_previous + 1)
    {
        write_start = m_cached < total() - m_chunk ? m_cached : total() - m_chunk;
        write_end = write_start + m_chunk;
    }
    else
    {
        write_end = m_cached < total() ? m_cached : total();
        write_start = write_end > m_chunk ? write_end - m_chunk : 0;
    }
    read_start = 0;
    read_end = write_end - write_start;
    if (m_sink > 0 && !overlaps_sink() && write_start < m_sink)
    {
        write_start = m_sink;
        const uint32_t kept = write_end - write_start;
        read_end = m_chunk;
        read_start = read_end - kept;
    }
}

uint32_t DitKvCache::visible_end() const
{
    if (m_current == m_previous + 1)
    {
        return m_cached + m_chunk < total() ? m_cached + m_chunk : total();
    }
    return m_cached < total() ? m_cached : total();
}

void DitKvCache::after_update()
{
    if (m_current == m_previous + 1)
    {
        const bool steady = total() == m_cached && !overlaps_sink();
        if (!steady)
        {
            m_cached = m_cached + m_chunk < total() ? m_cached + m_chunk : total();
        }
        ++m_previous;
    }
    m_updating = false;
}

DitCudaRunner::~DitCudaRunner()
{
    release();
}

bool DitCudaRunner::upload(CudaBuffer& buffer, const void* data, size_t bytes, Error& error)
{
    return m_device->create_buffer(bytes, buffer, error) && m_device->upload(buffer, data, bytes, error);
}

void DitCudaRunner::release()
{
    if (m_device == nullptr)
    {
        return;
    }
    Error ignored;
    for (uint32_t b = 0; b < m_block_count; ++b)
    {
        Block& block = m_blocks[b];
        CudaBuffer* streamed[] = {&block.qkv, &block.out, &block.cross_q, &block.cross_out, &block.mlp_up,
                                  &block.mlp_down};
        CudaBuffer* buffers[] = {&block.q_norm, &block.k_norm, &block.cross_kv, &block.cross_q_norm,
                                 &block.cross_k_norm, &block.mod_down[0], &block.mod_down[1], &block.mod_down[2],
                                 &block.mod_up[0], &block.mod_up[1], &block.mod_up[2], &block.text_kv, &block.cache};
        for (CudaBuffer* buffer : streamed)
        {
            if (!m_streaming && buffer->bytes != 0)
            {
                m_device->destroy_buffer(*buffer, ignored);
            }
            *buffer = CudaBuffer();
        }
        for (CudaBuffer* buffer : buffers)
        {
            if (buffer->bytes != 0)
            {
                m_device->destroy_buffer(*buffer, ignored);
            }
            *buffer = CudaBuffer();
        }
    }
    CudaBuffer* buffers[] = {&m_embedder, &m_t_linear1, &m_t_linear2, &m_final_mod_down, &m_final_mod_up,
                             &m_final_linear, &m_text_proj, &m_text_proj_bias, &m_input, &m_x, &m_h, &m_q,
                             &m_attn, &m_hidden, &m_angles, &m_vectors, &m_modulation, &m_text, &m_text_in,
                             &m_output};
    for (CudaBuffer* buffer : buffers)
    {
        if (buffer->bytes != 0)
        {
            m_device->destroy_buffer(*buffer, ignored);
        }
        *buffer = CudaBuffer();
    }
    for (int i = 0; i < 2; ++i)
    {
        if (m_slots[i].bytes != 0)
        {
            m_device->destroy_buffer(m_slots[i], ignored);
        }
        m_slots[i] = CudaBuffer();
        if (m_streaming)
        {
            m_device->destroy_event(m_loaded[i], ignored);
            m_device->destroy_event(m_free[i], ignored);
        }
    }
    if (m_host_blocks.bytes != 0)
    {
        m_device->destroy_host_buffer(m_host_blocks, ignored);
    }
    if (m_streaming)
    {
        m_device->destroy_stream(m_copy_stream, ignored);
    }
    m_streaming = false;
    m_device->destroy_stream(m_stream, ignored);
    m_kernels.release();
    m_device = nullptr;
    m_block_count = 0;
}

bool DitCudaRunner::init(CudaDevice& device, const DitHostWeights& weights, const DitConfig& config, Error& error)
{
    release();
    if (weights.block_count == 0 || weights.block_count > DIT_MAX_BLOCKS || config.frames == 0 ||
        config.height == 0 || config.width == 0 || config.window_chunks == 0)
    {
        error.set("invalid DiT configuration");
        return false;
    }
    m_device = &device;
    m_config = config;
    m_chunk = config.frames * config.height * config.width;
    m_hdmap = weights.hdmap_embedder != nullptr;
    const uint32_t L = m_chunk;
    if (!m_kernels.init(device, error) || !device.create_stream(m_stream, error))
    {
        release();
        return false;
    }

    // x_embedder and the HD map embedder side by side, one GEMM over [patches | HD map | 0]
    Array<uint16_t> embedder;
    if (!embedder.resize_uninitialized(static_cast<size_t>(D) * EMBED_IN))
    {
        error.set("out of memory");
        release();
        return false;
    }
    for (uint32_t r = 0; r < D; ++r)
    {
        uint16_t* row = embedder.data() + static_cast<size_t>(r) * EMBED_IN;
        std::memset(row, 0, EMBED_IN * 2);
        std::memcpy(row, weights.x_embedder + static_cast<size_t>(r) * DIT_PATCH_IN, DIT_PATCH_IN * 2);
        if (m_hdmap)
        {
            std::memcpy(row + DIT_PATCH_IN, weights.hdmap_embedder + static_cast<size_t>(r) * DIT_HDMAP_IN,
                        DIT_HDMAP_IN * 2);
        }
    }
    if (!m_t_norm.resize_uninitialized(D))
    {
        error.set("out of memory");
        release();
        return false;
    }
    for (uint32_t i = 0; i < D; ++i)
    {
        m_t_norm.data()[i] = bf16_to_float(weights.t_norm[i]);
    }
    const uint32_t dim_hw = 128 / 6 * 2;
    const uint32_t dim_t = 128 - 2 * dim_hw;
    bool ok = compute_freqs(dim_t, config.t_extrapolation, m_raw_t) &&
              compute_freqs(dim_hw, config.h_extrapolation, m_raw_h) &&
              compute_freqs(dim_hw, config.w_extrapolation, m_raw_w) &&
              m_staging.resize_uninitialized(static_cast<size_t>(L) * EMBED_IN) &&
              m_host_vectors.resize_uninitialized(static_cast<size_t>(L) * 64 > 2 * D ? static_cast<size_t>(L) * 64
                                                                                     : 2 * D);
    if (!ok)
    {
        error.set("out of memory");
        release();
        return false;
    }

    ok = upload(m_embedder, embedder.data(), embedder.size() * 2, error) &&
         upload(m_t_linear1, weights.t_linear1, size_t(D) * D * 2, error) &&
         upload(m_t_linear2, weights.t_linear2, size_t(3 * D) * D * 2, error) &&
         upload(m_final_mod_down, weights.final_mod_down, size_t(DIT_LORA) * D * 2, error) &&
         upload(m_final_mod_up, weights.final_mod_up, size_t(2 * D) * DIT_LORA * 2, error) &&
         upload(m_final_linear, weights.final_linear, size_t(DIT_PATCH_OUT) * D * 2, error) &&
         upload(m_text_proj, weights.text_proj, size_t(DIT_TEXT_WIDTH) * DIT_TEXT_IN * 2, error);
    if (ok)
    {
        Array<float> bias;
        ok = bias.resize_uninitialized(DIT_TEXT_WIDTH);
        for (uint32_t i = 0; ok && i < DIT_TEXT_WIDTH; ++i)
        {
            bias.data()[i] = bf16_to_float(weights.text_proj_bias[i]);
        }
        ok = ok && upload(m_text_proj_bias, bias.data(), DIT_TEXT_WIDTH * 4, error);
    }
    m_streaming = config.stream_weights;
    if (ok && m_streaming)
    {
        ok = device.create_host_buffer(SLOT_BYTES * weights.block_count, m_host_blocks, error) &&
             device.create_buffer(SLOT_BYTES, m_slots[0], error) && device.create_buffer(SLOT_BYTES, m_slots[1], error) &&
             device.create_stream(m_copy_stream, error) && device.create_event(m_loaded[0], error) &&
             device.create_event(m_loaded[1], error) && device.create_event(m_free[0], error) &&
             device.create_event(m_free[1], error);
    }
    for (uint32_t b = 0; ok && b < weights.block_count; ++b)
    {
        const DitHostBlock& w = weights.blocks[b];
        Block& block = m_blocks[b];
        m_block_count = b + 1;
        const size_t square = size_t(D) * D * 2;
        const size_t context = size_t(D) * DIT_TEXT_WIDTH * 2;
        if (m_streaming)
        {
            uint8_t* host = static_cast<uint8_t*>(m_host_blocks.data) + b * SLOT_BYTES;
            std::memcpy(host, w.self_q, square);
            std::memcpy(host + square, w.self_k, square);
            std::memcpy(host + 2 * square, w.self_v, square);
            std::memcpy(host + SLOT_OFFSETS[1], w.self_out, square);
            std::memcpy(host + SLOT_OFFSETS[2], w.cross_q, square);
            std::memcpy(host + SLOT_OFFSETS[3], w.cross_out, square);
            std::memcpy(host + SLOT_OFFSETS[4], w.mlp_up, 4 * square);
            std::memcpy(host + SLOT_OFFSETS[5], w.mlp_down, 4 * square);
        }
        else
        {
            ok = device.create_buffer(3 * square, block.qkv, error) &&
                 device.upload(block.qkv, w.self_q, square, error) &&
                 device.upload(block.qkv, w.self_k, square, error, square) &&
                 device.upload(block.qkv, w.self_v, square, error, 2 * square) &&
                 upload(block.out, w.self_out, square, error) && upload(block.cross_q, w.cross_q, square, error) &&
                 upload(block.cross_out, w.cross_out, square, error) &&
                 upload(block.mlp_up, w.mlp_up, size_t(DIT_MLP) * D * 2, error) &&
                 upload(block.mlp_down, w.mlp_down, size_t(DIT_MLP) * D * 2, error);
        }
        ok = ok && upload(block.q_norm, w.self_q_norm, 256, error) && upload(block.k_norm, w.self_k_norm, 256, error) &&
             device.create_buffer(2 * context, block.cross_kv, error) &&
             device.upload(block.cross_kv, w.cross_k, context, error) &&
             device.upload(block.cross_kv, w.cross_v, context, error, context) &&
             upload(block.cross_q_norm, w.cross_q_norm, 256, error) &&
             upload(block.cross_k_norm, w.cross_k_norm, 256, error);
        for (int i = 0; ok && i < 3; ++i)
        {
            ok = upload(block.mod_down[i], w.mod_down[i], size_t(DIT_LORA) * D * 2, error) &&
                 upload(block.mod_up[i], w.mod_up[i], size_t(3 * D) * DIT_LORA * 2, error);
        }
        const uint32_t window = b < config.short_history_blocks ? L : L * config.window_chunks;
        block.kv.init(L, window, config.sink);
        ok = ok && device.create_buffer(size_t(DIT_TEXT_TOKENS) * 2 * D * 2, block.text_kv, error) &&
             device.create_buffer(size_t(block.kv.total()) * 2 * D * 2, block.cache, error);
    }
    ok = ok && device.create_buffer(size_t(L) * EMBED_IN * 2, m_input, error) &&
         device.create_buffer(size_t(L) * D * 4, m_x, error) && device.create_buffer(size_t(L) * D * 2, m_h, error) &&
         device.create_buffer(size_t(L) * D * 2, m_q, error) && device.create_buffer(size_t(L) * D * 2, m_attn, error) &&
         device.create_buffer(size_t(L) * DIT_MLP * 2, m_hidden, error) &&
         device.create_buffer(size_t(L) * 64 * 4, m_angles, error) &&
         device.create_buffer(V_COUNT * 4, m_vectors, error) &&
         device.create_buffer((size_t(m_block_count) * 9 + 2) * D * 4, m_modulation, error) &&
         device.create_buffer(size_t(DIT_TEXT_TOKENS) * DIT_TEXT_WIDTH * 2, m_text, error) &&
         device.create_buffer(size_t(L) * DIT_PATCH_OUT * 4, m_output, error);
    if (!ok)
    {
        release();
        return false;
    }
    return true;
}

void DitCudaRunner::reset_cache()
{
    for (uint32_t b = 0; b < m_block_count; ++b)
    {
        m_blocks[b].kv.reset();
    }
}

bool DitCudaRunner::set_text(const uint16_t* text, Error& error)
{
    using Out = CudaKernels::GemmOutput;
    if (m_device == nullptr)
    {
        error.set("DiT runner used before init");
        return false;
    }
    bool ok = upload(m_text_in, text, size_t(DIT_TEXT_TOKENS) * DIT_TEXT_IN * 2, error) &&
              m_kernels.gemm_f16(m_text_in, m_text_proj, m_text, DIT_TEXT_TOKENS, DIT_TEXT_WIDTH, DIT_TEXT_IN, true,
                                 m_stream, error, Out::HalfGelu, &m_text_proj_bias);
    for (uint32_t b = 0; ok && b < m_block_count; ++b)
    {
        Block& block = m_blocks[b];
        ok = m_kernels.gemm_f16(m_text, block.cross_kv, block.text_kv, DIT_TEXT_TOKENS, 2 * D, DIT_TEXT_WIDTH, true,
                                m_stream, error, Out::Half) &&
             m_kernels.dit_qk_norm_rope(block.text_kv, 2 * D, 0, block.cross_k_norm, nullptr, DIT_TEXT_TOKENS,
                                        DIT_HEADS, true, m_stream, error);
    }
    ok = ok && m_device->synchronize(m_stream, error);
    Error ignored;
    m_device->destroy_buffer(m_text_in, ignored);
    m_text_in = CudaBuffer();
    return ok;
}

bool DitCudaRunner::run_block(Block& block, uint32_t index, int64_t chunk_index, Error& error)
{
    using Out = CudaKernels::GemmOutput;
    const uint32_t L = m_chunk;
    const uint64_t row = 2ull * D * 2;
    uint32_t shift = 0;
    uint32_t keep = 0;
    if (!block.kv.before_update(chunk_index, shift, keep, error))
    {
        return false;
    }
    // the window moves left by shift rows; copied forward in pieces of shift rows, each piece
    // lands on rows already read
    const uint32_t sink = m_config.sink;
    for (uint32_t done = 0; done < keep; done += shift)
    {
        const uint32_t count = keep - done < shift ? keep - done : shift;
        if (!m_kernels.dit_copy(block.cache, (sink + shift + done) * row, block.cache, (sink + done) * row,
                                count * row, m_stream, error))
        {
            return false;
        }
    }
    uint32_t write_start = 0, write_end = 0, read_start = 0, read_end = 0;
    block.kv.write_bounds(write_start, write_end, read_start, read_end);
    if (read_start != 0 || read_end != L)
    {
        error.set("DiT cache: a chunk written partly over the sink is not supported yet");
        return false;
    }
    const uint64_t base = static_cast<uint64_t>(index) * 9 * D * 4;
    const uint64_t f = static_cast<uint64_t>(D) * 4;
    const CudaBuffer chunk_kv = view(block.cache, write_start * row);
    const float scale = 1.0f / std::sqrt(128.0f);
    const bool ok =
        m_kernels.dit_modulate_norm(m_x, m_modulation, base, base + f, m_h, L, D, true, m_stream, error) &&
        m_kernels.gemm_f16(m_h, block.qkv, m_q, L, D, D, true, m_stream, error, Out::Half) &&
        m_kernels.gemm_f16(m_h, view(block.qkv, uint64_t(D) * D * 2), chunk_kv, L, 2 * D, D, true, m_stream, error,
                           Out::Half) &&
        m_kernels.dit_qk_norm_rope(m_q, D, 0, block.q_norm, &m_angles, L, DIT_HEADS, true, m_stream, error) &&
        m_kernels.dit_qk_norm_rope(chunk_kv, 2 * D, 0, block.k_norm, &m_angles, L, DIT_HEADS, true, m_stream,
                                   error) &&
        m_kernels.attention_f16(m_q, D, block.cache, view(block.cache, uint64_t(D) * 2), 2 * D, m_attn, D, L,
                                block.kv.visible_end(), DIT_HEADS, scale, true, m_stream, error) &&
        m_kernels.gemm_f16(m_attn, block.out, m_x, L, D, D, true, m_stream, error, Out::Residual, &m_modulation,
                           base + 2 * f) &&
        m_kernels.dit_modulate_norm(m_x, m_modulation, base + 3 * f, base + 4 * f, m_h, L, D, true, m_stream,
                                    error) &&
        m_kernels.gemm_f16(m_h, block.cross_q, m_q, L, D, D, true, m_stream, error, Out::Half) &&
        m_kernels.dit_qk_norm_rope(m_q, D, 0, block.cross_q_norm, nullptr, L, DIT_HEADS, true, m_stream, error) &&
        m_kernels.attention_f16(m_q, D, block.text_kv, view(block.text_kv, uint64_t(D) * 2), 2 * D, m_attn, D, L,
                                DIT_TEXT_TOKENS, DIT_HEADS, scale, true, m_stream, error) &&
        m_kernels.gemm_f16(m_attn, block.cross_out, m_x, L, D, D, true, m_stream, error, Out::Residual,
                           &m_modulation, base + 5 * f) &&
        m_kernels.dit_modulate_norm(m_x, m_modulation, base + 6 * f, base + 7 * f, m_h, L, D, true, m_stream,
                                    error) &&
        m_kernels.gemm_f16(m_h, block.mlp_up, m_hidden, L, DIT_MLP, D, true, m_stream, error, Out::HalfGelu) &&
        m_kernels.gemm_f16(m_hidden, block.mlp_down, m_x, L, D, DIT_MLP, true, m_stream, error, Out::Residual,
                           &m_modulation, base + 8 * f);
    block.kv.after_update();
    return ok;
}

void DitCudaRunner::point_at_slot(Block& block, uint32_t slot)
{
    CudaBuffer* streamed[] = {&block.qkv, &block.out, &block.cross_q, &block.cross_out, &block.mlp_up, &block.mlp_down};
    for (int i = 0; i < 6; ++i)
    {
        *streamed[i] = range(m_slots[slot], SLOT_OFFSETS[i], SLOT_SIZES[i]);
    }
}

bool DitCudaRunner::copy_block(uint32_t index, Error& error)
{
    const uint32_t slot = index % 2;
    return (index < 2 || m_device->wait_event(m_copy_stream, m_free[slot], error)) &&
           m_device->upload_async(m_slots[slot], m_host_blocks, SLOT_BYTES, m_copy_stream, error, 0,
                                  index * SLOT_BYTES) &&
           m_device->record_event(m_loaded[slot], m_copy_stream, error);
}

bool DitCudaRunner::run_blocks(int64_t chunk_index, Error& error)
{
    if (!m_streaming)
    {
        for (uint32_t b = 0; b < m_block_count; ++b)
        {
            if (!run_block(m_blocks[b], b, chunk_index, error))
            {
                return false;
            }
        }
        return true;
    }
    // block b computes from slot b % 2 once its copy is in; the copy of block b + 2 waits until
    // block b has finished with that slot
    if (!copy_block(0, error) || (m_block_count > 1 && !copy_block(1, error)))
    {
        return false;
    }
    for (uint32_t b = 0; b < m_block_count; ++b)
    {
        const uint32_t slot = b % 2;
        point_at_slot(m_blocks[b], slot);
        if (!m_device->wait_event(m_stream, m_loaded[slot], error) || !run_block(m_blocks[b], b, chunk_index, error) ||
            !m_device->record_event(m_free[slot], m_stream, error) ||
            (b + 2 < m_block_count && !copy_block(b + 2, error)))
        {
            return false;
        }
    }
    return true;
}

bool DitCudaRunner::forward(const uint16_t* tokens, const uint16_t* hdmap, float timestep, int64_t chunk_index,
                            float* out, Error& error)
{
    using Out = CudaKernels::GemmOutput;
    if (m_device == nullptr || tokens == nullptr || (hdmap != nullptr && !m_hdmap) || chunk_index < 0)
    {
        error.set("invalid DiT forward");
        return false;
    }
    const uint32_t L = m_chunk;

    for (uint32_t r = 0; r < L; ++r)
    {
        uint16_t* row = m_staging.data() + static_cast<size_t>(r) * EMBED_IN;
        std::memset(row, 0, EMBED_IN * 2);
        std::memcpy(row, tokens + static_cast<size_t>(r) * DIT_PATCH_IN, DIT_PATCH_IN * 2);
        if (hdmap != nullptr)
        {
            std::memcpy(row + DIT_PATCH_IN, hdmap + static_cast<size_t>(r) * DIT_HDMAP_IN, DIT_HDMAP_IN * 2);
        }
    }
    // rope.py RotaryPositionEmbedding3D.shift_t: positions (t h w), time offset by chunk * frames,
    // each angle outer(position, raw) (+ offset * raw for time) in float32
    float* angles = m_host_vectors.data();
    const float offset = static_cast<float>(chunk_index * m_config.frames);
    const uint32_t nt = static_cast<uint32_t>(m_raw_t.size());
    const uint32_t nh = static_cast<uint32_t>(m_raw_h.size());
    for (uint32_t t = 0; t < m_config.frames; ++t)
    {
        for (uint32_t h = 0; h < m_config.height; ++h)
        {
            for (uint32_t w = 0; w < m_config.width; ++w)
            {
                float* a = angles + ((static_cast<size_t>(t) * m_config.height + h) * m_config.width + w) * 64;
                for (uint32_t i = 0; i < nt; ++i)
                {
                    a[i] = static_cast<float>(t) * m_raw_t.data()[i] + offset * m_raw_t.data()[i];
                }
                for (uint32_t i = 0; i < nh; ++i)
                {
                    a[nt + i] = static_cast<float>(h) * m_raw_h.data()[i];
                    a[nt + nh + i] = static_cast<float>(w) * m_raw_w.data()[i];
                }
            }
        }
    }
    if (!m_device->upload(m_input, m_staging.data(), static_cast<size_t>(L) * EMBED_IN * 2, error) ||
        !m_device->upload(m_angles, angles, static_cast<size_t>(L) * 64 * 4, error))
    {
        return false;
    }

    // modules.py Timesteps (cos, sin of t * exp(-ln(10000) i / 1024)) on the scaled timestep, and
    // t_embedding_norm (RMSNorm, eps 1e-6) of it; both in double, rounded to float
    float* vectors = m_host_vectors.data();
    const float scaled = timestep * 0.001f;
    double squares = 0;
    for (uint32_t i = 0; i < D / 2; ++i)
    {
        const float exponent = -9.210340371976184f * static_cast<float>(i) / static_cast<float>(D / 2);
        const double angle = static_cast<double>(scaled) * static_cast<double>(std::exp(exponent));
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        vectors[i] = static_cast<float>(c);
        vectors[D / 2 + i] = static_cast<float>(s);
        squares += c * c + s * s;
    }
    const double inverse = 1.0 / std::sqrt(squares / D + 1e-6);
    for (uint32_t i = 0; i < D; ++i)
    {
        vectors[D + i] = static_cast<float>(static_cast<double>(vectors[i]) * inverse * m_t_norm.data()[i]);
    }
    if (!m_device->upload(m_vectors, vectors, 2 * D * 4, error))
    {
        return false;
    }

    // TimestepEmbedding with AdaLN-LoRA: adaln_lora = linear_2(silu(linear_1(sinusoid)));
    // per block and branch (Block.forward 1002-1009) and for the final layer (FinalLayer.forward):
    // linear_up(linear_down(silu(t_emb))) + adaln_lora
    bool ok = m_kernels.dit_matvec(m_t_linear1, m_vectors, V_SIN * 4, nullptr, 0, m_vectors, V_H1 * 4, D, D, false,
                                   m_stream, error) &&
              m_kernels.dit_matvec(m_t_linear2, m_vectors, V_H1 * 4, nullptr, 0, m_vectors, V_LORA * 4, 3 * D, D, true,
                                   m_stream, error);
    for (uint32_t b = 0; ok && b < m_block_count; ++b)
    {
        for (int branch = 0; ok && branch < 3; ++branch)
        {
            ok = m_kernels.dit_matvec(m_blocks[b].mod_down[branch], m_vectors, V_TEMB * 4, nullptr, 0, m_vectors,
                                      V_TMP * 4, DIT_LORA, D, true, m_stream, error) &&
                 m_kernels.dit_matvec(m_blocks[b].mod_up[branch], m_vectors, V_TMP * 4, &m_vectors, V_LORA * 4,
                                      m_modulation, (static_cast<uint64_t>(b) * 9 + branch * 3) * D * 4, 3 * D,
                                      DIT_LORA, false, m_stream, error);
        }
    }
    const uint64_t final_mod = static_cast<uint64_t>(m_block_count) * 9 * D * 4;
    ok = ok &&
         m_kernels.dit_matvec(m_final_mod_down, m_vectors, V_TEMB * 4, nullptr, 0, m_vectors, V_TMP * 4, DIT_LORA, D,
                              true, m_stream, error) &&
         m_kernels.dit_matvec(m_final_mod_up, m_vectors, V_TMP * 4, &m_vectors, V_LORA * 4, m_modulation, final_mod,
                              2 * D, DIT_LORA, false, m_stream, error) &&
         m_kernels.gemm_f16(m_input, m_embedder, m_x, L, D, EMBED_IN, true, m_stream, error, Out::Float);
    ok = ok && run_blocks(chunk_index, error);
    ok = ok &&
         m_kernels.dit_modulate_norm(m_x, m_modulation, final_mod, final_mod + uint64_t(D) * 4, m_h, L, D, true,
                                     m_stream, error) &&
         m_kernels.gemm_f16(m_h, m_final_linear, m_output, L, DIT_PATCH_OUT, D, true, m_stream, error, Out::Float) &&
         m_device->synchronize(m_stream, error) &&
         m_device->download(m_output, out, static_cast<size_t>(L) * DIT_PATCH_OUT * 4, error);
    return ok;
}

}
