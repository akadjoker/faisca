// faisca_dreams: an OmniDreams rollout on the GPU, all local. First frame + text embeddings +
// HD map frames in, generated video frames out.
//
// The pipeline is FlashDreams' OMNIDREAMS_PIPELINE_CONFIG (integrations_v2/omnidreams/config.py):
// lightvae encoder for the first frame and the HD map, the DiT with 2 flow matching steps
// (timesteps 1000, 500, shift 5) and a third pass at context noise 128 that writes the KV cache
// (DiffusionModel.generate / finalize, scheduler/fm.py), the first frame's latent pinned on the
// first chunk (CosmosTransformer._maybe_inject_image) and the lighttae decoder.
// Offline: the HD map is encoded before the DiT starts, so the encoder never shares the GPU with it.
#include "faisca/image.h"
#include "faisca/omnidreams.h"
#include "faisca/omnidreams_cuda.h"
#include "faisca/taehv_cuda.h"
#include "faisca/torch_ckpt.h"
#include "faisca/wanvae_cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{

constexpr uint32_t HEIGHT = 704;
constexpr uint32_t WIDTH = 1280;
constexpr uint32_t LH = HEIGHT / 8; // latent
constexpr uint32_t LW = WIDTH / 8;
constexpr uint32_t PH = LH / 2; // patches
constexpr uint32_t PW = LW / 2;
constexpr uint32_t LATENT_FRAMES = 2; // per chunk (len_t)

double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint16_t to_bf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_round(float value)
{
    const uint32_t bits = static_cast<uint32_t>(to_bf16(value)) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// splitmix64 + Box-Muller: our own normal noise (PyTorch's Philox stream is not reproduced)
struct Normal
{
    uint64_t state;
    bool has_spare = false;
    float spare = 0.0f;
    uint64_t next()
    {
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    float operator()()
    {
        if (has_spare)
        {
            has_spare = false;
            return spare;
        }
        const double u1 = (static_cast<double>(next() >> 11) + 1.0) * 0x1p-53;
        const double u2 = static_cast<double>(next() >> 11) * 0x1p-53;
        const double r = std::sqrt(-2.0 * std::log(u1));
        spare = static_cast<float>(r * std::sin(6.283185307179586 * u2));
        has_spare = true;
        return static_cast<float>(r * std::cos(6.283185307179586 * u2));
    }
};

// latent [t][16][LH][LW] <-> tokens [(t h w)][(c kh kw)] (network.py patchify, kt = 1, k = 2)
void patchify(const float* latent, float* tokens)
{
    for (uint32_t t = 0; t < LATENT_FRAMES; ++t)
        for (uint32_t y = 0; y < PH; ++y)
            for (uint32_t x = 0; x < PW; ++x)
            {
                float* token = tokens + ((size_t(t) * PH + y) * PW + x) * 64;
                for (uint32_t c = 0; c < 16; ++c)
                    for (uint32_t ky = 0; ky < 2; ++ky)
                        for (uint32_t kx = 0; kx < 2; ++kx)
                        {
                            token[c * 4 + ky * 2 + kx] =
                                latent[((size_t(t) * 16 + c) * LH + y * 2 + ky) * LW + x * 2 + kx];
                        }
            }
}

void unpatchify(const float* tokens, float* latent)
{
    for (uint32_t t = 0; t < LATENT_FRAMES; ++t)
        for (uint32_t y = 0; y < PH; ++y)
            for (uint32_t x = 0; x < PW; ++x)
            {
                const float* token = tokens + ((size_t(t) * PH + y) * PW + x) * 64;
                for (uint32_t c = 0; c < 16; ++c)
                    for (uint32_t ky = 0; ky < 2; ++ky)
                        for (uint32_t kx = 0; kx < 2; ++kx)
                        {
                            latent[((size_t(t) * 16 + c) * LH + y * 2 + ky) * LW + x * 2 + kx] =
                                token[c * 4 + ky * 2 + kx];
                        }
            }
}

// RGB8 rows -> 3 x HEIGHT x WIDTH floats in [-1, 1] (x / 127.5 - 1, the driver's scaling)
void to_planes(const uint8_t* rgb, float* planes)
{
    const size_t plane = size_t(HEIGHT) * WIDTH;
    for (size_t p = 0; p < plane; ++p)
    {
        for (int c = 0; c < 3; ++c)
        {
            planes[c * plane + p] = static_cast<float>(rgb[p * 3 + c]) / 127.5f - 1.0f;
        }
    }
}

bool write_ppm(const std::string& path, const float* planes)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
    {
        return false;
    }
    std::fprintf(file, "P6\n%u %u\n255\n", WIDTH, HEIGHT);
    const size_t plane = size_t(HEIGHT) * WIDTH;
    std::vector<uint8_t> row(size_t(WIDTH) * 3);
    bool ok = true;
    for (uint32_t y = 0; y < HEIGHT && ok; ++y)
    {
        for (uint32_t x = 0; x < WIDTH; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                const float v = planes[c * plane + size_t(y) * WIDTH + x];
                row[x * 3 + c] = static_cast<uint8_t>(std::lround(std::fmin(std::fmax(v, 0.0f), 1.0f) * 255.0f));
            }
        }
        ok = std::fwrite(row.data(), 1, row.size(), file) == row.size();
    }
    return std::fclose(file) == 0 && ok;
}

const char* option(int argc, char** argv, const char* name, const char* fallback)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::strcmp(argv[i], name) == 0)
        {
            return argv[i + 1];
        }
    }
    return fallback;
}

}

int main(int argc, char** argv)
{
    const char* checkpoint = option(argc, argv, "--dit", nullptr);
    const char* vae = option(argc, argv, "--vae", nullptr);
    const char* tae = option(argc, argv, "--tae", nullptr);
    const char* text_path = option(argc, argv, "--text", nullptr);
    const char* image_path = option(argc, argv, "--image", nullptr);
    const char* hdmap_path = option(argc, argv, "--hdmap", nullptr);
    const char* out_dir = option(argc, argv, "--out", nullptr);
    const int frames = std::atoi(option(argc, argv, "--frames", "21"));
    const uint64_t seed = std::strtoull(option(argc, argv, "--seed", "42"), nullptr, 10);
    bool ppm = false;
    for (int i = 1; i < argc; ++i)
    {
        ppm = ppm || std::strcmp(argv[i], "--ppm") == 0;
    }
    if (checkpoint == nullptr || vae == nullptr || tae == nullptr || text_path == nullptr || image_path == nullptr ||
        hdmap_path == nullptr || out_dir == nullptr || frames < 5 || (frames - 5) % 8 != 0)
    {
        std::printf("usage: faisca_dreams --dit <omnidreams .pt> --vae <lightvaew2_1.pth> --tae <lighttaew2_1.pth>\n"
                    "         --text <text_embeddings.pt> --image <first frame 1280x704 png>\n"
                    "         --hdmap <frames x 704 x 1280 RGB8 raw> --frames <5 + 8k> --out <dir> [--seed 42] [--ppm]\n");
        return 2;
    }
    const uint32_t chunks = 1 + static_cast<uint32_t>(frames - 5) / 8;
    const size_t frame_floats = size_t(3) * HEIGHT * WIDTH;
    const size_t latent_floats = size_t(16) * LH * LW;
    const size_t L = size_t(LATENT_FRAMES) * PH * PW;
    faisca::Error error;
    const double start = now();

    // inputs
    faisca::Image image;
    if (!faisca::read_image(image_path, image, error) || image.width != WIDTH || image.height != HEIGHT)
    {
        std::printf("first frame: %s (needs %u x %u RGB)\n", error.message[0] ? error.message : "wrong size", WIDTH,
                    HEIGHT);
        return 1;
    }
    std::vector<uint8_t> rgb(image.rgb.data(), image.rgb.data() + size_t(WIDTH) * HEIGHT * 3);
    std::vector<uint8_t> hdmap(size_t(frames) * WIDTH * HEIGHT * 3);
    FILE* hd = std::fopen(hdmap_path, "rb");
    if (hd == nullptr || std::fread(hdmap.data(), 1, hdmap.size(), hd) != hdmap.size())
    {
        std::printf("cannot read %d HD map frames from %s\n", frames, hdmap_path);
        return 1;
    }
    std::fclose(hd);

    faisca::CudaDevice device;
    if (!device.init(error))
    {
        std::printf("CUDA: %s\n", error.message);
        return 1;
    }

    // phase 1: the encoder alone; image latent, then the HD map latents chunk by chunk
    std::vector<float> image_latent(latent_floats), hd_latents(size_t(chunks) * LATENT_FRAMES * latent_floats);
    {
        faisca::WanVaeCudaEncoder encoder;
        std::vector<float> planes(size_t(8) * frame_floats);
        uint32_t produced = 0;
        to_planes(rgb.data(), planes.data());
        if (!encoder.init(device, vae, HEIGHT, WIDTH, error) ||
            !encoder.encode(planes.data(), 1, image_latent.data(), produced, error) || produced != 1)
        {
            std::printf("encoder: %s\n", error.message);
            return 1;
        }
        encoder.reset();
        size_t offset = 0;
        for (uint32_t k = 0; k < chunks; ++k)
        {
            const uint32_t count = k == 0 ? 5 : 8;
            for (uint32_t f = 0; f < count; ++f)
            {
                to_planes(hdmap.data() + (offset + f) * size_t(WIDTH) * HEIGHT * 3, planes.data() + f * frame_floats);
            }
            offset += count;
            if (!encoder.encode(planes.data(), count, hd_latents.data() + size_t(k) * LATENT_FRAMES * latent_floats,
                                produced, error) ||
                produced != LATENT_FRAMES)
            {
                std::printf("encoder: %s (%u latent frames)\n", error.message, produced);
                return 1;
            }
        }
        std::printf("encoded the first frame and %d HD map frames in %.1f s\n", frames, now() - start);
    }

    // phase 2: the DiT (block weights streamed), the text, the decoder
    faisca::DitCheckpoint dit_weights;
    faisca::CkptFile text;
    if (!dit_weights.open(checkpoint, error) || !text.open(text_path, error))
    {
        std::printf("%s\n", error.message);
        return 1;
    }
    const faisca::CkptTensor* text_tensor = text.find("");
    if (text_tensor == nullptr || text_tensor->dtype != faisca::CkptDtype::BF16 ||
        text_tensor->n_elements != size_t(faisca::DIT_TEXT_TOKENS) * faisca::DIT_TEXT_IN)
    {
        std::printf("%s: expected one bf16 tensor of 512 x 100352\n", text_path);
        return 1;
    }
    faisca::DitConfig config;
    config.stream_weights = true;
    faisca::DitCudaRunner dit;
    faisca::TaehvCudaDecoder decoder;
    if (!dit.init(device, dit_weights.weights(), config, error) ||
        !dit.set_text(reinterpret_cast<const uint16_t*>(text_tensor->data), error) ||
        !decoder.init(device, tae, LH, LW, error))
    {
        std::printf("%s\n", error.message);
        return 1;
    }
    size_t free_bytes = 0, total_bytes = 0;
    device.memory_info(free_bytes, total_bytes, error);
    std::printf("DiT and decoder ready at %.1f s, %.2f GB of device memory free\n", now() - start,
                static_cast<double>(free_bytes) / 1e9);

    // scheduler/fm.py: sigmas = warp(linspace(1, 0, 1001)[:-1], shift 5), timesteps = sigmas * 1000;
    // steps 1000 and 500 read at index 1000 - t; the network gets the timestep in bf16
    const float shift = 5.0f;
    const auto warp = [&](float s) { return shift * s / (1.0f + (shift - 1.0f) * s); };
    const auto linspace = [](int i) { return 1.0f - static_cast<float>(i) * (1.0f / 1000.0f); };
    const int step_index[2] = {0, 500};
    float sigma[2], timestep[2];
    for (int i = 0; i < 2; ++i)
    {
        sigma[i] = warp(linspace(step_index[i]));
        timestep[i] = bf16_round(sigma[i] * 1000.0f);
    }
    // finalize: add_noise at the table entry nearest to timestep 128
    int nearest = 0;
    for (int i = 1; i < 1000; ++i)
    {
        if (std::fabs(warp(linspace(i)) * 1000.0f - 128.0f) < std::fabs(warp(linspace(nearest)) * 1000.0f - 128.0f))
        {
            nearest = i;
        }
    }
    const float context_sigma = warp(linspace(nearest));
    std::printf("steps: sigma %.4f t %.1f, sigma %.4f t %.1f; context noise sigma %.4f\n", sigma[0], timestep[0],
                sigma[1], timestep[1], context_sigma);

    Normal normal{seed};
    std::vector<float> image_tokens(L * 64), hd_tokens(L * 64), noisy(L * 64), clean(L * 64), flow(L * 64);
    std::vector<uint16_t> input(L * faisca::DIT_PATCH_IN), hd_input(L * faisca::DIT_HDMAP_IN);
    std::vector<float> two_image(LATENT_FRAMES * latent_floats, 0.0f), latents(LATENT_FRAMES * latent_floats);
    std::memcpy(two_image.data(), image_latent.data(), latent_floats * 4);
    patchify(two_image.data(), image_tokens.data());
    const size_t first_frame_tokens = size_t(PH) * PW;
    std::vector<float> video(size_t(8) * frame_floats);
    FILE* raw = std::fopen((std::string(out_dir) + "/video.rgb").c_str(), "wb");
    if (raw == nullptr)
    {
        std::printf("cannot write into %s\n", out_dir);
        return 1;
    }
    int written = 0;
    for (uint32_t k = 0; k < chunks; ++k)
    {
        const double chunk_start = now();
        patchify(hd_latents.data() + size_t(k) * LATENT_FRAMES * latent_floats, hd_tokens.data());
        for (size_t i = 0; i < hd_tokens.size(); ++i)
        {
            hd_input[i] = to_bf16(hd_tokens[i]);
        }
        // the network's input tokens: the latent (the first frame's tokens replaced by the image on
        // chunk 0), then the patchified condition mask (1 there, 0 elsewhere)
        const auto run = [&](const std::vector<float>& latent, float t) {
            for (size_t token = 0; token < L; ++token)
            {
                const bool pinned = k == 0 && token < first_frame_tokens;
                const float* source = pinned ? &image_tokens[token * 64] : &latent[token * 64];
                uint16_t* row = &input[token * faisca::DIT_PATCH_IN];
                for (int c = 0; c < 64; ++c)
                {
                    row[c] = to_bf16(source[c]);
                }
                for (int c = 64; c < 68; ++c)
                {
                    row[c] = to_bf16(pinned ? 1.0f : 0.0f);
                }
            }
            return dit.forward(input.data(), hd_input.data(), t, k, flow.data(), error);
        };
        for (size_t i = 0; i < noisy.size(); ++i)
        {
            noisy[i] = normal();
        }
        for (int step = 0; step < 2; ++step)
        {
            if (step > 0)
            {
                for (size_t i = 0; i < noisy.size(); ++i)
                {
                    noisy[i] = (1.0f - sigma[step]) * clean[i] + sigma[step] * normal();
                }
            }
            if (!run(noisy, timestep[step]))
            {
                std::printf("DiT: %s\n", error.message);
                return 1;
            }
            for (size_t i = 0; i < clean.size(); ++i)
            {
                clean[i] = noisy[i] - sigma[step] * flow[i];
            }
        }
        if (k == 0)
        {
            std::memcpy(clean.data(), image_tokens.data(), first_frame_tokens * 64 * 4);
        }
        // finalize: the clean latent renoised to context noise 128 writes the KV cache
        for (size_t i = 0; i < noisy.size(); ++i)
        {
            noisy[i] = (1.0f - context_sigma) * clean[i] + context_sigma * normal();
        }
        if (!run(noisy, 128.0f))
        {
            std::printf("DiT: %s\n", error.message);
            return 1;
        }
        const double dit_done = now();
        unpatchify(clean.data(), latents.data());
        uint32_t produced = 0;
        if (!decoder.decode(latents.data(), LATENT_FRAMES, video.data(), produced, error))
        {
            std::printf("decoder: %s\n", error.message);
            return 1;
        }
        std::vector<uint8_t> bytes(size_t(WIDTH) * HEIGHT * 3);
        for (uint32_t f = 0; f < produced; ++f)
        {
            const float* planes = video.data() + f * frame_floats;
            if (ppm)
            {
                char name[64];
                std::snprintf(name, sizeof(name), "/frame_%04d.ppm", written);
                write_ppm(std::string(out_dir) + name, planes);
            }
            const size_t plane = size_t(HEIGHT) * WIDTH;
            for (size_t p = 0; p < plane; ++p)
            {
                for (int c = 0; c < 3; ++c)
                {
                    bytes[p * 3 + c] = static_cast<uint8_t>(
                        std::lround(std::fmin(std::fmax(planes[c * plane + p], 0.0f), 1.0f) * 255.0f));
                }
            }
            std::fwrite(bytes.data(), 1, bytes.size(), raw);
            ++written;
        }
        std::printf("chunk %u/%u: DiT 3 passes %.1f s, decode %.2f s, %u frames, %.0f s so far\n", k + 1, chunks,
                    dit_done - chunk_start, now() - dit_done, produced, now() - start);
        std::fflush(stdout);
    }
    std::fclose(raw);
    std::printf("%d frames in %s (video.rgb: ffmpeg -f rawvideo -pix_fmt rgb24 -s %ux%u -r 30 -i video.rgb "
                "-pix_fmt yuv420p video.mp4), %.1f s in all\n",
                written, out_dir, WIDTH, HEIGHT, now() - start);
    return 0;
}
