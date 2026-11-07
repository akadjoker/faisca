#include "faisca/triposr.h"

#include <chrono>
#include <cstring>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct Dump
{
    std::string dir;
};

void write_trace(void* user, const char* name, int layer, const float* data, size_t count)
{
    const Dump& dump = *static_cast<const Dump*>(user);
    std::string path = dump.dir + "/" + name + (layer >= 0 ? std::to_string(layer) : std::string()) + ".f32";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr || std::fwrite(data, sizeof(float), count, f) != count)
    {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
    }
    if (f != nullptr)
    {
        std::fclose(f);
    }
}

double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: triposr_dump <model.ckpt> <image .png/.jpg or reference dir with rgb_cond.f32> <out dir, - for no traces> [threads] [mesh resolution, 64]\n");
        return 2;
    }
    const int threads = argc > 4 ? std::atoi(argv[4]) : static_cast<int>(std::thread::hardware_concurrency());

    const size_t size = faisca::TripoModel::IMAGE_SIZE;
    std::vector<float> rgb(size * size * 3);
    const std::string input = argv[2];
    const bool from_image = input.size() > 4 && (input.compare(input.size() - 4, 4, ".png") == 0 ||
                                                  input.compare(input.size() - 4, 4, ".jpg") == 0);
    if (from_image)
    {
        faisca::Error error;
        faisca::Image image;
        faisca::Array<float> prepared;
        if (!faisca::read_image(input.c_str(), image, error) ||
            !faisca::tripo_prepare_image(image, 0.85f, prepared, error))
        {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        std::memcpy(rgb.data(), prepared.data(), rgb.size() * sizeof(float));
    }
    else
    {
        const std::string rgb_path = input + "/rgb_cond.f32";
        FILE* f = std::fopen(rgb_path.c_str(), "rb");
        if (f == nullptr || std::fread(rgb.data(), sizeof(float), rgb.size(), f) != rgb.size())
        {
            std::fprintf(stderr, "cannot read %s\n", rgb_path.c_str());
            return 1;
        }
        std::fclose(f);
    }

    faisca::Error error;
    auto start = std::chrono::steady_clock::now();
    faisca::TripoModel model;
    if (!model.open(argv[1], error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "open %.3f s\n", seconds_since(start));

    faisca::ThreadPool pool;
    faisca::TripoRunner runner;
    if (!pool.start(threads, error) || !runner.init(model, pool, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    Dump dump = {argv[3]};
    if (dump.dir != "-")
    {
        runner.set_trace(write_trace, &dump);
        write_trace(&dump, "rgb_cond", -1, rgb.data(), rgb.size());
    }

    start = std::chrono::steady_clock::now();
    if (!runner.encode_image(rgb.data(), error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "image encoder %.3f s with %d threads\n", seconds_since(start), threads);
    start = std::chrono::steady_clock::now();
    if (!runner.build_triplane(error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "triplane %.3f s\n", seconds_since(start));

    const uint32_t resolution = argc > 5 ? static_cast<uint32_t>(std::atoi(argv[5])) : 64;
    start = std::chrono::steady_clock::now();
    faisca::TripoMesh mesh;
    if (!runner.extract_mesh(resolution, 25.0f, mesh, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "mesh %u^3: %zu vertices, %zu faces, %.3f s\n", resolution, mesh.positions.size() / 3,
                 mesh.faces.size() / 3, seconds_since(start));
    if (dump.dir != "-")
    {
        write_trace(&dump, "mc_vertices", -1, mesh.positions.data(), mesh.positions.size());
        write_trace(&dump, "mc_colors", -1, mesh.colors.data(), mesh.colors.size());
        const std::string faces = dump.dir + "/mc_faces.u32";
        FILE* out = std::fopen(faces.c_str(), "wb");
        if (out != nullptr)
        {
            std::fwrite(mesh.faces.data(), sizeof(uint32_t), mesh.faces.size(), out);
            std::fclose(out);
        }
    }
    return 0;
}
