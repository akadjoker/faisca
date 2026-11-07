#include "faisca/triposr.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace
{

template <typename T>
std::vector<T> read_file(const std::string& path)
{
    std::vector<T> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr)
    {
        return out;
    }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(static_cast<size_t>(bytes) / sizeof(T));
    if (std::fread(out.data(), sizeof(T), out.size(), f) != out.size())
    {
        out.clear();
    }
    std::fclose(f);
    return out;
}

typedef std::array<uint32_t, 3> Key;

Key key_of(const float* v)
{
    Key k;
    std::memcpy(k.data(), v, sizeof(k));
    return k;
}

std::vector<std::array<Key, 3>> triangles(const std::vector<float>& v, const uint32_t* faces, size_t n_faces)
{
    std::vector<std::array<Key, 3>> out;
    for (size_t i = 0; i < n_faces; ++i)
    {
        std::array<Key, 3> t = {key_of(&v[3 * faces[3 * i]]), key_of(&v[3 * faces[3 * i + 1]]),
                                key_of(&v[3 * faces[3 * i + 2]])};
        const size_t first = static_cast<size_t>(std::min_element(t.begin(), t.end()) - t.begin());
        std::rotate(t.begin(), t.begin() + static_cast<long>(first), t.end());
        out.push_back(t);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}

// the reference density grid through tripo_marching_cubes: vertices and oriented
// triangles must equal torchmcubes' bit for bit (as sets; its slab order is OpenMP's)
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: triposr_mc_check <reference dir> <resolution>\n");
        return 2;
    }
    const std::string dir = argv[1];
    const uint32_t resolution = static_cast<uint32_t>(std::atoi(argv[2]));
    std::vector<float> vol = read_file<float>(dir + "/density_act.f32");
    const std::vector<float> ref_v = read_file<float>(dir + "/mc_vertices.f32");
    const std::vector<uint32_t> ref_f = read_file<uint32_t>(dir + "/mc_faces.i32");
    if (vol.size() != size_t(resolution) * resolution * resolution || ref_v.empty() || ref_f.empty())
    {
        std::fprintf(stderr, "reference files missing or of the wrong size\n");
        return 1;
    }
    for (float& v : vol)
    {
        v = v - 25.0f;
    }
    faisca::Error error;
    faisca::ThreadPool pool;
    faisca::Array<float> points;
    faisca::Array<uint32_t> faces;
    if (!pool.start(static_cast<int>(std::thread::hardware_concurrency()), error) ||
        !faisca::tripo_marching_cubes(pool, vol.data(), resolution, points, faces, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    const float r = faisca::TripoModel::RADIUS;
    const float scale = static_cast<float>(resolution) - 1.0f;
    std::vector<float> ours(points.size());
    for (size_t i = 0; i < points.size() / 3; ++i)
    {
        const float v[3] = {points[3 * i + 2] / scale, points[3 * i + 1] / scale, points[3 * i] / scale};
        for (int d = 0; d < 3; ++d)
        {
            ours[3 * i + size_t(d)] = v[d] * (r - -r) + -r;
        }
    }
    std::vector<Key> a, b;
    for (size_t i = 0; i < ours.size(); i += 3)
    {
        a.push_back(key_of(&ours[i]));
    }
    for (size_t i = 0; i < ref_v.size(); i += 3)
    {
        b.push_back(key_of(&ref_v[i]));
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    const bool same_vertices = a == b;
    const bool same_triangles =
        triangles(ours, faces.data(), faces.size() / 3) == triangles(ref_v, ref_f.data(), ref_f.size() / 3);
    std::printf("%s triposr_mc_check %u^3: %zu vertices (reference %zu), %zu faces (reference %zu), vertices %s, "
                "triangles %s\n",
                same_vertices && same_triangles ? "ok  " : "FAIL", resolution, ours.size() / 3, ref_v.size() / 3,
                faces.size() / 3, ref_f.size() / 3, same_vertices ? "identical" : "differ",
                same_triangles ? "identical" : "differ");
    return same_vertices && same_triangles ? 0 : 1;
}
