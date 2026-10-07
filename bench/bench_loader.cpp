#include "faisca/gguf.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace faisca;

namespace
{

double microseconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

void report(const char* what, std::vector<double>& samples, const char* unit)
{
    std::sort(samples.begin(), samples.end());
    std::printf("  %-44s best %9.3f  median %9.3f  %s\n", what, samples.front(), samples[samples.size() / 2], unit);
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: bench_loader <model.gguf> [rounds]\n");
        return 2;
    }
    const char* path = argv[1];
    const int rounds = argc > 2 ? std::max(10, std::atoi(argv[2])) : 2000;

    Error error;
    MappedFile image;
    if (!image.open(path, error))
    {
        std::fprintf(stderr, "bench_loader: %s\n", error.c_str());
        return 1;
    }

    uint64_t checksum = 0;

    std::vector<double> parse_times;
    for (int round = 0; round < rounds + 20; ++round)
    {
        const auto start = std::chrono::steady_clock::now();
        GgufFile file;
        if (!file.parse(image.data(), image.size(), error))
        {
            std::fprintf(stderr, "bench_loader: %s\n", error.c_str());
            return 1;
        }
        checksum += file.metadata().size() + file.tensors().size() + file.data_offset();
        const double elapsed = microseconds_since(start);
        if (round >= 20)
        {
            parse_times.push_back(elapsed);
        }
    }

    std::vector<double> open_times;
    for (int round = 0; round < rounds / 4 + 20; ++round)
    {
        const auto start = std::chrono::steady_clock::now();
        GgufFile file;
        if (!file.open(path, error))
        {
            std::fprintf(stderr, "bench_loader: %s\n", error.c_str());
            return 1;
        }
        checksum += file.tensors().size();
        file.close();
        const double elapsed = microseconds_since(start);
        if (round >= 20)
        {
            open_times.push_back(elapsed);
        }
    }

    GgufFile file;
    if (!file.parse(image.data(), image.size(), error))
    {
        return 1;
    }
    const size_t tensor_count = file.tensors().size();
    const size_t key_count = file.metadata().size();

    std::vector<double> tensor_lookup_times;
    std::vector<double> key_lookup_times;
    const int lookup_rounds = 2000;
    for (int round = 0; round < 60; ++round)
    {
        auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < lookup_rounds; ++repeat)
        {
            for (const GgufTensor& tensor : file.tensors())
            {
                checksum += file.find_tensor(tensor.name)->n_bytes;
            }
        }
        double elapsed = microseconds_since(start);
        tensor_lookup_times.push_back(elapsed * 1000.0 / (static_cast<double>(tensor_count) * lookup_rounds));

        start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < lookup_rounds; ++repeat)
        {
            for (const GgufKeyValue& kv : file.metadata())
            {
                checksum += static_cast<uint64_t>(file.find(kv.key)->type);
            }
        }
        elapsed = microseconds_since(start);
        key_lookup_times.push_back(elapsed * 1000.0 / (static_cast<double>(key_count) * lookup_rounds));
    }

    std::printf("%s: %zu metadata keys, %zu tensors, checksum %" PRIu64 "\n", FAISCA_BENCH_NAME, key_count,
                tensor_count, checksum);
    report("parse header, metadata and tensor list", parse_times, "us");
    report("open: map + parse + unmap", open_times, "us");
    report("find_tensor by name", tensor_lookup_times, "ns per lookup");
    report("find metadata key", key_lookup_times, "ns per lookup");
    return 0;
}
