#include "faisca/gguf.h"

#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{

double microseconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

struct Timing
{
    std::vector<double> samples;

    double best()
    {
        std::sort(samples.begin(), samples.end());
        return samples.front();
    }

    double median()
    {
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }
};

void print_row(const char* what, Timing& ggml, Timing& faisca, const char* unit)
{
    std::printf("  %-34s ggml %10.2f (med %10.2f)   faisca %9.2f (med %9.2f)  %s   %.1fx\n", what, ggml.best(),
                ggml.median(), faisca.best(), faisca.median(), unit, ggml.best() / faisca.best());
}

bool same_contents(const gguf_context* reference, const faisca::GgufFile& file)
{
    if (static_cast<size_t>(gguf_get_n_kv(reference)) != file.metadata().size() ||
        static_cast<size_t>(gguf_get_n_tensors(reference)) != file.tensors().size() ||
        gguf_get_alignment(reference) != file.alignment() || gguf_get_data_offset(reference) != file.data_offset())
    {
        return false;
    }

    for (size_t i = 0; i < file.metadata().size(); ++i)
    {
        const faisca::GgufKeyValue& kv = file.metadata()[i];
        const int64_t id = static_cast<int64_t>(i);
        if (kv.key != gguf_get_key(reference, id) ||
            static_cast<int>(kv.value.type) != static_cast<int>(gguf_get_kv_type(reference, id)))
        {
            return false;
        }
        if (kv.value.type == faisca::GgufType::Array &&
            (kv.value.array.count != gguf_get_arr_n(reference, id) ||
             static_cast<int>(kv.value.array.elem_type) != static_cast<int>(gguf_get_arr_type(reference, id))))
        {
            return false;
        }
    }

    for (size_t i = 0; i < file.tensors().size(); ++i)
    {
        const faisca::GgufTensor& tensor = file.tensors()[i];
        const int64_t id = static_cast<int64_t>(i);
        if (tensor.name != gguf_get_tensor_name(reference, id) || tensor.offset != gguf_get_tensor_offset(reference, id) ||
            tensor.n_bytes != gguf_get_tensor_size(reference, id) ||
            static_cast<int>(tensor.type) != static_cast<int>(gguf_get_tensor_type(reference, id)))
        {
            return false;
        }
    }
    return true;
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: compare_ggml <model.gguf> [rounds]\n");
        return 2;
    }
    const char* path = argv[1];
    const int rounds = argc > 2 ? std::max(10, std::atoi(argv[2])) : 200;

    faisca::Error error;
    faisca::MappedFile image;
    if (!image.open(path, error))
    {
        std::fprintf(stderr, "compare_ggml: %s\n", error.c_str());
        return 1;
    }

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context* reference = gguf_init_from_file(path, params);
    faisca::GgufFile file;
    if (reference == nullptr || !file.open(path, error))
    {
        std::fprintf(stderr, "compare_ggml: cannot read '%s' with both libraries\n", path);
        return 1;
    }
    if (!same_contents(reference, file))
    {
        std::fprintf(stderr, "compare_ggml: ggml and faisca disagree about '%s'\n", path);
        return 1;
    }
    std::printf("%s\n%zu metadata keys and %zu tensors, identical in both libraries\n", path, file.metadata().size(),
                file.tensors().size());

    uint64_t sink = 0;
    Timing ggml_buffer;
    Timing faisca_buffer;
    Timing ggml_file;
    Timing faisca_file;

    for (int round = -3; round < rounds; ++round)
    {
        auto start = std::chrono::steady_clock::now();
        gguf_context* context = gguf_init_from_buffer(image.data(), image.size(), params);
        sink += static_cast<uint64_t>(gguf_get_n_tensors(context));
        gguf_free(context);
        double elapsed = microseconds_since(start);
        if (round >= 0)
        {
            ggml_buffer.samples.push_back(elapsed);
        }

        start = std::chrono::steady_clock::now();
        {
            faisca::GgufFile parsed;
            parsed.parse(image.data(), image.size(), error);
            sink += parsed.tensors().size();
        }
        elapsed = microseconds_since(start);
        if (round >= 0)
        {
            faisca_buffer.samples.push_back(elapsed);
        }

        start = std::chrono::steady_clock::now();
        context = gguf_init_from_file(path, params);
        sink += static_cast<uint64_t>(gguf_get_n_tensors(context));
        gguf_free(context);
        elapsed = microseconds_since(start);
        if (round >= 0)
        {
            ggml_file.samples.push_back(elapsed);
        }

        start = std::chrono::steady_clock::now();
        {
            faisca::GgufFile opened;
            opened.open(path, error);
            sink += opened.tensors().size();
        }
        elapsed = microseconds_since(start);
        if (round >= 0)
        {
            faisca_file.samples.push_back(elapsed);
        }
    }

    std::vector<std::string> names;
    for (const faisca::GgufTensor& tensor : file.tensors())
    {
        names.emplace_back(tensor.name);
    }

    Timing ggml_find;
    Timing faisca_find;
    const int repeats = 200;
    for (int round = 0; round < 30; ++round)
    {
        auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            for (const std::string& name : names)
            {
                sink += static_cast<uint64_t>(gguf_find_tensor(reference, name.c_str()));
            }
        }
        ggml_find.samples.push_back(microseconds_since(start) * 1000.0 / (static_cast<double>(names.size()) * repeats));

        start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            for (const std::string& name : names)
            {
                sink += file.find_tensor(name)->offset;
            }
        }
        faisca_find.samples.push_back(microseconds_since(start) * 1000.0 /
                                      (static_cast<double>(names.size()) * repeats));
    }

    std::printf("\n");
    print_row("parse from memory", ggml_buffer, faisca_buffer, "us");
    print_row("open from disk (page cache warm)", ggml_file, faisca_file, "us");
    print_row("find a tensor by name", ggml_find, faisca_find, "ns");
    std::printf("\nsink %" PRIu64 "\n", sink);

    gguf_free(reference);
    return 0;
}
