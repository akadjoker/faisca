#include "faisca/torch_ckpt.h"

#include <chrono>
#include <cstdio>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: ckpt_dump <model.ckpt>\n");
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    faisca::CkptFile file;
    faisca::Error error;
    if (!file.open(argv[1], error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    uint64_t total = 0;
    for (size_t i = 0; i < file.tensor_count(); ++i)
    {
        const faisca::CkptTensor& t = file.tensor(i);
        std::printf("%.*s\t%s\t", static_cast<int>(t.name.size()), t.name.data(), faisca::ckpt_dtype_name(t.dtype));
        for (uint32_t d = 0; d < t.n_dims; ++d)
        {
            std::printf(d == 0 ? "%llu" : ",%llu", static_cast<unsigned long long>(t.shape[d]));
        }
        std::printf("\t%llu\t%llu\n", static_cast<unsigned long long>(t.data - file.data()),
                    static_cast<unsigned long long>(t.n_bytes));
        total += t.n_elements;
    }
    std::fprintf(stderr, "%zu tensors, %llu elements, open %.2f ms\n", file.tensor_count(),
                 static_cast<unsigned long long>(total), ms);
    return 0;
}
