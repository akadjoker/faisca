#include "faisca/gguf.h"
#include "faisca/tokenizer.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace
{

double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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

struct Workload
{
    const char* name;
    std::vector<std::string_view> texts;
    uint64_t bytes = 0;
    uint64_t tokens = 0;
};

uint64_t mix(uint64_t hash, uint64_t value)
{
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return hash;
}

uint64_t hash_tokens(const int32_t* tokens, size_t count, uint64_t hash)
{
    for (size_t i = 0; i < count; ++i)
    {
        hash = mix(hash, static_cast<uint64_t>(static_cast<uint32_t>(tokens[i])));
    }
    return hash;
}

uint64_t hash_bytes(const char* data, size_t count, uint64_t hash)
{
    for (size_t i = 0; i < count; ++i)
    {
        hash = mix(hash, static_cast<uint8_t>(data[i]));
    }
    return hash;
}

void quiet(ggml_log_level, const char*, void*)
{
}

void print_row(const char* what, Timing& llama, Timing& faisca, const char* unit)
{
    std::printf("  %-32s llama.cpp %9.2f (med %9.2f)   faisca %9.2f (med %9.2f)  %s   %.1fx\n", what, llama.best(),
                llama.median(), faisca.best(), faisca.median(), unit, llama.best() / faisca.best());
}

class LlamaVocab
{
public:
    bool load(const char* path)
    {
        llama_model_params params = llama_model_default_params();
        params.vocab_only = true;
        m_model = llama_model_load_from_file(path, params);
        if (m_model == nullptr)
        {
            return false;
        }
        m_vocab = llama_model_get_vocab(m_model);
        return true;
    }

    void free()
    {
        if (m_model != nullptr)
        {
            llama_model_free(m_model);
            m_model = nullptr;
            m_vocab = nullptr;
        }
    }

    ~LlamaVocab()
    {
        free();
    }

    const llama_vocab* vocab() const
    {
        return m_vocab;
    }

private:
    llama_model* m_model = nullptr;
    const llama_vocab* m_vocab = nullptr;
};

struct Corpus
{
    std::vector<std::string> files;
    std::vector<Workload> workloads;
    std::vector<int32_t> tokens;
    std::vector<size_t> token_starts;
};

bool read_file(const char* path, std::string& out)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    return true;
}

void split_lines(std::string_view text, std::vector<std::string_view>& out)
{
    size_t start = 0;
    while (start < text.size())
    {
        size_t end = text.find('\n', start);
        if (end == std::string_view::npos)
        {
            end = text.size();
        }
        if (end > start)
        {
            out.push_back(text.substr(start, end - start));
        }
        start = end + 1;
    }
}

}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: bench_tokenizer <model.gguf> <text files...> [--rounds N]\n");
        return 2;
    }

    const char* model_path = argv[1];
    int rounds = 30;
    Corpus corpus;
    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--rounds") == 0 && i + 1 < argc)
        {
            rounds = std::max(3, std::atoi(argv[++i]));
            continue;
        }
        std::string text;
        if (!read_file(argv[i], text))
        {
            std::fprintf(stderr, "bench_tokenizer: cannot read '%s'\n", argv[i]);
            return 1;
        }
        corpus.files.push_back(std::move(text));
    }

    Workload whole{"encode whole files", {}, 0, 0};
    Workload lines{"encode line by line", {}, 0, 0};
    for (const std::string& file : corpus.files)
    {
        whole.texts.push_back(file);
        split_lines(file, lines.texts);
    }
    for (Workload* workload : {&whole, &lines})
    {
        for (const std::string_view text : workload->texts)
        {
            workload->bytes += text.size();
        }
    }

    llama_log_set(quiet, nullptr);
    llama_backend_init();

    faisca::Error error;
    faisca::GgufFile file;
    faisca::Tokenizer tokenizer;
    if (!file.open(model_path, error) || !tokenizer.load(file, error))
    {
        std::fprintf(stderr, "bench_tokenizer: %s\n", error.c_str());
        return 1;
    }

    LlamaVocab reference;
    if (!reference.load(model_path))
    {
        std::fprintf(stderr, "bench_tokenizer: llama.cpp cannot load '%s'\n", model_path);
        return 1;
    }

    std::printf("%s\n%zu tokens in the vocabulary, %zu text files, %" PRIu64 " bytes whole, %zu lines\n", model_path,
                tokenizer.vocab_size(), corpus.files.size(), whole.bytes, lines.texts.size());

    // reference tokens of the whole files, for the decode workloads
    for (const std::string_view text : whole.texts)
    {
        std::vector<llama_token> out(text.size() * 3 + 16);
        const int32_t count = llama_tokenize(reference.vocab(), text.data(), static_cast<int32_t>(text.size()),
                                             out.data(), static_cast<int32_t>(out.size()), false, true);
        if (count < 0)
        {
            std::fprintf(stderr, "bench_tokenizer: llama.cpp could not tokenize a file\n");
            return 1;
        }
        corpus.token_starts.push_back(corpus.tokens.size());
        corpus.tokens.insert(corpus.tokens.end(), out.begin(), out.begin() + count);
    }
    corpus.token_starts.push_back(corpus.tokens.size());

    std::vector<llama_token> llama_out(whole.bytes * 3 + 16);
    std::vector<char> llama_text(whole.bytes * 4 + 64);
    faisca::Array<int32_t> faisca_out;
    faisca::Array<char> faisca_text;
    if (!faisca_out.reserve(whole.bytes + 16) || !faisca_text.reserve(whole.bytes * 4 + 64))
    {
        return 1;
    }

    const auto encode_llama = [&](const Workload& workload, bool parse_special) -> uint64_t
    {
        uint64_t hash = 0;
        for (const std::string_view text : workload.texts)
        {
            const int32_t count = llama_tokenize(reference.vocab(), text.data(), static_cast<int32_t>(text.size()),
                                                 llama_out.data(), static_cast<int32_t>(llama_out.size()), false,
                                                 parse_special);
            if (count < 0)
            {
                return 0;
            }
            hash = hash_tokens(llama_out.data(), static_cast<size_t>(count), hash);
        }
        return hash;
    };

    const auto encode_faisca = [&](const Workload& workload, bool parse_special) -> uint64_t
    {
        uint64_t hash = 0;
        for (const std::string_view text : workload.texts)
        {
            faisca_out.clear();
            if (!tokenizer.encode(text, false, parse_special, faisca_out))
            {
                return 0;
            }
            hash = hash_tokens(faisca_out.data(), faisca_out.size(), hash);
        }
        return hash;
    };

    const auto decode_llama = [&]() -> uint64_t
    {
        uint64_t hash = 0;
        for (size_t f = 0; f + 1 < corpus.token_starts.size(); ++f)
        {
            const size_t start = corpus.token_starts[f];
            const int32_t count = static_cast<int32_t>(corpus.token_starts[f + 1] - start);
            const int32_t written = llama_detokenize(reference.vocab(), corpus.tokens.data() + start, count,
                                                     llama_text.data(), static_cast<int32_t>(llama_text.size()), false,
                                                     true);
            if (written < 0)
            {
                return 0;
            }
            hash = hash_bytes(llama_text.data(), static_cast<size_t>(written), hash);
        }
        return hash;
    };

    const auto decode_faisca = [&]() -> uint64_t
    {
        uint64_t hash = 0;
        for (size_t f = 0; f + 1 < corpus.token_starts.size(); ++f)
        {
            const size_t start = corpus.token_starts[f];
            faisca_text.clear();
            if (!tokenizer.decode(corpus.tokens.data() + start, corpus.token_starts[f + 1] - start, false, true,
                                  faisca_text))
            {
                return 0;
            }
            hash = hash_bytes(faisca_text.data(), faisca_text.size(), hash);
        }
        return hash;
    };

    const auto pieces_llama = [&]() -> uint64_t
    {
        uint64_t hash = 0;
        char piece[512];
        for (const int32_t token : corpus.tokens)
        {
            const int32_t written = llama_token_to_piece(reference.vocab(), token, piece, sizeof(piece), 0, true);
            if (written < 0)
            {
                return 0;
            }
            hash = hash_bytes(piece, static_cast<size_t>(written), hash);
        }
        return hash;
    };

    const auto pieces_faisca = [&]() -> uint64_t
    {
        uint64_t hash = 0;
        for (const int32_t token : corpus.tokens)
        {
            faisca_text.clear();
            if (!tokenizer.token_to_piece(token, true, faisca_text))
            {
                return 0;
            }
            hash = hash_bytes(faisca_text.data(), faisca_text.size(), hash);
        }
        return hash;
    };

    whole.tokens = corpus.tokens.size();
    {
        faisca_out.clear();
        for (const std::string_view text : lines.texts)
        {
            const size_t before = faisca_out.size();
            tokenizer.encode(text, false, true, faisca_out);
            lines.tokens += faisca_out.size() - before;
        }
    }

    struct Measured
    {
        const char* name;
        const char* unit;
        double per;
        Timing llama;
        Timing faisca;
        uint64_t checksum_llama;
        uint64_t checksum_faisca;
    };
    std::vector<Measured> results;

    const auto measure = [&](const char* name, const char* unit, double per, const auto& run_llama,
                             const auto& run_faisca)
    {
        Measured measured{name, unit, per, {}, {}, 0, 0};
        for (int round = -2; round < rounds; ++round)
        {
            auto start = std::chrono::steady_clock::now();
            const uint64_t a = run_llama();
            const double llama_seconds = seconds_since(start);

            start = std::chrono::steady_clock::now();
            const uint64_t b = run_faisca();
            const double faisca_seconds = seconds_since(start);

            if (round < 0)
            {
                measured.checksum_llama = a;
                measured.checksum_faisca = b;
                continue;
            }
            if (a != measured.checksum_llama || b != measured.checksum_faisca)
            {
                std::fprintf(stderr, "bench_tokenizer: %s is not deterministic\n", name);
                std::exit(1);
            }
            measured.llama.samples.push_back(llama_seconds * 1e9 / per);
            measured.faisca.samples.push_back(faisca_seconds * 1e9 / per);
        }
        if (measured.checksum_llama != measured.checksum_faisca || measured.checksum_llama == 0)
        {
            std::fprintf(stderr, "bench_tokenizer: %s: llama.cpp and faisca disagree\n", name);
            std::exit(1);
        }
        results.push_back(std::move(measured));
    };

    measure("encode whole files", "ns/token", static_cast<double>(whole.tokens),
            [&]()
            {
                return encode_llama(whole, true);
            },
            [&]()
            {
                return encode_faisca(whole, true);
            });
    measure("encode whole, no specials", "ns/token", static_cast<double>(whole.tokens),
            [&]()
            {
                return encode_llama(whole, false);
            },
            [&]()
            {
                return encode_faisca(whole, false);
            });
    measure("encode line by line", "ns/token", static_cast<double>(lines.tokens),
            [&]()
            {
                return encode_llama(lines, true);
            },
            [&]()
            {
                return encode_faisca(lines, true);
            });
    measure("decode whole files", "ns/token", static_cast<double>(whole.tokens), decode_llama, decode_faisca);
    measure("token_to_piece", "ns/token", static_cast<double>(corpus.tokens.size()), pieces_llama, pieces_faisca);

    std::printf("\n%" PRIu64 " tokens in the whole files, %" PRIu64 " in the lines, %d rounds, best and median\n",
                whole.tokens, lines.tokens, rounds);
    for (Measured& measured : results)
    {
        print_row(measured.name, measured.llama, measured.faisca, measured.unit);
    }

    const double whole_bytes = static_cast<double>(whole.bytes);
    std::printf("\n  encode throughput: llama.cpp %.1f MB/s, faisca %.1f MB/s\n",
                whole_bytes / (results[0].llama.best() * 1e-9 * static_cast<double>(whole.tokens)) / 1e6,
                whole_bytes / (results[0].faisca.best() * 1e-9 * static_cast<double>(whole.tokens)) / 1e6);

    // vocabulary load, fewer rounds: llama.cpp takes long
    {
        Timing llama_load;
        Timing faisca_load;
        const int load_rounds = std::min(rounds, 10);
        for (int round = -1; round < load_rounds; ++round)
        {
            reference.free();
            auto start = std::chrono::steady_clock::now();
            if (!reference.load(model_path))
            {
                return 1;
            }
            const double llama_seconds = seconds_since(start);

            tokenizer.release();
            file.close();
            start = std::chrono::steady_clock::now();
            if (!file.open(model_path, error) || !tokenizer.load(file, error))
            {
                return 1;
            }
            const double faisca_seconds = seconds_since(start);

            if (round >= 0)
            {
                llama_load.samples.push_back(llama_seconds * 1e3);
                faisca_load.samples.push_back(faisca_seconds * 1e3);
            }
        }
        std::printf("\n");
        print_row("load vocabulary", llama_load, faisca_load, "ms");
    }

    llama_backend_free();
    return 0;
}
