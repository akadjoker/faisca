#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/runner.h"
#include "faisca/speculative.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// usage: speculative_check <target.gguf> <draft.gguf> <threads> <draft tokens> <generate N> <token ids...>
//
// Generates N tokens greedily with the target alone and then with the draft, and
// checks that the two sequences are identical; prints the acceptance rate and the
// speed of both.

namespace
{

double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int32_t argmax(const faisca::Array<float>& logits)
{
    int32_t best = 0;
    for (size_t t = 1; t < logits.size(); ++t)
    {
        if (logits[t] > logits[static_cast<size_t>(best)])
        {
            best = static_cast<int32_t>(t);
        }
    }
    return best;
}

}

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::fprintf(stderr, "usage: speculative_check <target.gguf> <draft.gguf> <threads> <draft tokens> <generate N> <token ids...>\n");
        return 2;
    }
    const int threads = std::atoi(argv[3]);
    const size_t n_draft = static_cast<size_t>(std::atoi(argv[4]));
    const size_t generate = static_cast<size_t>(std::atoi(argv[5]));
    std::vector<int32_t> prompt;
    for (int i = 6; i < argc; ++i)
    {
        prompt.push_back(std::atoi(argv[i]));
    }

    faisca::Error error;
    faisca::GgufFile target_file;
    faisca::GgufFile draft_file;
    faisca::LlamaModel target;
    faisca::LlamaModel draft;
    faisca::ThreadPool pool;
    faisca::Runner target_runner;
    faisca::Runner draft_runner;
    const uint32_t n_ctx = static_cast<uint32_t>(prompt.size() + generate + n_draft + 2);
    if (!target_file.open(argv[1], error) || !target.load(target_file, error) ||
        !draft_file.open(argv[2], error) || !draft.load(draft_file, error) || !pool.start(threads, error) ||
        !target_runner.init(target, n_ctx, pool, error) || !draft_runner.init(draft, n_ctx, pool, error))
    {
        std::fprintf(stderr, "speculative_check: %s\n", error.c_str());
        return 1;
    }

    std::vector<int32_t> plain;
    faisca::Array<float> logits;
    const double t0 = now();
    if (!target_runner.forward(prompt.data(), prompt.size(), logits, error))
    {
        std::fprintf(stderr, "speculative_check: %s\n", error.c_str());
        return 1;
    }
    const double t1 = now();
    while (plain.size() < generate)
    {
        const int32_t best = argmax(logits);
        plain.push_back(best);
        if (plain.size() < generate && !target_runner.forward(&best, 1, logits, error))
        {
            std::fprintf(stderr, "speculative_check: %s\n", error.c_str());
            return 1;
        }
    }
    const double t2 = now();

    faisca::Speculative speculative;
    std::vector<int32_t> fast;
    faisca::Array<int32_t> round;
    int32_t last = 0;
    const double t3 = now();
    if (!speculative.init(target_runner, draft_runner, n_draft, error) ||
        !speculative.prompt(prompt.data(), prompt.size(), last, error))
    {
        std::fprintf(stderr, "speculative_check: %s\n", error.c_str());
        return 1;
    }
    const double t4 = now();
    fast.push_back(last);
    while (fast.size() < generate)
    {
        round.clear();
        if (!speculative.round(last, round, error))
        {
            std::fprintf(stderr, "speculative_check: %s\n", error.c_str());
            return 1;
        }
        for (size_t i = 0; i < round.size(); ++i)
        {
            fast.push_back(round[i]);
        }
        last = round[round.size() - 1];
    }
    const double t5 = now();

    bool same = true;
    size_t first_diff = generate;
    for (size_t i = 0; i < generate; ++i)
    {
        if (plain[i] != fast[i])
        {
            same = false;
            first_diff = i;
            break;
        }
    }
    const double plain_rate = static_cast<double>(generate - 1) / (t2 - t1);
    const double fast_rate = static_cast<double>(generate - 1) / (t5 - t4);
    std::printf("%s %zu tokens after a prompt of %zu, %d threads, draft %zu: plain %.1f tok/s, speculative %.1f tok/s (%.2fx), "
                "%zu rounds, %zu proposed, %zu accepted (%.0f%%)%s\n",
                same ? "ok  " : "FAIL", generate, prompt.size(), threads, n_draft, plain_rate, fast_rate, fast_rate / plain_rate,
                speculative.rounds(), speculative.drafted(), speculative.accepted(),
                speculative.drafted() > 0 ? 100.0 * static_cast<double>(speculative.accepted()) / static_cast<double>(speculative.drafted()) : 0.0,
                same ? "" : "  SEQUENCES DIFFER");
    if (!same)
    {
        std::printf("  first difference at token %zu: plain %d, speculative %d\n", first_diff, plain[first_diff], fast[first_diff]);
    }
    (void)t0;
    (void)t3;
    return same ? 0 : 1;
}
