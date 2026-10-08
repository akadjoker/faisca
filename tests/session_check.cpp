#include "faisca/gguf.h"
#include "faisca/model.h"
#include "faisca/runner.h"
#include "faisca/session.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// usage: session_check <model.gguf> <session file to write> <token ids...>
//
// Prefix reuse, snapshots, and a session saved to disk and loaded in a fresh
// runner must continue a generation exactly as an uninterrupted run does: the
// logits of every step are compared bit for bit.

namespace
{

int g_failures = 0;

void check(bool condition, const char* what)
{
    std::printf("  %s %s\n", condition ? "ok  " : "FAIL", what);
    g_failures += condition ? 0 : 1;
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

bool same(const faisca::Array<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<float> copy_of(const faisca::Array<float>& a)
{
    return std::vector<float>(a.data(), a.data() + a.size());
}

// greedy tokens and the logits of every step after feeding tokens into a fresh runner
struct Run
{
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> logits;
};

bool run_fresh(faisca::Runner& runner, const std::vector<int32_t>& prompt, size_t generate, Run& out, faisca::Error& error)
{
    runner.reset();
    out.tokens.clear();
    out.logits.clear();
    faisca::Array<float> logits;
    if (!runner.forward(prompt.data(), prompt.size(), logits, error))
    {
        return false;
    }
    for (size_t g = 0; g < generate; ++g)
    {
        out.logits.push_back(copy_of(logits));
        const int32_t best = argmax(logits);
        out.tokens.push_back(best);
        if (g + 1 < generate && !runner.forward(&best, 1, logits, error))
        {
            return false;
        }
    }
    return true;
}

}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: session_check <model.gguf> <session file> <token ids...>\n");
        return 2;
    }
    std::vector<int32_t> prompt;
    for (int i = 3; i < argc; ++i)
    {
        prompt.push_back(std::atoi(argv[i]));
    }
    const size_t generate = 8;

    faisca::Error error;
    faisca::GgufFile file;
    faisca::LlamaModel model;
    faisca::ThreadPool pool;
    faisca::Runner reference;
    faisca::Runner runner;
    faisca::Runner fresh;
    const uint32_t n_ctx = static_cast<uint32_t>(2 * prompt.size() + 4 * generate + 8);
    if (!file.open(argv[1], error) || !model.load(file, error) || !pool.start(2, error) ||
        !reference.init(model, n_ctx, pool, error) || !runner.init(model, n_ctx, pool, error) ||
        !fresh.init(model, n_ctx, pool, error))
    {
        std::fprintf(stderr, "session_check: %s\n", error.c_str());
        return 1;
    }
    std::string_view name;
    file.get_string("general.name", name);
    const std::string model_name(name);

    // 1. a session fed with the prompt generates what a plain runner generates
    Run plain;
    if (!run_fresh(reference, prompt, generate, plain, error))
    {
        std::fprintf(stderr, "session_check: %s\n", error.c_str());
        return 1;
    }
    faisca::Session session;
    faisca::Array<float> logits;
    size_t reused = 0;
    bool ok = session.init(runner, model_name.c_str(), error) && session.feed(prompt.data(), prompt.size(), logits, error, &reused);
    std::vector<int32_t> generated;
    for (size_t g = 0; ok && g < generate; ++g)
    {
        ok = same(logits, plain.logits[g]);
        const int32_t best = argmax(logits);
        generated.push_back(best);
        if (g + 1 < generate)
        {
            ok = ok && session.push(best, logits, error);
        }
    }
    check(ok && generated == plain.tokens && reused == 0, "session equals a plain run from an empty cache");

    // 2. the same prompt again keeps everything but the last token
    ok = session.feed(prompt.data(), prompt.size(), logits, error, &reused);
    check(ok && reused == prompt.size() - 1 && same(logits, plain.logits[0]), "feeding the prompt again reuses all but its last token");

    // 3. a longer prompt sharing the prefix: only the new tokens run, the result matches a fresh run
    std::vector<int32_t> longer = prompt;
    for (int32_t t : plain.tokens)
    {
        longer.push_back(t);
    }
    longer.push_back(prompt[1]);
    longer.push_back(prompt[2]);
    Run plain_longer;
    if (!run_fresh(reference, longer, generate, plain_longer, error))
    {
        std::fprintf(stderr, "session_check: %s\n", error.c_str());
        return 1;
    }
    ok = session.feed(longer.data(), longer.size(), logits, error, &reused);
    check(ok && reused == prompt.size(), "a longer prompt reuses the cached prefix");
    check(ok && same(logits, plain_longer.logits[0]), "its logits equal a fresh run of the longer prompt");

    // 4. a prompt that diverges in the middle: the cache is cut at the divergence
    std::vector<int32_t> diverged = longer;
    diverged[prompt.size() + 2] = prompt[0];
    Run plain_diverged;
    if (!run_fresh(reference, diverged, generate, plain_diverged, error))
    {
        std::fprintf(stderr, "session_check: %s\n", error.c_str());
        return 1;
    }
    ok = session.feed(diverged.data(), diverged.size(), logits, error, &reused);
    check(ok && reused == prompt.size() + 2 && same(logits, plain_diverged.logits[0]), "a prompt that diverges is cut at the divergence");

    // 5. snapshot in memory, continue, restore, continue again: identical
    ok = session.feed(prompt.data(), prompt.size(), logits, error);
    for (size_t g = 0; ok && g < 3; ++g)
    {
        ok = session.push(argmax(logits), logits, error);
    }
    faisca::Array<uint8_t> blob;
    ok = ok && session.snapshot(blob, error);
    std::vector<std::vector<float>> after_snapshot;
    const std::vector<float> at_snapshot = copy_of(logits);
    for (size_t g = 0; ok && g < 4; ++g)
    {
        ok = session.push(argmax(logits), logits, error);
        after_snapshot.push_back(copy_of(logits));
    }
    check(ok, "snapshot taken and generation continued");
    ok = session.restore(blob.data(), blob.size(), error);
    check(ok && session.position() == prompt.size() + 3 && session.tokens().size() == prompt.size() + 3, "restore brings the position back");
    bool identical = ok;
    logits.clear();
    logits.append(at_snapshot.data(), at_snapshot.size());
    for (size_t g = 0; identical && g < 4; ++g)
    {
        identical = session.push(argmax(logits), logits, error) && same(logits, after_snapshot[g]);
    }
    check(identical, "generation after restore is bit identical");

    // 6. the same through a file, into a fresh runner
    faisca::Session loaded;
    ok = session.restore(blob.data(), blob.size(), error) && session.save(argv[2], error) &&
         loaded.init(fresh, model_name.c_str(), error) && loaded.load(argv[2], error);
    check(ok && loaded.position() == prompt.size() + 3, "session saved and loaded in a fresh runner");
    identical = ok;
    logits.clear();
    logits.append(at_snapshot.data(), at_snapshot.size());
    for (size_t g = 0; identical && g < 4; ++g)
    {
        identical = loaded.push(argmax(logits), logits, error) && same(logits, after_snapshot[g]);
    }
    check(identical, "generation after loading the file is bit identical");

    // 7. a wrong model name is refused
    faisca::Session other;
    const bool refused = other.init(fresh, "another model", error) && !other.restore(blob.data(), blob.size(), error);
    check(refused, "a snapshot of another model is refused");
    if (refused)
    {
        std::printf("       (%s)\n", error.c_str());
    }

    std::printf("%s session_check: %d failures, snapshot of %zu tokens is %zu bytes\n", g_failures == 0 ? "ok  " : "FAIL",
                g_failures, prompt.size() + 3, blob.size());
    return g_failures == 0 ? 0 : 1;
}
