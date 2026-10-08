#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/runner.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// greedy speculative decoding: a small draft model proposes n_draft tokens one at a
// time, the target checks them all in one batched pass and keeps the longest prefix
// it agrees with plus one token of its own. The tokens are exactly the ones greedy
// decoding of the target alone would produce; only the number of target passes changes.
class Speculative
{
public:
    Speculative() = default;

    Speculative(const Speculative&) = delete;
    Speculative& operator=(const Speculative&) = delete;

    // both runners must be initialized and at position 0
    bool init(Runner& target, Runner& draft, size_t n_draft, Error& error);

    // feeds the prompt to both models; first is the target's token after it
    bool prompt(const int32_t* tokens, size_t count, int32_t& first, Error& error);

    // one round after last, a token the target produced but has not consumed yet:
    // appends between 1 and n_draft + 1 tokens to out, the last of them is the next last
    bool round(int32_t last, Array<int32_t>& out, Error& error);

    size_t rounds() const
    {
        return m_rounds;
    }

    size_t drafted() const
    {
        return m_drafted;
    }

    size_t accepted() const
    {
        return m_accepted;
    }

private:
    static int32_t argmax(const float* logits, size_t n);

    Runner* m_target = nullptr;
    Runner* m_draft = nullptr;
    size_t m_n_draft = 0;
    size_t m_rounds = 0;
    size_t m_drafted = 0;
    size_t m_accepted = 0;
    Array<int32_t> m_tokens; // every token the target has consumed, by position
    Array<int32_t> m_batch;
    Array<float> m_logits;
    Array<float> m_draft_logits;
};

}
