#include "faisca/speculative.h"

namespace faisca
{

int32_t Speculative::argmax(const float* logits, size_t n)
{
    size_t best = 0;
    for (size_t t = 1; t < n; ++t)
    {
        if (logits[t] > logits[best])
        {
            best = t;
        }
    }
    return static_cast<int32_t>(best);
}

bool Speculative::init(Runner& target, Runner& draft, size_t n_draft, Error& error)
{
    if (target.model() == nullptr || draft.model() == nullptr)
    {
        error.set("the runners are not initialized");
        return false;
    }
    if (target.model()->hparams().n_vocab != draft.model()->hparams().n_vocab)
    {
        error.set("the draft model has a vocabulary of %u tokens, the target %u", draft.model()->hparams().n_vocab,
                  target.model()->hparams().n_vocab);
        return false;
    }
    if (n_draft == 0)
    {
        error.set("the draft must propose at least one token");
        return false;
    }
    m_target = &target;
    m_draft = &draft;
    m_n_draft = n_draft;
    m_rounds = 0;
    m_drafted = 0;
    m_accepted = 0;
    m_tokens.clear();
    return true;
}

bool Speculative::prompt(const int32_t* tokens, size_t count, int32_t& first, Error& error)
{
    m_target->reset();
    m_draft->reset();
    m_tokens.clear();
    if (!m_tokens.append(tokens, count))
    {
        error.set("out of memory");
        return false;
    }
    if (!m_target->forward(tokens, count, m_logits, error) || !m_draft->forward(tokens, count, m_draft_logits, error))
    {
        return false;
    }
    first = argmax(m_logits.data(), m_logits.size());
    return true;
}

bool Speculative::round(int32_t last, Array<int32_t>& out, Error& error)
{
    const size_t n_vocab = m_target->model()->hparams().n_vocab;
    const uint32_t pos = m_target->position();
    const size_t room_target = m_target->context_size() > pos + 1 ? m_target->context_size() - pos - 1 : 0;
    const size_t room_draft = m_draft->context_size() > pos ? m_draft->context_size() - pos : 0;
    size_t n_draft = m_n_draft < room_target ? m_n_draft : room_target;
    n_draft = n_draft < room_draft ? n_draft : room_draft;
    ++m_rounds;

    if (n_draft == 0)
    {
        if (!m_target->forward(&last, 1, m_logits, error) || !m_tokens.push_back(last))
        {
            return false;
        }
        return out.push_back(argmax(m_logits.data(), n_vocab));
    }

    // the draft catches up with what the target kept, then proposes
    m_batch.clear();
    for (uint32_t p = m_draft->position(); p < pos; ++p)
    {
        if (!m_batch.push_back(m_tokens[p]))
        {
            error.set("out of memory");
            return false;
        }
    }
    if (!m_batch.push_back(last) || !m_draft->forward(m_batch.data(), m_batch.size(), m_draft_logits, error))
    {
        return false;
    }
    m_batch.clear();
    if (!m_batch.push_back(last))
    {
        error.set("out of memory");
        return false;
    }
    for (size_t i = 0; i < n_draft; ++i)
    {
        const int32_t proposed = argmax(m_draft_logits.data(), n_vocab);
        if (!m_batch.push_back(proposed))
        {
            error.set("out of memory");
            return false;
        }
        if (i + 1 < n_draft && !m_draft->forward(&proposed, 1, m_draft_logits, error))
        {
            return false;
        }
    }

    // the target checks last and the n_draft proposals in one pass
    if (!m_target->forward(m_batch.data(), m_batch.size(), m_logits, error, true))
    {
        return false;
    }
    size_t kept = 0;
    while (kept < n_draft && argmax(m_logits.data() + kept * n_vocab, n_vocab) == m_batch[kept + 1])
    {
        ++kept;
    }
    for (size_t i = 0; i <= kept; ++i)
    {
        if (!out.push_back(i < kept ? m_batch[i + 1] : argmax(m_logits.data() + kept * n_vocab, n_vocab)) ||
            !m_tokens.push_back(m_batch[i]))
        {
            error.set("out of memory");
            return false;
        }
    }
    m_drafted += n_draft;
    m_accepted += kept;
    const uint32_t next = pos + 1 + static_cast<uint32_t>(kept);
    m_target->truncate(next);
    m_draft->truncate(next);
    return true;
}

}
