#pragma once

#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/runner.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// A conversation over a Runner: the tokens the cache holds, by position. feed keeps
// the longest prefix the new tokens share with them and forwards only the rest;
// snapshots serialize tokens and cache together, and a session file is a snapshot
// with a header and an offset table (see src/session.cpp).
class Session
{
public:
    static constexpr uint32_t FILE_VERSION = 1;

    Session() = default;

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // the runner must be initialized; the session starts empty
    bool init(Runner& runner, const char* model_name, Error& error);

    const Array<int32_t>& tokens() const
    {
        return m_tokens;
    }

    uint32_t position() const
    {
        return m_runner == nullptr ? 0 : m_runner->position();
    }

    // makes the cache hold exactly these tokens and gives the logits after the last one;
    // reused tells how many were already there (at most count - 1, the last is always run)
    bool feed(const int32_t* tokens, size_t count, Array<float>& logits, Error& error, size_t* reused = nullptr);

    // one more token, as generation produces them
    bool push(int32_t token, Array<float>& logits, Error& error);

    // drops every token from count on
    bool truncate(size_t count, Error& error);

    bool snapshot(Array<uint8_t>& out, Error& error) const;
    bool restore(const uint8_t* data, size_t bytes, Error& error);
    bool save(const char* path, Error& error) const;
    bool load(const char* path, Error& error);

private:
    size_t snapshot_bytes() const;

    Runner* m_runner = nullptr;
    Array<int32_t> m_tokens;
    char m_model_name[128] = {};
};

}
