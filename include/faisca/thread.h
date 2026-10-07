#pragma once

#include "faisca/error.h"

#include <atomic>
#include <cstddef>

namespace faisca
{

typedef void (*ThreadTask)(void* user, int thread_index, int thread_count);

// a fixed set of threads that all run the same task; the caller is thread 0
class ThreadPool
{
public:
    struct Worker;

    ThreadPool() = default;

    ~ThreadPool()
    {
        stop();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // starts count - 1 workers; count 1 runs everything on the caller
    bool start(int count, Error& error);

    void stop();

    int size() const
    {
        return m_count;
    }

    // runs task on every thread and returns when all have finished
    void run(ThreadTask task, void* user);

    // workers only: the loop a worker thread lives in
    void worker_loop(int index);

private:
    int m_count = 1;
    Worker* m_workers = nullptr;
    ThreadTask m_task = nullptr;
    void* m_user = nullptr;
    std::atomic<unsigned> m_generation{0};
    std::atomic<int> m_pending{0};
    std::atomic<bool> m_stopping{false};
};

}
