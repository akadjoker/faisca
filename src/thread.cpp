#include "faisca/thread.h"

#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define FAISCA_PAUSE() _mm_pause()
#else
#define FAISCA_PAUSE() ((void)0)
#endif

namespace faisca
{

struct ThreadPool::Worker
{
    ThreadPool* pool;
    int index;
#if defined(_WIN32)
    HANDLE handle;
#else
    pthread_t handle;
#endif
};

namespace
{

#if defined(_WIN32)
DWORD WINAPI worker_entry(LPVOID argument)
{
    ThreadPool::Worker* worker = static_cast<ThreadPool::Worker*>(argument);
    worker->pool->worker_loop(worker->index);
    return 0;
}
#else
void* worker_entry(void* argument)
{
    ThreadPool::Worker* worker = static_cast<ThreadPool::Worker*>(argument);
    worker->pool->worker_loop(worker->index);
    return nullptr;
}
#endif

void yield_briefly()
{
#if defined(_WIN32)
    SwitchToThread();
#else
    sched_yield();
#endif
}

// spins for a while, then yields: generation changes hundreds of times per token
template <typename Condition>
void wait_until(Condition condition)
{
    for (int spins = 0; !condition(); ++spins)
    {
        if (spins < 4096)
        {
            FAISCA_PAUSE();
        }
        else
        {
            yield_briefly();
        }
    }
}

}

bool ThreadPool::start(int count, Error& error)
{
    stop();
    if (count < 1)
    {
        error.set("the thread count must be at least 1");
        return false;
    }
    m_count = count;
    if (count == 1)
    {
        return true;
    }

    m_workers = static_cast<Worker*>(std::calloc(static_cast<size_t>(count - 1), sizeof(Worker)));
    if (m_workers == nullptr)
    {
        error.set("out of memory");
        m_count = 1;
        return false;
    }

    m_stopping.store(false);
    for (int i = 0; i < count - 1; ++i)
    {
        Worker& worker = m_workers[i];
        worker.pool = this;
        worker.index = i + 1;
#if defined(_WIN32)
        worker.handle = CreateThread(nullptr, 0, worker_entry, &worker, 0, nullptr);
        const bool created = worker.handle != nullptr;
#else
        const bool created = pthread_create(&worker.handle, nullptr, worker_entry, &worker) == 0;
#endif
        if (!created)
        {
            error.set("cannot create thread %d of %d", i + 2, count);
            m_count = i + 1;
            stop();
            return false;
        }
    }
    return true;
}

void ThreadPool::stop()
{
    if (m_workers != nullptr)
    {
        m_stopping.store(true);
        m_generation.fetch_add(1);
        for (int i = 0; i < m_count - 1; ++i)
        {
#if defined(_WIN32)
            WaitForSingleObject(m_workers[i].handle, INFINITE);
            CloseHandle(m_workers[i].handle);
#else
            pthread_join(m_workers[i].handle, nullptr);
#endif
        }
        std::free(m_workers);
        m_workers = nullptr;
    }
    m_count = 1;
    m_stopping.store(false);
}

void ThreadPool::run(ThreadTask task, void* user)
{
    if (m_count == 1)
    {
        task(user, 0, 1);
        return;
    }

    m_task = task;
    m_user = user;
    m_pending.store(m_count - 1);
    m_generation.fetch_add(1, std::memory_order_release);

    task(user, 0, m_count);

    wait_until([this]()
    {
        return m_pending.load(std::memory_order_acquire) == 0;
    });
}

void ThreadPool::worker_loop(int index)
{
    unsigned seen = 0;
    for (;;)
    {
        wait_until([this, seen]()
        {
            return m_generation.load(std::memory_order_acquire) != seen;
        });
        seen = m_generation.load(std::memory_order_acquire);
        if (m_stopping.load())
        {
            return;
        }
        m_task(m_user, index, m_count);
        m_pending.fetch_sub(1, std::memory_order_acq_rel);
    }
}

}
