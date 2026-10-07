#include "faisca/memory.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace faisca
{

namespace
{

constexpr size_t HUGE_PAGE = 2u << 20;

size_t round_up(size_t bytes)
{
    return (bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE;
}

}

void* allocate_large(size_t bytes)
{
    if (bytes == 0)
    {
        return nullptr;
    }
    const size_t size = round_up(bytes);
#if defined(_WIN32)
    return VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* block = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (block == MAP_FAILED)
    {
        return nullptr;
    }
#if defined(MADV_HUGEPAGE)
    madvise(block, size, MADV_HUGEPAGE);
#endif
    return block;
#endif
}

void free_large(void* block, size_t bytes)
{
    if (block == nullptr)
    {
        return;
    }
#if defined(_WIN32)
    (void)bytes;
    VirtualFree(block, 0, MEM_RELEASE);
#else
    munmap(block, round_up(bytes));
#endif
}

}
