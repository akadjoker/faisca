#include "faisca/mapped_file.h"

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <limits>

#else

#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

namespace faisca
{

MappedFile::~MappedFile()
{
    close();
}

#if defined(_WIN32)

namespace
{

constexpr int MAX_WIDE_PATH = 32768;

struct ErrorText
{
    char text[256];
};

ErrorText last_error_text(DWORD code)
{
    ErrorText out;
    DWORD length = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0,
                                  out.text, static_cast<DWORD>(sizeof(out.text)), nullptr);

    while (length > 0 && (out.text[length - 1] == '\r' || out.text[length - 1] == '\n' ||
                          out.text[length - 1] == ' ' || out.text[length - 1] == '.'))
    {
        --length;
    }
    out.text[length] = '\0';

    if (length == 0)
    {
        std::snprintf(out.text, sizeof(out.text), "Windows error %lu", static_cast<unsigned long>(code));
    }
    return out;
}

bool to_wide_path(const char* path, wchar_t* out, int capacity)
{
    const size_t size = std::strlen(path);
    if (size == 0 || size >= static_cast<size_t>(capacity))
    {
        return false;
    }
    const int length = static_cast<int>(size);

    int written = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, length, out, capacity - 1);
    if (written <= 0)
    {
        // argv is not UTF-8 on Windows
        written = MultiByteToWideChar(CP_ACP, 0, path, length, out, capacity - 1);
    }
    if (written <= 0)
    {
        return false;
    }
    out[written] = L'\0';
    return true;
}

}

bool MappedFile::open(const char* path, Error& error)
{
    close();

    if (path == nullptr)
    {
        error.set("cannot open a null path");
        return false;
    }

    wchar_t wide_path[MAX_WIDE_PATH];
    if (!to_wide_path(path, wide_path, MAX_WIDE_PATH))
    {
        error.set("cannot open '%s': invalid path", path);
        return false;
    }

    const HANDLE file = CreateFileW(wide_path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        error.set("cannot open '%s': %s", path, last_error_text(GetLastError()).text);
        return false;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size))
    {
        error.set("cannot stat '%s': %s", path, last_error_text(GetLastError()).text);
        CloseHandle(file);
        return false;
    }
    if (file_size.QuadPart <= 0)
    {
        error.set("'%s' is empty", path);
        CloseHandle(file);
        return false;
    }
    if (static_cast<unsigned long long>(file_size.QuadPart) > std::numeric_limits<size_t>::max())
    {
        error.set("'%s' is too large to map in this process", path);
        CloseHandle(file);
        return false;
    }

    const HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr)
    {
        error.set("cannot map '%s': %s", path, last_error_text(GetLastError()).text);
        CloseHandle(file);
        return false;
    }

    void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    const DWORD view_error = GetLastError();

    // the view keeps the file mapped
    CloseHandle(mapping);
    CloseHandle(file);

    if (view == nullptr)
    {
        error.set("cannot map '%s': %s", path, last_error_text(view_error).text);
        return false;
    }

    m_data = static_cast<const uint8_t*>(view);
    m_size = static_cast<size_t>(file_size.QuadPart);
    return true;
}

void MappedFile::close()
{
    if (m_data != nullptr)
    {
        UnmapViewOfFile(m_data);
    }
    m_data = nullptr;
    m_size = 0;
}

#else

bool MappedFile::open(const char* path, Error& error)
{
    close();

    if (path == nullptr)
    {
        error.set("cannot open a null path");
        return false;
    }

    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        error.set("cannot open '%s': %s", path, std::strerror(errno));
        return false;
    }

    struct stat st;
    if (::fstat(fd, &st) != 0)
    {
        error.set("cannot stat '%s': %s", path, std::strerror(errno));
        ::close(fd);
        return false;
    }

    if (!S_ISREG(st.st_mode))
    {
        error.set("cannot open '%s': not a regular file", path);
        ::close(fd);
        return false;
    }

    if (st.st_size <= 0)
    {
        error.set("'%s' is empty", path);
        ::close(fd);
        return false;
    }

    const size_t size = static_cast<size_t>(st.st_size);
    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int mmap_errno = errno;

    ::close(fd);

    if (addr == MAP_FAILED)
    {
        error.set("cannot map '%s': %s", path, std::strerror(mmap_errno));
        return false;
    }

    m_data = static_cast<const uint8_t*>(addr);
    m_size = size;
    return true;
}

void MappedFile::close()
{
    if (m_data != nullptr)
    {
        ::munmap(const_cast<uint8_t*>(m_data), m_size);
    }
    m_data = nullptr;
    m_size = 0;
}

#endif

}
