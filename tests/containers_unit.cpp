#include "faisca/array.h"
#include "faisca/error.h"
#include "faisca/string_map.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace faisca;

namespace
{

int g_checks = 0;
int g_failures = 0;

#define CHECK(condition)                                                              \
    do                                                                                \
    {                                                                                 \
        ++g_checks;                                                                   \
        if (!(condition))                                                             \
        {                                                                             \
            ++g_failures;                                                             \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        }                                                                             \
    } while (false)

struct Record
{
    uint64_t id;
    double weight;
    char tag[12];
};

void test_array_basics()
{
    Array<int> numbers;
    CHECK(numbers.empty());
    CHECK(numbers.size() == 0);
    CHECK(numbers.capacity() == 0);
    CHECK(numbers.data() == nullptr);
    CHECK(numbers.begin() == numbers.end());

    for (int i = 0; i < 1000; ++i)
    {
        CHECK(numbers.push_back(i * 3));
    }
    CHECK(numbers.size() == 1000);
    CHECK(numbers.capacity() >= 1000);

    bool in_order = true;
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        in_order = in_order && numbers[i] == static_cast<int>(i) * 3;
    }
    CHECK(in_order);

    long long sum = 0;
    for (const int value : numbers)
    {
        sum += value;
    }
    CHECK(sum == 3LL * 999 * 1000 / 2);

    numbers[10] = -7;
    CHECK(numbers[10] == -7);

    const size_t capacity = numbers.capacity();
    numbers.clear();
    CHECK(numbers.empty());
    CHECK(numbers.capacity() == capacity);

    numbers.release();
    CHECK(numbers.capacity() == 0);
    CHECK(numbers.data() == nullptr);
    CHECK(numbers.push_back(5));
    CHECK(numbers.size() == 1 && numbers[0] == 5);
}

void test_array_reserve_and_resize()
{
    Array<Record> records;
    CHECK(records.reserve(100));
    CHECK(records.capacity() == 100);
    CHECK(records.size() == 0);

    const Record* block = records.data();
    for (uint64_t i = 0; i < 100; ++i)
    {
        Record record;
        record.id = i;
        record.weight = static_cast<double>(i) * 0.5;
        std::snprintf(record.tag, sizeof(record.tag), "r%u", static_cast<unsigned>(i));
        records.push_back_unchecked(record);
    }
    CHECK(records.data() == block);
    CHECK(records.size() == 100);
    CHECK(records[99].id == 99 && records[99].weight == 49.5 && std::strcmp(records[99].tag, "r99") == 0);

    CHECK(records.reserve(10));
    CHECK(records.capacity() == 100);

    Array<uint8_t> bytes;
    CHECK(bytes.resize_uninitialized(4096));
    CHECK(bytes.size() == 4096);
    std::memset(bytes.data(), 0xab, bytes.size());
    CHECK(bytes[0] == 0xab && bytes[4095] == 0xab);
    CHECK(bytes.resize_uninitialized(16));
    CHECK(bytes.size() == 16 && bytes[15] == 0xab);
    CHECK(bytes.resize_uninitialized(0));
    CHECK(bytes.empty());
}

void test_array_self_reference()
{
    Array<Record> records;
    Record first;
    first.id = 42;
    first.weight = 1.25;
    std::snprintf(first.tag, sizeof(first.tag), "first");
    CHECK(records.push_back(first));

    for (int i = 0; i < 2000; ++i)
    {
        CHECK(records.push_back(records[0]));
    }
    bool all_equal = true;
    for (const Record& record : records)
    {
        all_equal = all_equal && record.id == 42 && record.weight == 1.25 && std::strcmp(record.tag, "first") == 0;
    }
    CHECK(all_equal);
    CHECK(records.size() == 2001);
}

void test_array_limits()
{
    Array<Record> records;
    CHECK(!records.reserve(static_cast<size_t>(-1)));
    CHECK(!records.reserve(static_cast<size_t>(-1) / sizeof(Record) + 1));
    CHECK(!records.resize_uninitialized(static_cast<size_t>(-1) / 2));
    CHECK(records.size() == 0 && records.capacity() == 0);
    CHECK(records.push_back(Record()));
}

void test_partial_loads()
{
    const unsigned char source[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    for (size_t size = 0; size < 8; ++size)
    {
        char* exact = static_cast<char*>(std::malloc(size == 0 ? 1 : size));
        std::memcpy(exact, source, size);

        uint64_t expected = 0;
        std::memcpy(&expected, source, size);
        CHECK(detail::load_partial(exact, size) == expected);

        std::free(exact);
    }
}

void test_hash()
{
    const char text[] = "blk.10.attn_output.weight";
    for (size_t size = 0; size <= sizeof(text) - 1; ++size)
    {
        char* copy = static_cast<char*>(std::malloc(size == 0 ? 1 : size));
        std::memcpy(copy, text, size);
        CHECK(hash_bytes(text, size) == hash_bytes(copy, size));
        if (size > 0)
        {
            CHECK(hash_bytes(text, size) != hash_bytes(text, size - 1));
        }
        std::free(copy);
    }

    const char zeros[16] = {};
    CHECK(hash_bytes(zeros, 1) != hash_bytes(zeros, 2));
    CHECK(hash_bytes(zeros, 8) != hash_bytes(zeros, 9));
    CHECK(hash_bytes(zeros, 15) != hash_bytes(zeros, 16));

    bool every_byte_counts = true;
    for (size_t size = 1; size <= 40; ++size)
    {
        char buffer[40];
        for (size_t i = 0; i < size; ++i)
        {
            buffer[i] = static_cast<char>('a' + i % 26);
        }
        const uint64_t original = hash_bytes(buffer, size);
        for (size_t i = 0; i < size; ++i)
        {
            buffer[i] = static_cast<char>(buffer[i] ^ 0x01);
            every_byte_counts = every_byte_counts && hash_bytes(buffer, size) != original;
            buffer[i] = static_cast<char>(buffer[i] ^ 0x01);
        }
    }
    CHECK(every_byte_counts);
}

uint64_t unmix_word(uint64_t mixed)
{
    uint64_t inverse = detail::HASH_MULTIPLIER;
    for (int i = 0; i < 6; ++i)
    {
        inverse *= 2 - detail::HASH_MULTIPLIER * inverse;
    }

    uint64_t word = mixed * inverse;
    word ^= word >> 47;
    word *= inverse;
    return word;
}

uint64_t mix_word(uint64_t word)
{
    word *= detail::HASH_MULTIPLIER;
    word ^= word >> 47;
    word *= detail::HASH_MULTIPLIER;
    return word;
}

// two different keys built to have the same 64-bit hash
void test_map_hash_collision()
{
    const size_t size = 24;
    const uint64_t start = detail::HASH_SEED ^ (static_cast<uint64_t>(size) * detail::HASH_MULTIPLIER);

    uint64_t first[3] = {0x1111111111111111ull, 0x2222222222222222ull, 0x3333333333333333ull};
    uint64_t second[3] = {0x4444444444444444ull, 0, 0x3333333333333333ull};

    const uint64_t after_first = (start ^ mix_word(first[0])) * detail::HASH_MULTIPLIER;
    const uint64_t after_second = (start ^ mix_word(second[0])) * detail::HASH_MULTIPLIER;
    second[1] = unmix_word(after_first ^ mix_word(first[1]) ^ after_second);

    char key_a[24];
    char key_b[24];
    std::memcpy(key_a, first, size);
    std::memcpy(key_b, second, size);

    CHECK(std::memcmp(key_a, key_b, size) != 0);
    CHECK(hash_bytes(key_a, size) == hash_bytes(key_b, size));

    StringMap map;
    CHECK(map.insert(std::string_view(key_a, size), 1) == StringMap::Insert::Added);
    CHECK(map.insert(std::string_view(key_b, size), 2) == StringMap::Insert::Added);
    CHECK(map.size() == 2);

    uint32_t value = 0;
    CHECK(map.find(std::string_view(key_a, size), value) && value == 1);
    CHECK(map.find(std::string_view(key_b, size), value) && value == 2);
}

void test_map_basics()
{
    StringMap map;
    uint32_t value = 123;

    CHECK(map.size() == 0);
    CHECK(!map.find("anything", value));
    CHECK(value == 123);

    CHECK(map.insert("general.name", 1) == StringMap::Insert::Added);
    CHECK(map.insert("general.architecture", 2) == StringMap::Insert::Added);
    CHECK(map.insert("", 3) == StringMap::Insert::Added);
    CHECK(map.size() == 3);

    CHECK(map.insert("general.name", 99) == StringMap::Insert::Exists);
    CHECK(map.find("general.name", value) && value == 1);
    CHECK(map.find("general.architecture", value) && value == 2);
    CHECK(map.find("", value) && value == 3);
    CHECK(map.size() == 3);

    CHECK(!map.find("general.nam", value));
    CHECK(!map.find("general.name ", value));
    CHECK(!map.find("General.name", value));

    const char raw[20] = {'a', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    StringMap zeros;
    for (uint32_t size = 1; size <= 20; ++size)
    {
        CHECK(zeros.insert(std::string_view(raw, size), size) == StringMap::Insert::Added);
    }
    bool all_found = true;
    for (uint32_t size = 1; size <= 20; ++size)
    {
        uint32_t found = 0;
        all_found = all_found && zeros.find(std::string_view(raw, size), found) && found == size;
    }
    CHECK(all_found);
    CHECK(zeros.size() == 20);

    map.release();
    CHECK(map.size() == 0);
    CHECK(!map.find("general.name", value));
    CHECK(map.insert("again", 7) == StringMap::Insert::Added);
    CHECK(map.find("again", value) && value == 7);
}

void test_map_long_keys_compare_bytes()
{
    std::string first = "blk.0.attn_output.weight.first";
    std::string second = "blk.0.attn_output.weight.secon";
    std::string first_copy = first;

    StringMap map;
    CHECK(map.insert(first, 10) == StringMap::Insert::Added);
    CHECK(map.insert(second, 20) == StringMap::Insert::Added);
    CHECK(map.insert(first_copy, 30) == StringMap::Insert::Exists);

    uint32_t value = 0;
    CHECK(map.find(first_copy, value) && value == 10);
    CHECK(map.find(second, value) && value == 20);
}

struct ExactKey
{
    char* data;
    size_t size;

    std::string_view view() const
    {
        return std::string_view(data, size);
    }
};

void test_map_against_std(uint64_t seed, size_t key_count, size_t alphabet, size_t max_length)
{
    std::mt19937_64 rng(seed);

    std::vector<ExactKey> keys;
    keys.reserve(key_count);
    for (size_t i = 0; i < key_count; ++i)
    {
        ExactKey key;
        key.size = static_cast<size_t>(rng() % (max_length + 1));
        key.data = static_cast<char*>(std::malloc(key.size == 0 ? 1 : key.size));
        for (size_t b = 0; b < key.size; ++b)
        {
            key.data[b] = static_cast<char>(rng() % alphabet);
        }
        keys.push_back(key);
    }

    StringMap map;
    std::unordered_map<std::string, uint32_t> expected;
    bool inserts_agree = true;

    for (size_t i = 0; i < keys.size(); ++i)
    {
        const std::string owned(keys[i].data, keys[i].size);
        const bool is_new = expected.emplace(owned, static_cast<uint32_t>(i)).second;
        const StringMap::Insert result = map.insert(keys[i].view(), static_cast<uint32_t>(i));
        inserts_agree = inserts_agree && result == (is_new ? StringMap::Insert::Added : StringMap::Insert::Exists);
    }
    CHECK(inserts_agree);
    CHECK(map.size() == expected.size());

    bool hits_agree = true;
    for (const ExactKey& key : keys)
    {
        uint32_t value = 0;
        const auto it = expected.find(std::string(key.data, key.size));
        hits_agree = hits_agree && map.find(key.view(), value) && value == it->second;
    }
    CHECK(hits_agree);

    bool probes_agree = true;
    for (size_t i = 0; i < key_count * 2; ++i)
    {
        ExactKey probe;
        probe.size = static_cast<size_t>(rng() % (max_length + 2));
        probe.data = static_cast<char*>(std::malloc(probe.size == 0 ? 1 : probe.size));
        for (size_t b = 0; b < probe.size; ++b)
        {
            probe.data[b] = static_cast<char>(rng() % alphabet);
        }

        uint32_t value = 0;
        const bool found = map.find(probe.view(), value);
        const auto it = expected.find(std::string(probe.data, probe.size));
        probes_agree = probes_agree && found == (it != expected.end()) && (!found || value == it->second);

        std::free(probe.data);
    }
    CHECK(probes_agree);

    for (const ExactKey& key : keys)
    {
        std::free(key.data);
    }
}

void test_map_shared_halves()
{
    const uint32_t count = 20000;
    std::vector<char> storage(static_cast<size_t>(count) * 12 * 2);

    StringMap map;
    bool all_added = true;
    for (uint32_t i = 0; i < count; ++i)
    {
        char* same_start = storage.data() + static_cast<size_t>(i) * 24;
        std::memcpy(same_start, "AAAAAAAA", 8);
        std::memcpy(same_start + 8, &i, 4);

        char* same_end = same_start + 12;
        std::memcpy(same_end, &i, 4);
        std::memcpy(same_end + 4, "BBBBBBBB", 8);

        all_added = all_added && map.insert(std::string_view(same_start, 12), i) == StringMap::Insert::Added;
        all_added = all_added && map.insert(std::string_view(same_end, 12), i + count) == StringMap::Insert::Added;
    }
    CHECK(all_added);
    CHECK(map.size() == static_cast<size_t>(count) * 2);

    bool all_found = true;
    for (uint32_t i = 0; i < count; ++i)
    {
        const char* same_start = storage.data() + static_cast<size_t>(i) * 24;
        uint32_t value = 0;
        all_found = all_found && map.find(std::string_view(same_start, 12), value) && value == i;
        all_found = all_found && map.find(std::string_view(same_start + 12, 12), value) && value == i + count;
    }
    CHECK(all_found);
}

void test_map_reserve()
{
    StringMap map;
    CHECK(map.reserve(1000));
    CHECK(!map.reserve(static_cast<size_t>(-1)));
    uint32_t value = 0;
    CHECK(!map.find("x", value));
    CHECK(map.insert("x", 1) == StringMap::Insert::Added);
    CHECK(map.find("x", value) && value == 1);
}

void test_error()
{
    Error error;
    CHECK(error.empty());
    CHECK(std::strcmp(error.c_str(), "") == 0);

    error.set("tensor '%s': %d dimensions", "weights", 5);
    CHECK(!error.empty());
    CHECK(std::strcmp(error.c_str(), "tensor 'weights': 5 dimensions") == 0);

    error.set("replaced");
    CHECK(std::strcmp(error.c_str(), "replaced") == 0);

    error.clear();
    CHECK(error.empty());

    std::string huge(5000, 'x');
    error.set("path %s", huge.c_str());
    const size_t length = std::strlen(error.c_str());
    CHECK(length == Error::CAPACITY - 1);
    CHECK(std::strcmp(error.c_str() + length - 3, "...") == 0);
    CHECK(std::strncmp(error.c_str(), "path xxx", 8) == 0);
}

}

int main()
{
    test_array_basics();
    test_array_reserve_and_resize();
    test_array_self_reference();
    test_array_limits();
    test_partial_loads();
    test_hash();
    test_map_basics();
    test_map_long_keys_compare_bytes();
    test_map_hash_collision();
    test_map_shared_halves();
    test_map_reserve();
    test_error();

    test_map_against_std(1, 20000, 3, 6);
    test_map_against_std(2, 20000, 4, 16);
    test_map_against_std(3, 20000, 2, 40);
    test_map_against_std(4, 50000, 255, 24);
    test_map_against_std(5, 2000, 1, 64);
    for (uint64_t seed = 100; seed < 140; ++seed)
    {
        test_map_against_std(seed, 300 + static_cast<size_t>(seed), 5, 20);
    }

    std::printf("%s containers_unit: %d checks, %d failed\n", g_failures == 0 ? "ok  " : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
