#include "faisca/array.h"
#include "faisca/gguf.h"
#include "faisca/string_map.h"

#include <ct/hashmap.hpp>
#include <ct/span.hpp>
#include <ct/vector.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace
{

using faisca::GgufTensor;

struct Candidate
{
    const char* name;
    std::function<uint64_t()> run;
};

struct MurmurForCt
{
    std::uint64_t operator()(ct::StringView text) const noexcept
    {
        return faisca::hash_bytes(text.data(), text.size());
    }
};

int g_rounds = 15;

double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void compare(const char* title, uint64_t operations, const std::vector<Candidate>& candidates)
{
    std::vector<std::vector<double>> times(candidates.size());
    std::vector<uint64_t> checksums(candidates.size(), 0);

    for (int round = -1; round < g_rounds; ++round)
    {
        for (size_t c = 0; c < candidates.size(); ++c)
        {
            const auto start = std::chrono::steady_clock::now();
            const uint64_t checksum = candidates[c].run();
            const double elapsed = seconds_since(start);

            if (round < 0)
            {
                checksums[c] = checksum;
                continue;
            }
            if (checksum != checksums[c])
            {
                std::fprintf(stderr, "%s: %s is not deterministic\n", title, candidates[c].name);
                std::exit(1);
            }
            times[c].push_back(elapsed);
        }
    }

    for (size_t c = 1; c < candidates.size(); ++c)
    {
        if (checksums[c] != checksums[0])
        {
            std::fprintf(stderr, "%s: %s and %s disagree (%" PRIu64 " vs %" PRIu64 ")\n", title, candidates[0].name,
                         candidates[c].name, checksums[0], checksums[c]);
            std::exit(1);
        }
    }

    std::printf("\n%s  (%" PRIu64 " operations per run)\n", title, operations);
    std::printf("  %-22s %12s %12s %10s\n", "", "best ns/op", "median ns/op", "vs first");

    double first_best = 0.0;
    for (size_t c = 0; c < candidates.size(); ++c)
    {
        std::sort(times[c].begin(), times[c].end());
        const double best = times[c].front() * 1e9 / static_cast<double>(operations);
        const double median = times[c][times[c].size() / 2] * 1e9 / static_cast<double>(operations);
        if (c == 0)
        {
            first_best = best;
        }
        std::printf("  %-22s %12.2f %12.2f %9.2fx\n", candidates[c].name, best, median, first_best / best);
    }
}

template <typename T, typename Make>
std::vector<Candidate> push_candidates(size_t count, size_t repeats, bool reserve, Make make)
{
    std::vector<Candidate> candidates;

    candidates.push_back({"std::vector", [=]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            std::vector<T> items;
            if (reserve)
            {
                items.reserve(count);
            }
            for (size_t i = 0; i < count; ++i)
            {
                items.push_back(make(i));
            }
            sum += static_cast<uint64_t>(items.size()) + reinterpret_cast<const unsigned char*>(&items[count - 1])[0];
        }
        return sum;
    }});

    candidates.push_back({"ct::Vector", [=]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            ct::Vector<T> items;
            if (reserve)
            {
                items.reserve(count);
            }
            for (size_t i = 0; i < count; ++i)
            {
                items.push_back(make(i));
            }
            sum += static_cast<uint64_t>(items.size()) + reinterpret_cast<const unsigned char*>(&items[count - 1])[0];
        }
        return sum;
    }});

    candidates.push_back({"faisca::Array", [=]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            faisca::Array<T> items;
            if (reserve && !items.reserve(count))
            {
                std::exit(1);
            }
            for (size_t i = 0; i < count; ++i)
            {
                if (!items.push_back(make(i)))
                {
                    std::exit(1);
                }
            }
            sum += static_cast<uint64_t>(items.size()) + reinterpret_cast<const unsigned char*>(&items[count - 1])[0];
        }
        return sum;
    }});

    if (reserve)
    {
        candidates.push_back({"faisca::Array unchecked", [=]()
        {
            uint64_t sum = 0;
            for (size_t r = 0; r < repeats; ++r)
            {
                faisca::Array<T> items;
                if (!items.reserve(count))
                {
                    std::exit(1);
                }
                for (size_t i = 0; i < count; ++i)
                {
                    items.push_back_unchecked(make(i));
                }
                sum += static_cast<uint64_t>(items.size()) +
                       reinterpret_cast<const unsigned char*>(&items[count - 1])[0];
            }
            return sum;
        }});
    }

    return candidates;
}

std::vector<Candidate> copy_out_candidates(const uint32_t* source, size_t count, size_t repeats)
{
    std::vector<Candidate> candidates;

    candidates.push_back({"std::vector resize", [=]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            std::vector<uint32_t> items;
            items.resize(count);
            std::memcpy(items.data(), source, count * sizeof(uint32_t));
            sum += items[count / 2] + items[count - 1];
        }
        return sum;
    }});

    candidates.push_back({"ct::Vector resize", [=]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            ct::Vector<uint32_t> items;
            items.resize(count);
            std::memcpy(items.data(), source, count * sizeof(uint32_t));
            sum += items[count / 2] + items[count - 1];
        }
        return sum;
    }});

    candidates.push_back({"faisca::Array uninit", [=]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            faisca::Array<uint32_t> items;
            if (!items.resize_uninitialized(count))
            {
                std::exit(1);
            }
            std::memcpy(items.data(), source, count * sizeof(uint32_t));
            sum += items[count / 2] + items[count - 1];
        }
        return sum;
    }});

    return candidates;
}

using Keys = std::vector<std::string_view>;

std::vector<Candidate> build_candidates(const Keys& keys, size_t repeats)
{
    std::vector<Candidate> candidates;

    candidates.push_back({"std::unordered_map", [&keys, repeats]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            std::unordered_map<std::string_view, uint32_t> map;
            map.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i)
            {
                map.emplace(keys[i], static_cast<uint32_t>(i));
            }
            sum += map.size();
        }
        return sum;
    }});

    candidates.push_back({"ct::HashMap", [&keys, repeats]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            ct::HashMap<ct::StringView, uint32_t> map;
            map.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i)
            {
                const ct::StringView key(keys[i].data(), keys[i].size());
                if (map.find(key) == nullptr)
                {
                    map.put(key, static_cast<uint32_t>(i));
                }
            }
            sum += map.size();
        }
        return sum;
    }});

    candidates.push_back({"ct::HashMap + murmur", [&keys, repeats]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            ct::HashMap<ct::StringView, uint32_t, MurmurForCt> map;
            map.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i)
            {
                const ct::StringView key(keys[i].data(), keys[i].size());
                if (map.find(key) == nullptr)
                {
                    map.put(key, static_cast<uint32_t>(i));
                }
            }
            sum += map.size();
        }
        return sum;
    }});

    candidates.push_back({"faisca::StringMap", [&keys, repeats]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            faisca::StringMap map;
            if (!map.reserve(keys.size()))
            {
                std::exit(1);
            }
            for (size_t i = 0; i < keys.size(); ++i)
            {
                if (map.insert(keys[i], static_cast<uint32_t>(i)) == faisca::StringMap::Insert::Failed)
                {
                    std::exit(1);
                }
            }
            sum += map.size();
        }
        return sum;
    }});

    return candidates;
}

struct Maps
{
    std::unordered_map<std::string_view, uint32_t> std_map;
    ct::HashMap<ct::StringView, uint32_t> ct_map;
    ct::HashMap<ct::StringView, uint32_t, MurmurForCt> ct_murmur_map;
    faisca::StringMap own_map;

    explicit Maps(const Keys& keys)
    {
        std_map.reserve(keys.size());
        ct_map.reserve(keys.size());
        ct_murmur_map.reserve(keys.size());
        if (!own_map.reserve(keys.size()))
        {
            std::exit(1);
        }

        for (size_t i = 0; i < keys.size(); ++i)
        {
            const uint32_t value = static_cast<uint32_t>(i);
            const ct::StringView key(keys[i].data(), keys[i].size());

            std_map.emplace(keys[i], value);
            if (ct_map.find(key) == nullptr)
            {
                ct_map.put(key, value);
            }
            if (ct_murmur_map.find(key) == nullptr)
            {
                ct_murmur_map.put(key, value);
            }
            own_map.insert(keys[i], value);
        }
    }
};

std::vector<Candidate> lookup_candidates(const Maps& maps, const Keys& queries, size_t repeats)
{
    std::vector<Candidate> candidates;

    candidates.push_back({"std::unordered_map", [&maps, &queries, repeats]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            for (const std::string_view query : queries)
            {
                const auto it = maps.std_map.find(query);
                if (it != maps.std_map.end())
                {
                    sum += it->second + 1u;
                }
            }
        }
        return sum;
    }});

    candidates.push_back({"ct::HashMap", [&maps, &queries, repeats]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            for (const std::string_view query : queries)
            {
                const uint32_t* value = maps.ct_map.find(ct::StringView(query.data(), query.size()));
                if (value != nullptr)
                {
                    sum += *value + 1u;
                }
            }
        }
        return sum;
    }});

    candidates.push_back({"ct::HashMap + murmur", [&maps, &queries, repeats]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            for (const std::string_view query : queries)
            {
                const uint32_t* value = maps.ct_murmur_map.find(ct::StringView(query.data(), query.size()));
                if (value != nullptr)
                {
                    sum += *value + 1u;
                }
            }
        }
        return sum;
    }});

    candidates.push_back({"faisca::StringMap", [&maps, &queries, repeats]()
    {
        uint64_t sum = 0;
        for (size_t r = 0; r < repeats; ++r)
        {
            for (const std::string_view query : queries)
            {
                uint32_t value = 0;
                if (maps.own_map.find(query, value))
                {
                    sum += value + 1u;
                }
            }
        }
        return sum;
    }});

    return candidates;
}

size_t count_hits(const Maps& maps, const Keys& queries)
{
    size_t hits = 0;
    for (const std::string_view query : queries)
    {
        hits += maps.std_map.count(query);
    }
    return hits;
}

Keys unique_keys(const Keys& keys)
{
    std::unordered_map<std::string_view, uint32_t> seen;
    Keys out;
    for (const std::string_view key : keys)
    {
        if (seen.emplace(key, 0u).second)
        {
            out.push_back(key);
        }
    }
    return out;
}

void bench_maps(const char* label, const Keys& all_keys, const std::vector<std::string>& text_words)
{
    const Keys keys = unique_keys(all_keys);
    if (keys.empty())
    {
        return;
    }

    size_t total_bytes = 0;
    for (const std::string_view key : keys)
    {
        total_bytes += key.size();
    }
    std::printf("\n=== %s: %zu keys, %.1f bytes per key on average ===\n", label, keys.size(),
                static_cast<double>(total_bytes) / static_cast<double>(keys.size()));

    const size_t build_repeats = keys.size() < 1000 ? 2000 : 20;
    char title[160];

    std::snprintf(title, sizeof(title), "%s: build the map", label);
    compare(title, keys.size() * build_repeats, build_candidates(keys, build_repeats));

    const Maps maps(keys);
    std::mt19937_64 rng(12345);

    Keys hits = keys;
    std::shuffle(hits.begin(), hits.end(), rng);
    const size_t lookup_repeats = keys.size() < 1000 ? 5000 : 40;

    std::snprintf(title, sizeof(title), "%s: lookups that hit, random order", label);
    compare(title, hits.size() * lookup_repeats, lookup_candidates(maps, hits, lookup_repeats));

    std::vector<std::string> miss_storage;
    miss_storage.reserve(hits.size());
    for (const std::string_view key : hits)
    {
        if (key.empty())
        {
            continue;
        }
        std::string changed(key);
        changed[changed.size() / 2] = static_cast<char>(changed[changed.size() / 2] ^ 0x15);
        if (maps.std_map.find(changed) == maps.std_map.end())
        {
            miss_storage.push_back(changed);
        }
    }
    Keys misses(miss_storage.begin(), miss_storage.end());

    std::snprintf(title, sizeof(title), "%s: lookups that miss", label);
    compare(title, misses.size() * lookup_repeats, lookup_candidates(maps, misses, lookup_repeats));

    if (!text_words.empty())
    {
        Keys stream(text_words.begin(), text_words.end());
        const size_t found = count_hits(maps, stream);
        const size_t stream_repeats = 2000000 / stream.size() + 1;

        std::snprintf(title, sizeof(title), "%s: words of a real text, in order (%.0f%% found)", label,
                      100.0 * static_cast<double>(found) / static_cast<double>(stream.size()));
        compare(title, stream.size() * stream_repeats, lookup_candidates(maps, stream, stream_repeats));
    }
}

void copy_strings(const faisca::GgufArray& array, Keys& out)
{
    faisca::Array<std::string_view> items;
    if (!array.strings_to(items))
    {
        std::exit(1);
    }
    out.assign(items.begin(), items.end());
}

std::vector<std::string> words_of(const std::string& text)
{
    std::vector<std::string> words;
    std::string current;
    for (const char c : text)
    {
        const bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        if (letter)
        {
            current += c;
        }
        else if (!current.empty())
        {
            words.push_back("\xc4\xa0" + current);
            current.clear();
        }
    }
    if (!current.empty())
    {
        words.push_back("\xc4\xa0" + current);
    }
    return words;
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: bench_containers <model.gguf> [text.txt] [rounds]\n");
        return 2;
    }
    if (argc > 3)
    {
        g_rounds = std::atoi(argv[3]);
        if (g_rounds < 3)
        {
            g_rounds = 3;
        }
    }

    faisca::Error error;
    faisca::GgufFile file;
    if (!file.open(argv[1], error))
    {
        std::fprintf(stderr, "bench_containers: %s\n", error.c_str());
        return 1;
    }

    std::vector<std::string> text_words;
    if (argc > 2 && argv[2][0] != '\0')
    {
        std::ifstream stream(argv[2], std::ios::binary);
        if (!stream)
        {
            std::fprintf(stderr, "bench_containers: cannot read '%s'\n", argv[2]);
            return 1;
        }
        const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        text_words = words_of(text);
    }

    Keys tokens;
    Keys merges;
    Keys tensor_names;
    if (const faisca::GgufArray* array = file.get_array("tokenizer.ggml.tokens"))
    {
        copy_strings(*array, tokens);
    }
    if (const faisca::GgufArray* array = file.get_array("tokenizer.ggml.merges"))
    {
        copy_strings(*array, merges);
    }
    for (const GgufTensor& tensor : file.tensors())
    {
        tensor_names.push_back(tensor.name);
    }

    std::printf("model: %s\n", argv[1]);
    std::printf("tokens %zu, merges %zu, tensors %zu, text words %zu, rounds %d\n", tokens.size(), merges.size(),
                tensor_names.size(), text_words.size(), g_rounds);

    const faisca::Array<GgufTensor>& tensors = file.tensors();
    if (!tensors.empty())
    {
        const GgufTensor* records = tensors.data();
        const size_t n = tensors.size();
        const auto real_record = [records, n](size_t i)
        {
            return records[i % n];
        };

        std::printf("\n=== arrays of GgufTensor (%zu bytes each) ===\n", sizeof(GgufTensor));
        compare("push_back of the model's tensors, no reserve", n * 2000,
                push_candidates<GgufTensor>(n, 2000, false, real_record));
        compare("push_back of the model's tensors, reserved", n * 2000,
                push_candidates<GgufTensor>(n, 2000, true, real_record));
        compare("push_back of 100000 records, no reserve", 100000 * 20,
                push_candidates<GgufTensor>(100000, 20, false, real_record));
        compare("push_back of 100000 records, reserved", 100000 * 20,
                push_candidates<GgufTensor>(100000, 20, true, real_record));
    }

    {
        const auto number = [](size_t i)
        {
            return static_cast<uint32_t>(i * 2654435761u);
        };
        std::printf("\n=== arrays of uint32_t ===\n");
        compare("push_back of 1000000 values, no reserve", 1000000 * 10,
                push_candidates<uint32_t>(1000000, 10, false, number));
        compare("push_back of 1000000 values, reserved", 1000000 * 10,
                push_candidates<uint32_t>(1000000, 10, true, number));

        std::vector<uint32_t> source(1000000);
        for (size_t i = 0; i < source.size(); ++i)
        {
            source[i] = number(i);
        }
        compare("copy 50000 values into a new buffer (a token type array)", 50000 * 2000,
                copy_out_candidates(source.data(), 50000, 2000));
        compare("copy 1000000 values into a new buffer", 1000000 * 100,
                copy_out_candidates(source.data(), 1000000, 100));
    }

    bench_maps("tensor names", tensor_names, {});
    bench_maps("tokens", tokens, text_words);
    bench_maps("merges", merges, {});

    return 0;
}
