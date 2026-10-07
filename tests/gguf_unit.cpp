#include "faisca/gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
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

class Image
{
public:
    template <typename T>
    Image& put(T value)
    {
        const size_t at = bytes.size();
        bytes.resize(at + sizeof(T));
        std::memcpy(bytes.data() + at, &value, sizeof(T));
        return *this;
    }

    Image& str(std::string_view text)
    {
        put<uint64_t>(text.size());
        bytes.insert(bytes.end(), text.begin(), text.end());
        return *this;
    }

    Image& header(uint32_t version, uint64_t tensor_count, uint64_t kv_count)
    {
        bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
        put<uint32_t>(version);
        put<uint64_t>(tensor_count);
        put<uint64_t>(kv_count);
        return *this;
    }

    Image& key(std::string_view name, GgufType type)
    {
        str(name);
        put<uint32_t>(static_cast<uint32_t>(type));
        return *this;
    }

    Image& array(GgufType elem_type, uint64_t count)
    {
        put<uint32_t>(static_cast<uint32_t>(elem_type));
        put<uint64_t>(count);
        return *this;
    }

    Image& tensor(std::string_view name, const std::vector<uint64_t>& dims, uint32_t type, uint64_t offset)
    {
        str(name);
        put<uint32_t>(static_cast<uint32_t>(dims.size()));
        for (const uint64_t dim : dims)
        {
            put<uint64_t>(dim);
        }
        put<uint32_t>(type);
        put<uint64_t>(offset);
        return *this;
    }

    Image& pad(size_t alignment)
    {
        while (bytes.size() % alignment != 0)
        {
            bytes.push_back(0);
        }
        return *this;
    }

    Image& fill(size_t count, uint8_t first)
    {
        for (size_t i = 0; i < count; ++i)
        {
            bytes.push_back(static_cast<uint8_t>(first + i));
        }
        return *this;
    }

    std::vector<uint8_t> bytes;
};

bool parse(GgufFile& file, const Image& image, Error& error)
{
    return file.parse(image.bytes.data(), image.bytes.size(), error);
}

void check_rejected(const Image& image, const char* expected, int line)
{
    Error error;
    GgufFile file;
    const bool accepted = parse(file, image, error);
    ++g_checks;
    if (accepted)
    {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: image was accepted, expected an error with '%s'\n", __FILE__, line, expected);
        return;
    }
    if (std::strstr(error.c_str(), expected) == nullptr)
    {
        ++g_failures;
        std::fprintf(stderr, "%s:%d: error '%s' does not mention '%s'\n", __FILE__, line, error.c_str(), expected);
    }
}

#define CHECK_REJECTED(image, expected) check_rejected(image, expected, __LINE__)

void test_minimal()
{
    for (const uint32_t version : {2u, 3u})
    {
        Image image;
        image.header(version, 0, 0);

        Error error;
        GgufFile file;
        const bool parsed = parse(file, image, error);
        CHECK(parsed);
        CHECK(file.is_open() == parsed);
        if (!parsed)
        {
            continue;
        }
        CHECK(file.version() == version);
        CHECK(file.alignment() == 32);
        CHECK(file.metadata().empty());
        CHECK(file.tensors().empty());
        CHECK(file.find("anything") == nullptr);
        CHECK(file.find_tensor("anything") == nullptr);
    }
}

void test_scalars_and_getters()
{
    Image image;
    image.header(3, 0, 9);
    image.key("a.u32", GgufType::UInt32).put<uint32_t>(7);
    image.key("a.i32", GgufType::Int32).put<int32_t>(-1);
    image.key("a.i16", GgufType::Int16).put<int16_t>(12);
    image.key("a.u64", GgufType::UInt64).put<uint64_t>(std::numeric_limits<uint64_t>::max());
    image.key("a.f32", GgufType::Float32).put<float>(0.25f);
    image.key("a.f64", GgufType::Float64).put<double>(-3.5);
    image.key("a.bool", GgufType::Bool).put<uint8_t>(1);
    image.key("a.str", GgufType::String).str("text");
    image.key("a.empty", GgufType::String).str("");

    Error error;
    GgufFile file;
    const bool parsed = parse(file, image, error);
    CHECK(parsed);
    if (!parsed)
    {
        std::fprintf(stderr, "  %s\n", error.c_str());
        return;
    }
    CHECK(file.metadata().size() == 9);
    CHECK(file.metadata()[0].key == "a.u32");
    CHECK(file.metadata()[8].key == "a.empty");

    uint64_t u = 99;
    int64_t i = 99;
    double f = 99.0;
    bool b = false;
    std::string_view s = "untouched";

    CHECK(file.get_uint("a.u32", u) && u == 7);
    CHECK(file.get_int("a.u32", i) && i == 7);
    CHECK(file.get_int("a.i32", i) && i == -1);
    CHECK(file.get_uint("a.i16", u) && u == 12);
    CHECK(file.get_uint("a.u64", u) && u == std::numeric_limits<uint64_t>::max());
    CHECK(file.get_float("a.f32", f) && f == 0.25);
    CHECK(file.get_float("a.f64", f) && f == -3.5);
    CHECK(file.get_bool("a.bool", b) && b);
    CHECK(file.get_string("a.str", s) && s == "text");
    CHECK(file.get_string("a.empty", s) && s.empty());

    u = 99;
    i = 99;
    f = 99.0;
    s = "untouched";
    CHECK(!file.get_uint("a.i32", u) && u == 99);
    CHECK(!file.get_int("a.u64", i) && i == 99);
    CHECK(!file.get_uint("a.f32", u) && u == 99);
    CHECK(!file.get_uint("a.bool", u) && u == 99);
    CHECK(!file.get_float("a.u32", f) && f == 99.0);
    CHECK(!file.get_string("a.u32", s) && s == "untouched");
    CHECK(!file.get_bool("a.u32", b));
    CHECK(!file.get_uint("missing", u) && u == 99);
    CHECK(file.get_array("a.u32") == nullptr);
    CHECK(file.get_array("missing") == nullptr);
}

void test_arrays()
{
    Image image;
    image.header(3, 0, 6);
    image.key("arr.u32", GgufType::Array).array(GgufType::UInt32, 3);
    image.put<uint32_t>(10).put<uint32_t>(20).put<uint32_t>(30);
    image.key("arr.str", GgufType::Array).array(GgufType::String, 3);
    image.str("one").str("").str("three");
    image.key("arr.empty_u32", GgufType::Array).array(GgufType::UInt32, 0);
    image.key("arr.empty_str", GgufType::Array).array(GgufType::String, 0);
    image.key("arr.nested", GgufType::Array).array(GgufType::Array, 2);
    image.array(GgufType::Int32, 2).put<int32_t>(1).put<int32_t>(2);
    image.array(GgufType::Int32, 1).put<int32_t>(3);
    image.key("after", GgufType::UInt32).put<uint32_t>(4242);

    Error error;
    GgufFile file;
    const bool parsed = parse(file, image, error);
    CHECK(parsed);
    if (!parsed)
    {
        std::fprintf(stderr, "  %s\n", error.c_str());
        return;
    }

    const GgufArray* numbers = file.get_array("arr.u32");
    CHECK(numbers != nullptr);
    if (numbers != nullptr)
    {
        CHECK(numbers->elem_type == GgufType::UInt32);
        CHECK(numbers->count == 3);
        CHECK(numbers->byte_size == 12);

        Array<uint32_t> values;
        CHECK(numbers->copy_to(values));
        CHECK(values.size() == 3 && values[0] == 10 && values[1] == 20 && values[2] == 30);

        Array<int32_t> wrong_type;
        CHECK(wrong_type.push_back(5));
        CHECK(!numbers->copy_to(wrong_type));
        CHECK(wrong_type.size() == 1 && wrong_type[0] == 5);
        Array<std::string_view> not_strings;
        CHECK(!numbers->strings_to(not_strings));
    }

    const GgufArray* strings = file.get_array("arr.str");
    CHECK(strings != nullptr);
    if (strings != nullptr)
    {
        Array<std::string_view> items;
        CHECK(strings->strings_to(items));
        CHECK(items.size() == 3);
        if (items.size() == 3)
        {
            CHECK(items[0] == "one");
            CHECK(items[1].empty());
            CHECK(items[2] == "three");
        }
        CHECK(strings->byte_size == 3 * 8 + 3 + 0 + 5);
    }

    const GgufArray* empty_numbers = file.get_array("arr.empty_u32");
    CHECK(empty_numbers != nullptr);
    if (empty_numbers != nullptr)
    {
        Array<uint32_t> values;
        CHECK(values.push_back(1) && values.push_back(2));
        CHECK(empty_numbers->count == 0 && empty_numbers->byte_size == 0);
        CHECK(empty_numbers->copy_to(values) && values.empty());
    }

    const GgufArray* empty_strings = file.get_array("arr.empty_str");
    CHECK(empty_strings != nullptr);
    if (empty_strings != nullptr)
    {
        Array<std::string_view> items;
        CHECK(items.push_back("x"));
        CHECK(empty_strings->strings_to(items) && items.empty());
    }

    const GgufArray* nested = file.get_array("arr.nested");
    CHECK(nested != nullptr);
    if (nested != nullptr)
    {
        CHECK(nested->elem_type == GgufType::Array);
        CHECK(nested->count == 2);
        CHECK(nested->byte_size == (12 + 8) + (12 + 4));
    }

    uint64_t after = 0;
    CHECK(file.get_uint("after", after) && after == 4242);
}

void test_tensors()
{
    Image image;
    image.header(3, 4, 0);
    image.tensor("scalar", {}, 0, 0);
    image.tensor("matrix", {4, 2}, 1, 32);
    image.tensor("empty", {0, 5}, 0, 64);
    image.tensor("quant", {32, 2}, 8, 64);
    image.pad(32);
    const size_t data_start = image.bytes.size();
    image.fill(4, 1).pad(32);
    image.fill(16, 100).pad(32);
    image.fill(68, 7);

    Error error;
    GgufFile file;
    const bool parsed = parse(file, image, error);
    CHECK(parsed);
    if (!parsed)
    {
        std::fprintf(stderr, "  %s\n", error.c_str());
        return;
    }
    CHECK(file.data_offset() == data_start);
    CHECK(file.tensors().size() == 4);

    const GgufTensor* scalar = file.find_tensor("scalar");
    CHECK(scalar != nullptr);
    if (scalar != nullptr)
    {
        CHECK(scalar->n_dims == 0 && scalar->n_elements == 1 && scalar->n_bytes == 4);
        CHECK(scalar->data == image.bytes.data() + data_start);
        CHECK(scalar->data[0] == 1 && scalar->data[3] == 4);
    }

    const GgufTensor* matrix = file.find_tensor("matrix");
    CHECK(matrix != nullptr);
    if (matrix != nullptr)
    {
        CHECK(matrix->type == GgmlType::F16);
        CHECK(matrix->n_dims == 2 && matrix->dims[0] == 4 && matrix->dims[1] == 2);
        CHECK(matrix->n_elements == 8 && matrix->n_bytes == 16);
        CHECK(matrix->data == image.bytes.data() + data_start + 32);
        CHECK(matrix->data[0] == 100 && matrix->data[15] == 115);
    }

    const GgufTensor* empty = file.find_tensor("empty");
    CHECK(empty != nullptr);
    if (empty != nullptr)
    {
        CHECK(empty->n_elements == 0 && empty->n_bytes == 0);
    }

    const GgufTensor* quant = file.find_tensor("quant");
    CHECK(quant != nullptr);
    if (quant != nullptr)
    {
        CHECK(quant->type == GgmlType::Q8_0);
        CHECK(quant->n_elements == 64 && quant->n_bytes == 68);
        CHECK(quant->data[0] == 7 && quant->data[67] == 74);
    }

    CHECK(file.find_tensor("Matrix") == nullptr);
}

void test_custom_alignment()
{
    Image image;
    image.header(3, 2, 1);
    image.key("general.alignment", GgufType::UInt32).put<uint32_t>(8);
    image.tensor("a", {3}, 24, 0);
    image.tensor("b", {2}, 24, 8);
    image.pad(8);
    image.fill(3, 1).pad(8);
    image.fill(2, 50);

    Error error;
    GgufFile file;
    const bool parsed = parse(file, image, error);
    CHECK(parsed);
    if (!parsed)
    {
        std::fprintf(stderr, "  %s\n", error.c_str());
        return;
    }
    CHECK(file.alignment() == 8);
    CHECK(file.data_offset() % 8 == 0);
    const GgufTensor* b = file.find_tensor("b");
    CHECK(b != nullptr && b->data[0] == 50 && b->data[1] == 51);
}

void test_rejections()
{
    {
        Image image;
        CHECK_REJECTED(image, "bad magic");
    }
    {
        Image image;
        image.header(3, 0, 0);
        image.bytes[1] = 'X';
        CHECK_REJECTED(image, "bad magic");
    }
    {
        Image image;
        image.header(1, 0, 0);
        CHECK_REJECTED(image, "unsupported GGUF version 1");
    }
    {
        Image image;
        image.header(4, 0, 0);
        CHECK_REJECTED(image, "unsupported GGUF version 4");
    }
    {
        Image image;
        image.header(0x03000000u, 0, 0);
        CHECK_REJECTED(image, "big-endian");
    }
    {
        Image image;
        image.header(3, 0, 0);
        image.bytes.resize(20);
        CHECK_REJECTED(image, "header is truncated");
    }
    {
        Image image;
        image.header(3, 0, 1);
        CHECK_REJECTED(image, "larger than the file");
    }
    {
        Image image;
        image.header(3, 0, 2);
        image.key("same", GgufType::UInt32).put<uint32_t>(1);
        image.key("same", GgufType::UInt32).put<uint32_t>(2);
        CHECK_REJECTED(image, "duplicate metadata key 'same'");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("", GgufType::UInt32).put<uint32_t>(1).put<uint32_t>(0);
        CHECK_REJECTED(image, "invalid length");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("flag", GgufType::Bool).put<uint8_t>(2);
        CHECK_REJECTED(image, "expected 0 or 1");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.str("odd").put<uint32_t>(13).put<uint64_t>(0);
        CHECK_REJECTED(image, "unknown value type 13");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("arr", GgufType::Array).put<uint32_t>(13).put<uint64_t>(0).put<uint64_t>(0);
        CHECK_REJECTED(image, "unknown element type 13");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("text", GgufType::String).put<uint64_t>(1000).put<uint64_t>(0);
        CHECK_REJECTED(image, "string is truncated");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("arr", GgufType::Array).array(GgufType::UInt32, 1ull << 62).put<uint64_t>(0);
        CHECK_REJECTED(image, "larger than the file");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("arr", GgufType::Array).array(GgufType::String, 1ull << 62).put<uint64_t>(0);
        CHECK_REJECTED(image, "larger than the file");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("deep", GgufType::Array);
        for (int level = 0; level < 10; ++level)
        {
            image.array(GgufType::Array, 1);
        }
        image.array(GgufType::UInt8, 0);
        CHECK_REJECTED(image, "nested too deeply");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("general.alignment", GgufType::UInt32).put<uint32_t>(0);
        CHECK_REJECTED(image, "general.alignment is 0");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("general.alignment", GgufType::UInt32).put<uint32_t>(12);
        CHECK_REJECTED(image, "general.alignment is 12");
    }
    {
        Image image;
        image.header(3, 0, 1);
        image.key("general.alignment", GgufType::String).str("32");
        CHECK_REJECTED(image, "not an unsigned integer");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {1, 1, 1, 1, 1}, 0, 0);
        image.pad(32).fill(4, 0);
        CHECK_REJECTED(image, "5 dimensions");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {4}, 4, 0);
        image.pad(32).fill(64, 0);
        CHECK_REJECTED(image, "unknown tensor type 4");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {4}, 999, 0);
        image.pad(32).fill(64, 0);
        CHECK_REJECTED(image, "unknown tensor type 999");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {33}, 8, 0);
        image.pad(32).fill(128, 0);
        CHECK_REJECTED(image, "not a multiple of the Q8_0 block size 32");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {4}, 0, 8);
        image.pad(32).fill(64, 0);
        CHECK_REJECTED(image, "is not aligned to 32");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {4}, 0, 0);
        image.pad(32).fill(15, 0);
        CHECK_REJECTED(image, "outside the file");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {4}, 0, 0xffffffffffffffe0ull);
        image.pad(32).fill(64, 0);
        CHECK_REJECTED(image, "outside the file");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {1ull << 40, 1ull << 40}, 0, 0);
        image.pad(32).fill(64, 0);
        CHECK_REJECTED(image, "element count overflows");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.tensor("t", {1ull << 63}, 28, 0);
        image.pad(32).fill(64, 0);
        CHECK_REJECTED(image, "byte size overflows");
    }
    {
        Image image;
        image.header(3, 2, 0);
        image.tensor("t", {4}, 0, 0);
        image.tensor("t", {4}, 0, 32);
        image.pad(32).fill(64, 0);
        CHECK_REJECTED(image, "duplicate tensor name 't'");
    }
    {
        Image image;
        image.header(3, 1, 0);
        image.str("t").put<uint32_t>(2).put<uint64_t>(4).put<uint32_t>(0);
        CHECK_REJECTED(image, "info is truncated");
    }
}

void test_type_table()
{
    CHECK(ggml_type_info(GgmlType::F32)->block_bytes == 4);
    CHECK(ggml_type_info(GgmlType::Q4_K)->block_elements == 256);
    CHECK(ggml_type_info(GgmlType::Q4_K)->block_bytes == 144);
    CHECK(ggml_type_info(static_cast<GgmlType>(4)) == nullptr);
    CHECK(ggml_type_info(static_cast<GgmlType>(31)) == nullptr);
    CHECK(ggml_type_info(static_cast<GgmlType>(1000)) == nullptr);
    CHECK(gguf_scalar_size(GgufType::Float64) == 8);
    CHECK(gguf_scalar_size(GgufType::String) == 0);

    const uint8_t bytes[11] = {1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1};
    CHECK(checksum64(bytes, 11) == 1 * 1 + 0x010002ull * 3 + 11);
    CHECK(checksum64(bytes, 0) == 0);
}

void test_files_on_disk()
{
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "faisca_gguf_unit";
    fs::create_directories(dir);

    Error error;
    GgufFile file;

    CHECK(!file.open((dir / "does_not_exist.gguf").u8string().c_str(), error));
    CHECK(std::strstr(error.c_str(), "cannot open") != nullptr);
    CHECK(!file.is_open());

    error.clear();
    CHECK(!file.open(dir.u8string().c_str(), error));
    CHECK(!error.empty());

    const fs::path empty_path = dir / "empty.gguf";
    {
        std::ofstream stream(empty_path, std::ios::binary);
    }
    error.clear();
    CHECK(!file.open(empty_path.u8string().c_str(), error));
    CHECK(std::strstr(error.c_str(), "is empty") != nullptr);

    Image image;
    image.header(3, 1, 1);
    image.key("general.name", GgufType::String).str("on disk");
    image.tensor("w", {2, 3}, 0, 0);
    image.pad(32);
    image.fill(24, 9);

    const fs::path good_path = dir / fs::u8path("modelo_a\xc3\xa7\xc3\xba" "car.gguf");
    {
        std::ofstream stream(good_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(image.bytes.data()),
                     static_cast<std::streamsize>(image.bytes.size()));
    }
    error.clear();
    const bool opened = file.open(good_path.u8string().c_str(), error);
    CHECK(opened);
    if (opened)
    {
        std::string_view name;
        CHECK(file.is_open());
        CHECK(file.get_string("general.name", name) && name == "on disk");
        CHECK(file.file_size() == image.bytes.size());
        const GgufTensor* w = file.find_tensor("w");
        CHECK(w != nullptr && w->n_bytes == 24 && w->data[0] == 9 && w->data[23] == 32);
    }
    else
    {
        std::fprintf(stderr, "  %s\n", error.c_str());
    }

    const fs::path bad_path = dir / "bad.gguf";
    {
        std::ofstream stream(bad_path, std::ios::binary);
        stream << "this is not a model";
    }
    error.clear();
    CHECK(!file.open(bad_path.u8string().c_str(), error));
    CHECK(std::strstr(error.c_str(), "bad.gguf") != nullptr && std::strstr(error.c_str(), "bad magic") != nullptr);
    CHECK(!file.is_open());
    CHECK(file.metadata().empty() && file.tensors().empty());
    CHECK(file.find("general.name") == nullptr);

    CHECK(file.open(good_path.u8string().c_str(), error));
    file.close();
    CHECK(!file.is_open());
    CHECK(file.find_tensor("w") == nullptr);

    std::error_code ignored;
    fs::remove_all(dir, ignored);
    CHECK(!fs::exists(dir));
}

}

int main()
{
    test_minimal();
    test_scalars_and_getters();
    test_arrays();
    test_tensors();
    test_custom_alignment();
    test_rejections();
    test_type_table();
    test_files_on_disk();

    std::printf("%s gguf_unit: %d checks, %d failed\n", g_failures == 0 ? "ok  " : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
