#include "faisca/image.h"

#include "faisca/mapped_file.h"

#include <cmath>
#include <cstring>

namespace faisca
{

namespace
{

// ---- inflate (RFC 1951) ----

struct BitReader
{
    const uint8_t* data;
    size_t size;
    size_t pos = 0;
    uint32_t bits = 0;
    int count = 0;

    bool need(int n)
    {
        while (count < n)
        {
            if (pos >= size)
            {
                return false;
            }
            bits |= static_cast<uint32_t>(data[pos++]) << count;
            count += 8;
        }
        return true;
    }

    bool read(int n, uint32_t& out)
    {
        if (n == 0)
        {
            out = 0;
            return true;
        }
        if (!need(n))
        {
            return false;
        }
        out = bits & ((1u << n) - 1u);
        bits >>= n;
        count -= n;
        return true;
    }

    void align()
    {
        bits >>= count % 8;
        count -= count % 8;
    }
};

struct Huffman
{
    uint16_t count[16];
    uint16_t symbol[288];
};

bool build_huffman(const uint8_t* lengths, size_t n, Huffman& h)
{
    std::memset(h.count, 0, sizeof(h.count));
    for (size_t i = 0; i < n; ++i)
    {
        h.count[lengths[i]]++;
    }
    h.count[0] = 0;
    int left = 1;
    for (int len = 1; len < 16; ++len)
    {
        left <<= 1;
        left -= h.count[len];
        if (left < 0)
        {
            return false;
        }
    }
    uint16_t offsets[16];
    offsets[1] = 0;
    for (int len = 1; len < 15; ++len)
    {
        offsets[len + 1] = static_cast<uint16_t>(offsets[len] + h.count[len]);
    }
    for (size_t i = 0; i < n; ++i)
    {
        if (lengths[i] != 0)
        {
            h.symbol[offsets[lengths[i]]++] = static_cast<uint16_t>(i);
        }
    }
    return true;
}

int decode_symbol(BitReader& r, const Huffman& h)
{
    int code = 0;
    int first = 0;
    int index = 0;
    for (int len = 1; len < 16; ++len)
    {
        uint32_t bit = 0;
        if (!r.read(1, bit))
        {
            return -1;
        }
        code |= static_cast<int>(bit);
        const int count = h.count[len];
        if (code - count < first)
        {
            return h.symbol[index + (code - first)];
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    return -1;
}

const uint16_t LENGTH_BASE[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const uint8_t LENGTH_EXTRA[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const uint16_t DIST_BASE[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
const uint8_t DIST_EXTRA[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

bool inflate_block(BitReader& r, const Huffman& lit, const Huffman& dist, Array<uint8_t>& out)
{
    for (;;)
    {
        const int symbol = decode_symbol(r, lit);
        if (symbol < 0)
        {
            return false;
        }
        if (symbol < 256)
        {
            if (!out.push_back(static_cast<uint8_t>(symbol)))
            {
                return false;
            }
            continue;
        }
        if (symbol == 256)
        {
            return true;
        }
        const int li = symbol - 257;
        if (li >= 29)
        {
            return false;
        }
        uint32_t extra = 0;
        if (!r.read(LENGTH_EXTRA[li], extra))
        {
            return false;
        }
        const size_t length = LENGTH_BASE[li] + extra;
        const int di = decode_symbol(r, dist);
        if (di < 0 || di >= 30 || !r.read(DIST_EXTRA[di], extra))
        {
            return false;
        }
        const size_t distance = DIST_BASE[di] + extra;
        if (distance > out.size())
        {
            return false;
        }
        for (size_t i = 0; i < length; ++i)
        {
            const uint8_t byte = out[out.size() - distance];
            if (!out.push_back(byte))
            {
                return false;
            }
        }
    }
}

bool inflate(const uint8_t* data, size_t size, Array<uint8_t>& out, Error& error)
{
    BitReader r{data, size};
    uint32_t final = 0;
    do
    {
        uint32_t type = 0;
        if (!r.read(1, final) || !r.read(2, type))
        {
            error.set("truncated deflate stream");
            return false;
        }
        if (type == 0)
        {
            r.align();
            uint32_t len = 0;
            uint32_t nlen = 0;
            if (!r.read(16, len) || !r.read(16, nlen) || (len ^ 0xffffu) != nlen)
            {
                error.set("bad stored block");
                return false;
            }
            for (uint32_t i = 0; i < len; ++i)
            {
                uint32_t byte = 0;
                if (!r.read(8, byte) || !out.push_back(static_cast<uint8_t>(byte)))
                {
                    error.set("truncated stored block");
                    return false;
                }
            }
            continue;
        }
        Huffman lit;
        Huffman dist;
        uint8_t lengths[320];
        if (type == 1)
        {
            for (int i = 0; i < 144; ++i)
            {
                lengths[i] = 8;
            }
            for (int i = 144; i < 256; ++i)
            {
                lengths[i] = 9;
            }
            for (int i = 256; i < 280; ++i)
            {
                lengths[i] = 7;
            }
            for (int i = 280; i < 288; ++i)
            {
                lengths[i] = 8;
            }
            build_huffman(lengths, 288, lit);
            for (int i = 0; i < 30; ++i)
            {
                lengths[i] = 5;
            }
            build_huffman(lengths, 30, dist);
        }
        else if (type == 2)
        {
            uint32_t hlit = 0;
            uint32_t hdist = 0;
            uint32_t hclen = 0;
            if (!r.read(5, hlit) || !r.read(5, hdist) || !r.read(4, hclen))
            {
                error.set("truncated deflate stream");
                return false;
            }
            hlit += 257;
            hdist += 1;
            hclen += 4;
            static const uint8_t ORDER[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            uint8_t code_lengths[19] = {};
            for (uint32_t i = 0; i < hclen; ++i)
            {
                uint32_t v = 0;
                if (!r.read(3, v))
                {
                    error.set("truncated deflate stream");
                    return false;
                }
                code_lengths[ORDER[i]] = static_cast<uint8_t>(v);
            }
            Huffman codes;
            if (!build_huffman(code_lengths, 19, codes))
            {
                error.set("bad code length codes");
                return false;
            }
            uint32_t n = 0;
            while (n < hlit + hdist)
            {
                const int symbol = decode_symbol(r, codes);
                if (symbol < 0)
                {
                    error.set("bad code lengths");
                    return false;
                }
                if (symbol < 16)
                {
                    lengths[n++] = static_cast<uint8_t>(symbol);
                    continue;
                }
                uint32_t repeat = 0;
                uint8_t value = 0;
                if (symbol == 16)
                {
                    if (n == 0 || !r.read(2, repeat))
                    {
                        error.set("bad code lengths");
                        return false;
                    }
                    value = lengths[n - 1];
                    repeat += 3;
                }
                else if (symbol == 17)
                {
                    if (!r.read(3, repeat))
                    {
                        error.set("bad code lengths");
                        return false;
                    }
                    repeat += 3;
                }
                else
                {
                    if (!r.read(7, repeat))
                    {
                        error.set("bad code lengths");
                        return false;
                    }
                    repeat += 11;
                }
                if (n + repeat > hlit + hdist)
                {
                    error.set("bad code lengths");
                    return false;
                }
                for (uint32_t i = 0; i < repeat; ++i)
                {
                    lengths[n++] = value;
                }
            }
            if (!build_huffman(lengths, hlit, lit) || !build_huffman(lengths + hlit, hdist, dist))
            {
                error.set("bad huffman tables");
                return false;
            }
        }
        else
        {
            error.set("bad deflate block type");
            return false;
        }
        if (!inflate_block(r, lit, dist, out))
        {
            error.set("bad deflate data");
            return false;
        }
    } while (final == 0);
    return true;
}

// ---- PNG ----

uint32_t read_be32(const uint8_t* p)
{
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) | (static_cast<uint32_t>(p[2]) << 8) | p[3];
}

int paeth(int a, int b, int c)
{
    const int p = a + b - c;
    const int pa = p > a ? p - a : a - p;
    const int pb = p > b ? p - b : b - p;
    const int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc)
    {
        return a;
    }
    return pb <= pc ? b : c;
}

bool decode_png(const uint8_t* data, size_t size, Image& out, Error& error)
{
    if (size < 8 || std::memcmp(data, "\x89PNG\r\n\x1a\n", 8) != 0)
    {
        error.set("not a PNG file");
        return false;
    }
    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t depth = 0;
    uint8_t color = 0;
    Array<uint8_t> zdata;
    uint8_t palette[256 * 3];
    size_t palette_entries = 0;
    size_t pos = 8;
    bool seen_header = false;
    while (pos + 8 <= size)
    {
        const uint32_t length = read_be32(data + pos);
        const uint8_t* type = data + pos + 4;
        const uint8_t* body = data + pos + 8;
        if (length > size - pos - 8)
        {
            error.set("truncated PNG chunk");
            return false;
        }
        if (std::memcmp(type, "IHDR", 4) == 0)
        {
            if (length != 13)
            {
                error.set("bad PNG header");
                return false;
            }
            width = read_be32(body);
            height = read_be32(body + 4);
            depth = body[8];
            color = body[9];
            if (body[12] != 0)
            {
                error.set("interlaced PNG is not supported");
                return false;
            }
            if ((depth != 8 && depth != 16) || (color != 0 && color != 2 && color != 3 && color != 4 && color != 6) ||
                (color == 3 && depth != 8))
            {
                error.set("PNG bit depth %d with colour type %d is not supported", depth, color);
                return false;
            }
            seen_header = true;
        }
        else if (std::memcmp(type, "PLTE", 4) == 0)
        {
            palette_entries = length / 3;
            if (palette_entries > 256)
            {
                error.set("bad PNG palette");
                return false;
            }
            std::memcpy(palette, body, palette_entries * 3);
        }
        else if (std::memcmp(type, "IDAT", 4) == 0)
        {
            if (!zdata.append(body, length))
            {
                error.set("out of memory");
                return false;
            }
        }
        else if (std::memcmp(type, "IEND", 4) == 0)
        {
            break;
        }
        pos += 12 + length;
    }
    if (!seen_header || width == 0 || height == 0 || zdata.size() < 2)
    {
        error.set("PNG has no image data");
        return false;
    }
    if ((zdata[0] & 0x0f) != 8 || ((zdata[0] << 8) | zdata[1]) % 31 != 0)
    {
        error.set("bad zlib header in PNG");
        return false;
    }
    const size_t channels = color == 0 ? 1 : color == 2 ? 3 : color == 3 ? 1 : color == 4 ? 2 : 4;
    const size_t bpp = channels * (depth / 8);
    const size_t stride = static_cast<size_t>(width) * bpp;
    Array<uint8_t> raw;
    if (!raw.reserve((stride + 1) * height) || !inflate(zdata.data() + 2, zdata.size() - 2, raw, error))
    {
        return false;
    }
    if (raw.size() < (stride + 1) * height)
    {
        error.set("PNG image data is short");
        return false;
    }
    if (!out.rgb.resize_uninitialized(static_cast<size_t>(width) * height * 3))
    {
        error.set("out of memory");
        return false;
    }
    out.width = width;
    out.height = height;
    Array<uint8_t> previous;
    Array<uint8_t> current;
    if (!previous.resize_uninitialized(stride) || !current.resize_uninitialized(stride))
    {
        error.set("out of memory");
        return false;
    }
    std::memset(previous.data(), 0, stride);
    for (uint32_t y = 0; y < height; ++y)
    {
        const uint8_t* line = raw.data() + y * (stride + 1);
        const uint8_t filter = line[0];
        const uint8_t* src = line + 1;
        uint8_t* cur = current.data();
        const uint8_t* prev = previous.data();
        for (size_t i = 0; i < stride; ++i)
        {
            const int a = i >= bpp ? cur[i - bpp] : 0;
            const int b = prev[i];
            const int c = i >= bpp ? prev[i - bpp] : 0;
            int value = src[i];
            switch (filter)
            {
            case 1:
                value += a;
                break;
            case 2:
                value += b;
                break;
            case 3:
                value += (a + b) / 2;
                break;
            case 4:
                value += paeth(a, b, c);
                break;
            default:
                break;
            }
            cur[i] = static_cast<uint8_t>(value);
        }
        uint8_t* dst = out.rgb.data() + static_cast<size_t>(y) * width * 3;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint8_t* px = cur + x * bpp;
            const size_t step = depth / 8;
            if (color == 3)
            {
                const uint8_t index = px[0];
                if (index >= palette_entries)
                {
                    error.set("PNG palette index out of range");
                    return false;
                }
                dst[0] = palette[index * 3];
                dst[1] = palette[index * 3 + 1];
                dst[2] = palette[index * 3 + 2];
            }
            else if (channels <= 2)
            {
                dst[0] = dst[1] = dst[2] = px[0];
            }
            else
            {
                dst[0] = px[0];
                dst[1] = px[step];
                dst[2] = px[2 * step];
            }
            dst += 3;
        }
        current.swap(previous);
    }
    return true;
}

// ---- baseline JPEG, the arithmetic of stb_image ----

struct JpegHuffman
{
    uint8_t lengths[17];
    uint8_t values[256];
    uint16_t codes[256];
    uint16_t count = 0;
    int maxcode[18];
    int valptr[17];
    int mincode[17];
};

struct JpegComponent
{
    int id = 0;
    int h = 1;
    int v = 1;
    int tq = 0;
    int td = 0;
    int ta = 0;
    int dc_pred = 0;
    size_t w2 = 0;
    size_t h2 = 0;
    Array<uint8_t> plane;
};

struct JpegReader
{
    const uint8_t* data;
    size_t size;
    size_t pos;
    uint32_t buffer = 0;
    int bits = 0;
    bool marker_hit = false;
    uint8_t marker = 0;

    void fill()
    {
        while (bits <= 24)
        {
            uint32_t byte = 0;
            if (!marker_hit && pos < size)
            {
                byte = data[pos++];
                if (byte == 0xff)
                {
                    const uint8_t next = pos < size ? data[pos] : 0xd9;
                    if (next == 0)
                    {
                        ++pos;
                    }
                    else
                    {
                        marker_hit = true;
                        marker = next;
                        byte = 0;
                    }
                }
            }
            buffer |= byte << (24 - bits);
            bits += 8;
        }
    }

    uint32_t get(int n)
    {
        if (n == 0)
        {
            return 0;
        }
        if (bits < n)
        {
            fill();
        }
        const uint32_t value = buffer >> (32 - n);
        buffer <<= n;
        bits -= n;
        return value;
    }

    void reset()
    {
        buffer = 0;
        bits = 0;
        marker_hit = false;
    }
};

int jpeg_decode(JpegReader& r, const JpegHuffman& h)
{
    if (r.bits < 16)
    {
        r.fill();
    }
    int code = 0;
    for (int len = 1; len <= 16; ++len)
    {
        code = (code << 1) | static_cast<int>(r.get(1));
        if (h.maxcode[len] >= 0 && code <= h.maxcode[len])
        {
            const int index = h.valptr[len] + code - h.mincode[len];
            return index >= 0 && index < h.count ? h.values[index] : -1;
        }
    }
    return -1;
}

int jpeg_extend(JpegReader& r, int n)
{
    static const int BIAS[16] = {0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767};
    const int k = static_cast<int>(r.get(n));
    return k < (1 << (n - 1)) ? k + BIAS[n] : k;
}

const uint8_t DEZIGZAG[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                              41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                              30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

inline int f2f(float x)
{
    return static_cast<int>(x * 4096 + 0.5);
}

inline int fsh(int x)
{
    return x * 4096;
}

inline uint8_t clamp_u8(int x)
{
    if (static_cast<unsigned>(x) > 255)
    {
        return x < 0 ? 0 : 255;
    }
    return static_cast<uint8_t>(x);
}

#define FAISCA_IDCT_1D(s0, s1, s2, s3, s4, s5, s6, s7)                                                                 \
    int t0, t1, t2, t3, p1, p2, p3, p4, p5, x0, x1, x2, x3;                                                            \
    p2 = s2;                                                                                                           \
    p3 = s6;                                                                                                           \
    p1 = (p2 + p3) * f2f(0.5411961f);                                                                                  \
    t2 = p1 + p3 * f2f(-1.847759065f);                                                                                 \
    t3 = p1 + p2 * f2f(0.765366865f);                                                                                  \
    p2 = s0;                                                                                                           \
    p3 = s4;                                                                                                           \
    t0 = fsh(p2 + p3);                                                                                                 \
    t1 = fsh(p2 - p3);                                                                                                 \
    x0 = t0 + t3;                                                                                                      \
    x3 = t0 - t3;                                                                                                      \
    x1 = t1 + t2;                                                                                                      \
    x2 = t1 - t2;                                                                                                      \
    t0 = s7;                                                                                                           \
    t1 = s5;                                                                                                           \
    t2 = s3;                                                                                                           \
    t3 = s1;                                                                                                           \
    p3 = t0 + t2;                                                                                                      \
    p4 = t1 + t3;                                                                                                      \
    p1 = t0 + t3;                                                                                                      \
    p2 = t1 + t2;                                                                                                      \
    p5 = (p3 + p4) * f2f(1.175875602f);                                                                                \
    t0 = t0 * f2f(0.298631336f);                                                                                       \
    t1 = t1 * f2f(2.053119869f);                                                                                       \
    t2 = t2 * f2f(3.072711026f);                                                                                       \
    t3 = t3 * f2f(1.501321110f);                                                                                       \
    p1 = p5 + p1 * f2f(-0.899976223f);                                                                                 \
    p2 = p5 + p2 * f2f(-2.562915447f);                                                                                 \
    p3 = p3 * f2f(-1.961570560f);                                                                                      \
    p4 = p4 * f2f(-0.390180644f);                                                                                      \
    t3 += p1 + p4;                                                                                                     \
    t2 += p2 + p3;                                                                                                     \
    t1 += p2 + p4;                                                                                                     \
    t0 += p1 + p3;

void idct_block(uint8_t* out, size_t out_stride, const short data[64])
{
    int val[64];
    int* v = val;
    const short* d = data;
    for (int i = 0; i < 8; ++i, ++d, ++v)
    {
        if (d[8] == 0 && d[16] == 0 && d[24] == 0 && d[32] == 0 && d[40] == 0 && d[48] == 0 && d[56] == 0)
        {
            const int dcterm = d[0] * 4;
            v[0] = v[8] = v[16] = v[24] = v[32] = v[40] = v[48] = v[56] = dcterm;
        }
        else
        {
            FAISCA_IDCT_1D(d[0], d[8], d[16], d[24], d[32], d[40], d[48], d[56])
            x0 += 512;
            x1 += 512;
            x2 += 512;
            x3 += 512;
            v[0] = (x0 + t3) >> 10;
            v[56] = (x0 - t3) >> 10;
            v[8] = (x1 + t2) >> 10;
            v[48] = (x1 - t2) >> 10;
            v[16] = (x2 + t1) >> 10;
            v[40] = (x2 - t1) >> 10;
            v[24] = (x3 + t0) >> 10;
            v[32] = (x3 - t0) >> 10;
        }
    }
    v = val;
    uint8_t* o = out;
    for (int i = 0; i < 8; ++i, v += 8, o += out_stride)
    {
        FAISCA_IDCT_1D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])
        x0 += 65536 + (128 << 17);
        x1 += 65536 + (128 << 17);
        x2 += 65536 + (128 << 17);
        x3 += 65536 + (128 << 17);
        o[0] = clamp_u8((x0 + t3) >> 17);
        o[7] = clamp_u8((x0 - t3) >> 17);
        o[1] = clamp_u8((x1 + t2) >> 17);
        o[6] = clamp_u8((x1 - t2) >> 17);
        o[2] = clamp_u8((x2 + t1) >> 17);
        o[5] = clamp_u8((x2 - t1) >> 17);
        o[3] = clamp_u8((x3 + t0) >> 17);
        o[4] = clamp_u8((x3 - t0) >> 17);
    }
}

#undef FAISCA_IDCT_1D

bool build_jpeg_huffman(const uint8_t* counts, const uint8_t* values, size_t n_values, JpegHuffman& h)
{
    h.count = static_cast<uint16_t>(n_values);
    std::memcpy(h.values, values, n_values);
    int code = 0;
    int k = 0;
    for (int len = 1; len <= 16; ++len)
    {
        h.valptr[len] = k;
        h.mincode[len] = code;
        for (int i = 0; i < counts[len]; ++i)
        {
            h.codes[k++] = static_cast<uint16_t>(code++);
        }
        h.maxcode[len] = counts[len] != 0 ? code - 1 : -1;
        code <<= 1;
        if (code > (1 << (len + 1)) && len < 16)
        {
            return false;
        }
    }
    h.maxcode[17] = 0x7fffffff;
    return k == static_cast<int>(n_values);
}

bool decode_jpeg_block(JpegReader& r, JpegComponent& c, const JpegHuffman& dc, const JpegHuffman& ac, const uint16_t* dequant,
                       short data[64])
{
    const int t = jpeg_decode(r, dc);
    if (t < 0 || t > 15)
    {
        return false;
    }
    std::memset(data, 0, 64 * sizeof(short));
    const int diff = t != 0 ? jpeg_extend(r, t) : 0;
    c.dc_pred += diff;
    data[0] = static_cast<short>(c.dc_pred * dequant[0]);
    int k = 1;
    do
    {
        const int rs = jpeg_decode(r, ac);
        if (rs < 0)
        {
            return false;
        }
        const int s = rs & 15;
        const int run = rs >> 4;
        if (s == 0)
        {
            if (rs != 0xf0)
            {
                break;
            }
            k += 16;
        }
        else
        {
            k += run;
            if (k > 63)
            {
                return false;
            }
            const int zig = DEZIGZAG[k++];
            data[zig] = static_cast<short>(jpeg_extend(r, s) * dequant[zig]);
        }
    } while (k < 64);
    return true;
}

inline uint8_t div4(int x)
{
    return static_cast<uint8_t>(x >> 2);
}

inline uint8_t div16(int x)
{
    return static_cast<uint8_t>(x >> 4);
}

void resample_hv_2(uint8_t* out, const uint8_t* near, const uint8_t* far, int w)
{
    if (w == 1)
    {
        out[0] = out[1] = div4(3 * near[0] + far[0] + 2);
        return;
    }
    int t1 = 3 * near[0] + far[0];
    out[0] = div4(t1 + 2);
    for (int i = 1; i < w; ++i)
    {
        const int t0 = t1;
        t1 = 3 * near[i] + far[i];
        out[i * 2 - 1] = div16(3 * t0 + t1 + 8);
        out[i * 2] = div16(3 * t1 + t0 + 8);
    }
    out[w * 2 - 1] = div4(t1 + 2);
}

void resample_h_2(uint8_t* out, const uint8_t* in, int w)
{
    if (w == 1)
    {
        out[0] = out[1] = in[0];
        return;
    }
    out[0] = in[0];
    out[1] = div4(in[0] * 3 + in[1] + 2);
    int i = 1;
    for (; i < w - 1; ++i)
    {
        const int n = 3 * in[i] + 2;
        out[i * 2] = div4(n + in[i - 1]);
        out[i * 2 + 1] = div4(n + in[i + 1]);
    }
    out[i * 2] = div4(in[w - 2] * 3 + in[w - 1] + 2);
    out[i * 2 + 1] = in[w - 1];
}

void resample_v_2(uint8_t* out, const uint8_t* near, const uint8_t* far, int w)
{
    for (int i = 0; i < w; ++i)
    {
        out[i] = div4(3 * near[i] + far[i] + 2);
    }
}

void resample_generic(uint8_t* out, const uint8_t* in, int w, int hs)
{
    for (int i = 0; i < w; ++i)
    {
        for (int j = 0; j < hs; ++j)
        {
            out[i * hs + j] = in[i];
        }
    }
}

inline int float2fixed(float x)
{
    return static_cast<int>(x * 4096.0f + 0.5f) << 8;
}

void ycbcr_to_rgb(uint8_t* out, const uint8_t* y, const uint8_t* pcb, const uint8_t* pcr, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        const int y_fixed = (y[i] << 20) + (1 << 19);
        const int cr = pcr[i] - 128;
        const int cb = pcb[i] - 128;
        int r = y_fixed + cr * float2fixed(1.40200f);
        int g = y_fixed + (cr * -float2fixed(0.71414f)) + ((cb * -float2fixed(0.34414f)) & 0xffff0000);
        int b = y_fixed + cb * float2fixed(1.77200f);
        r >>= 20;
        g >>= 20;
        b >>= 20;
        out[0] = clamp_u8(r);
        out[1] = clamp_u8(g);
        out[2] = clamp_u8(b);
        out += 3;
    }
}

bool decode_jpeg(const uint8_t* data, size_t size, Image& out, Error& error)
{
    if (size < 4 || data[0] != 0xff || data[1] != 0xd8)
    {
        error.set("not a JPEG file");
        return false;
    }
    uint16_t dequant[4][64];
    JpegHuffman huff_dc[4];
    JpegHuffman huff_ac[4];
    bool has_dc[4] = {};
    bool has_ac[4] = {};
    JpegComponent comps[3];
    int n_comps = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    int restart_interval = 0;
    size_t pos = 2;
    bool frame = false;
    for (;;)
    {
        while (pos < size && data[pos] != 0xff)
        {
            ++pos;
        }
        while (pos < size && data[pos] == 0xff)
        {
            ++pos;
        }
        if (pos >= size)
        {
            error.set("truncated JPEG");
            return false;
        }
        const uint8_t marker = data[pos++];
        if (marker == 0xd9)
        {
            error.set("JPEG has no scan");
            return false;
        }
        if (marker >= 0xd0 && marker <= 0xd7)
        {
            continue;
        }
        if (pos + 2 > size)
        {
            error.set("truncated JPEG");
            return false;
        }
        const size_t length = (static_cast<size_t>(data[pos]) << 8) | data[pos + 1];
        if (length < 2 || pos + length > size)
        {
            error.set("bad JPEG segment");
            return false;
        }
        const uint8_t* seg = data + pos + 2;
        const size_t seg_len = length - 2;
        if (marker == 0xdb)
        {
            size_t i = 0;
            while (i < seg_len)
            {
                const int precision = seg[i] >> 4;
                const int table = seg[i] & 15;
                ++i;
                if (table > 3 || i + (precision ? 128 : 64) > seg_len)
                {
                    error.set("bad JPEG quantization table");
                    return false;
                }
                for (int k = 0; k < 64; ++k)
                {
                    const uint16_t value = precision ? static_cast<uint16_t>((seg[i + 2 * k] << 8) | seg[i + 2 * k + 1]) : seg[i + k];
                    dequant[table][DEZIGZAG[k]] = value;
                }
                i += precision ? 128 : 64;
            }
        }
        else if (marker == 0xc4)
        {
            size_t i = 0;
            while (i < seg_len)
            {
                const int kind = seg[i] >> 4;
                const int table = seg[i] & 15;
                ++i;
                if (kind > 1 || table > 3 || i + 16 > seg_len)
                {
                    error.set("bad JPEG huffman table");
                    return false;
                }
                uint8_t counts[17] = {};
                size_t total = 0;
                for (int k = 1; k <= 16; ++k)
                {
                    counts[k] = seg[i + k - 1];
                    total += counts[k];
                }
                i += 16;
                if (total > 256 || i + total > seg_len)
                {
                    error.set("bad JPEG huffman table");
                    return false;
                }
                JpegHuffman& h = kind == 0 ? huff_dc[table] : huff_ac[table];
                std::memcpy(h.lengths, counts, 17);
                if (!build_jpeg_huffman(counts, seg + i, total, h))
                {
                    error.set("bad JPEG huffman table");
                    return false;
                }
                (kind == 0 ? has_dc : has_ac)[table] = true;
                i += total;
            }
        }
        else if (marker == 0xdd)
        {
            if (seg_len < 2)
            {
                error.set("bad JPEG restart interval");
                return false;
            }
            restart_interval = (seg[0] << 8) | seg[1];
        }
        else if (marker == 0xc0 || marker == 0xc1)
        {
            if (seg_len < 6 || seg[0] != 8)
            {
                error.set("JPEG precision %d is not supported", seg_len >= 1 ? seg[0] : 0);
                return false;
            }
            height = (seg[1] << 8) | seg[2];
            width = (seg[3] << 8) | seg[4];
            n_comps = seg[5];
            if ((n_comps != 1 && n_comps != 3) || width == 0 || height == 0 || seg_len < 6 + static_cast<size_t>(n_comps) * 3)
            {
                error.set("JPEG with %d components is not supported", n_comps);
                return false;
            }
            for (int i = 0; i < n_comps; ++i)
            {
                comps[i].id = seg[6 + i * 3];
                comps[i].h = seg[7 + i * 3] >> 4;
                comps[i].v = seg[7 + i * 3] & 15;
                comps[i].tq = seg[8 + i * 3];
                if (comps[i].h < 1 || comps[i].h > 4 || comps[i].v < 1 || comps[i].v > 4 || comps[i].tq > 3)
                {
                    error.set("bad JPEG sampling factors");
                    return false;
                }
            }
            frame = true;
        }
        else if (marker == 0xc2 || (marker >= 0xc3 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc))
        {
            error.set("progressive or non baseline JPEG is not supported");
            return false;
        }
        else if (marker == 0xda)
        {
            if (!frame)
            {
                error.set("JPEG scan before frame");
                return false;
            }
            const int ns = seg[0];
            if (ns != n_comps || seg_len < 1 + static_cast<size_t>(ns) * 2 + 3)
            {
                error.set("JPEG scans with a subset of the components are not supported");
                return false;
            }
            for (int i = 0; i < ns; ++i)
            {
                const int id = seg[1 + i * 2];
                int which = -1;
                for (int k = 0; k < n_comps; ++k)
                {
                    if (comps[k].id == id)
                    {
                        which = k;
                    }
                }
                if (which < 0)
                {
                    error.set("bad JPEG scan component");
                    return false;
                }
                comps[which].td = seg[2 + i * 2] >> 4;
                comps[which].ta = seg[2 + i * 2] & 15;
                if (comps[which].td > 3 || comps[which].ta > 3 || !has_dc[comps[which].td] || !has_ac[comps[which].ta])
                {
                    error.set("JPEG scan uses a missing huffman table");
                    return false;
                }
            }
            pos += length;
            break;
        }
        pos += length;
    }

    int h_max = 1;
    int v_max = 1;
    for (int i = 0; i < n_comps; ++i)
    {
        h_max = comps[i].h > h_max ? comps[i].h : h_max;
        v_max = comps[i].v > v_max ? comps[i].v : v_max;
    }
    const size_t mcu_w = static_cast<size_t>(h_max) * 8;
    const size_t mcu_h = static_cast<size_t>(v_max) * 8;
    const size_t mcu_x = (width + mcu_w - 1) / mcu_w;
    const size_t mcu_y = (height + mcu_h - 1) / mcu_h;
    for (int i = 0; i < n_comps; ++i)
    {
        comps[i].w2 = mcu_x * comps[i].h * 8;
        comps[i].h2 = mcu_y * comps[i].v * 8;
        if (!comps[i].plane.resize_uninitialized(comps[i].w2 * comps[i].h2))
        {
            error.set("out of memory");
            return false;
        }
    }

    JpegReader r{data, size, pos};
    short block[64];
    int todo = restart_interval != 0 ? restart_interval : 0x7fffffff;
    for (size_t j = 0; j < mcu_y; ++j)
    {
        for (size_t i = 0; i < mcu_x; ++i)
        {
            for (int k = 0; k < n_comps; ++k)
            {
                JpegComponent& c = comps[k];
                for (int y = 0; y < c.v; ++y)
                {
                    for (int x = 0; x < c.h; ++x)
                    {
                        if (!decode_jpeg_block(r, c, huff_dc[c.td], huff_ac[c.ta], dequant[c.tq], block))
                        {
                            error.set("corrupt JPEG data");
                            return false;
                        }
                        const size_t x2 = (i * c.h + x) * 8;
                        const size_t y2 = (j * c.v + y) * 8;
                        idct_block(c.plane.data() + c.w2 * y2 + x2, c.w2, block);
                    }
                }
            }
            if (--todo <= 0)
            {
                if (r.bits < 24)
                {
                    r.fill();
                }
                if (!(r.marker_hit && r.marker >= 0xd0 && r.marker <= 0xd7))
                {
                    break;
                }
                r.pos += 1;
                r.reset();
                for (int k = 0; k < n_comps; ++k)
                {
                    comps[k].dc_pred = 0;
                }
                todo = restart_interval;
            }
        }
    }

    if (!out.rgb.resize_uninitialized(static_cast<size_t>(width) * height * 3))
    {
        error.set("out of memory");
        return false;
    }
    out.width = width;
    out.height = height;

    struct Resampler
    {
        int hs;
        int vs;
        int ystep;
        int ypos;
        size_t w_lores;
        const uint8_t* line0;
        const uint8_t* line1;
    };
    Resampler res[3];
    Array<uint8_t> linebuf[3];
    for (int k = 0; k < n_comps; ++k)
    {
        Resampler& s = res[k];
        s.hs = h_max / comps[k].h;
        s.vs = v_max / comps[k].v;
        s.ystep = s.vs >> 1;
        s.ypos = 0;
        s.w_lores = (width + s.hs - 1) / s.hs;
        s.line0 = s.line1 = comps[k].plane.data();
        if (!linebuf[k].resize_uninitialized(static_cast<size_t>(width) + 3))
        {
            error.set("out of memory");
            return false;
        }
    }
    const size_t comp_rows[3] = {(height + res[0].vs - 1) / static_cast<size_t>(res[0].vs),
                                 n_comps > 1 ? (height + res[1].vs - 1) / static_cast<size_t>(res[1].vs) : 0,
                                 n_comps > 2 ? (height + res[2].vs - 1) / static_cast<size_t>(res[2].vs) : 0};
    for (uint32_t yy = 0; yy < height; ++yy)
    {
        const uint8_t* rows[3];
        for (int k = 0; k < n_comps; ++k)
        {
            Resampler& s = res[k];
            const bool y_bot = s.ystep >= (s.vs >> 1);
            const uint8_t* near = y_bot ? s.line1 : s.line0;
            const uint8_t* far = y_bot ? s.line0 : s.line1;
            uint8_t* dst = linebuf[k].data();
            if (s.hs == 1 && s.vs == 1)
            {
                rows[k] = near;
            }
            else
            {
                rows[k] = dst;
                if (s.hs == 1 && s.vs == 2)
                {
                    resample_v_2(dst, near, far, static_cast<int>(s.w_lores));
                }
                else if (s.hs == 2 && s.vs == 1)
                {
                    resample_h_2(dst, near, static_cast<int>(s.w_lores));
                }
                else if (s.hs == 2 && s.vs == 2)
                {
                    resample_hv_2(dst, near, far, static_cast<int>(s.w_lores));
                }
                else
                {
                    resample_generic(dst, near, static_cast<int>(s.w_lores), s.hs);
                }
            }
            if (++s.ystep >= s.vs)
            {
                s.ystep = 0;
                s.line0 = s.line1;
                if (static_cast<size_t>(++s.ypos) < comp_rows[k])
                {
                    s.line1 += comps[k].w2;
                }
            }
        }
        uint8_t* dst = out.rgb.data() + static_cast<size_t>(yy) * width * 3;
        if (n_comps == 3)
        {
            ycbcr_to_rgb(dst, rows[0], rows[1], rows[2], width);
        }
        else
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                dst[x * 3] = dst[x * 3 + 1] = dst[x * 3 + 2] = rows[0][x];
            }
        }
    }
    return true;
}

}

bool decode_image(const uint8_t* data, size_t size, Image& out, Error& error)
{
    if (size >= 8 && std::memcmp(data, "\x89PNG", 4) == 0)
    {
        return decode_png(data, size, out, error);
    }
    if (size >= 2 && data[0] == 0xff && data[1] == 0xd8)
    {
        return decode_jpeg(data, size, out, error);
    }
    error.set("not a PNG or JPEG file");
    return false;
}

bool read_image(const char* path, Image& out, Error& error)
{
    MappedFile file;
    if (!file.open(path, error))
    {
        return false;
    }
    return decode_image(file.data(), file.size(), out, error);
}

bool resize_image(const Image& in, uint32_t width, uint32_t height, ResizeFilter filter, Image& out, Error& error)
{
    constexpr int PRECISION_BITS = 22;
    const double filter_support = filter == ResizeFilter::Bicubic ? 2.0 : 1.0;
    const auto filter_weight = [filter](double x) -> double
    {
        x = x < 0.0 ? -x : x;
        if (filter == ResizeFilter::Bilinear)
        {
            return x < 1.0 ? 1.0 - x : 0.0;
        }
        constexpr double a = -0.5;
        if (x < 1.0)
        {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        }
        if (x < 2.0)
        {
            return (((x - 5) * x + 8) * x - 4) * a;
        }
        return 0.0;
    };
    const auto weights_for = [&](uint32_t in_size, uint32_t out_size, Array<int32_t>& bounds, Array<int32_t>& weights,
                                 int& ksize) -> bool
    {
        const double scale = static_cast<double>(in_size) / out_size;
        const double filterscale = scale < 1.0 ? 1.0 : scale;
        const double support = filter_support * filterscale;
        ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
        if (!bounds.resize_uninitialized(static_cast<size_t>(out_size) * 2) ||
            !weights.resize_uninitialized(static_cast<size_t>(out_size) * ksize))
        {
            return false;
        }
        Array<double> pre;
        if (!pre.resize_uninitialized(static_cast<size_t>(ksize)))
        {
            return false;
        }
        for (uint32_t xx = 0; xx < out_size; ++xx)
        {
            const double center = (xx + 0.5) * scale;
            const double ss = 1.0 / filterscale;
            int xmin = static_cast<int>(center - support + 0.5);
            xmin = xmin < 0 ? 0 : xmin;
            int xmax = static_cast<int>(center + support + 0.5);
            xmax = xmax > static_cast<int>(in_size) ? static_cast<int>(in_size) : xmax;
            xmax -= xmin;
            double ww = 0.0;
            for (int x = 0; x < xmax; ++x)
            {
                const double w = filter_weight((x + xmin - center + 0.5) * ss);
                pre[static_cast<size_t>(x)] = w;
                ww += w;
            }
            for (int x = 0; x < xmax; ++x)
            {
                if (ww != 0.0)
                {
                    pre[static_cast<size_t>(x)] /= ww;
                }
            }
            for (int x = xmax; x < ksize; ++x)
            {
                pre[static_cast<size_t>(x)] = 0.0;
            }
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
            const double fxp = std::ldexp(1.0, PRECISION_BITS);
            for (int x = 0; x < ksize; ++x)
            {
                const double value = pre[static_cast<size_t>(x)];
                weights[static_cast<size_t>(xx) * ksize + x] = static_cast<int32_t>(value * fxp + (value < 0 ? -0.5 : 0.5));
            }
        }
        return true;
    };
    const auto clip8 = [](int v) -> uint8_t
    {
        return v < 0 ? 0 : v > 255 ? 255 : static_cast<uint8_t>(v);
    };

    Image temp;
    const Image* src = &in;
    if (width != in.width)
    {
        Array<int32_t> bounds;
        Array<int32_t> weights;
        int ksize = 0;
        if (!weights_for(in.width, width, bounds, weights, ksize) ||
            !temp.rgb.resize_uninitialized(static_cast<size_t>(width) * in.height * 3))
        {
            error.set("out of memory");
            return false;
        }
        temp.width = width;
        temp.height = in.height;
        for (uint32_t yy = 0; yy < in.height; ++yy)
        {
            const uint8_t* src_row = in.rgb.data() + static_cast<size_t>(yy) * in.width * 3;
            uint8_t* dst_row = temp.rgb.data() + static_cast<size_t>(yy) * width * 3;
            for (uint32_t xx = 0; xx < width; ++xx)
            {
                const int xmin = bounds[xx * 2];
                const int xcnt = bounds[xx * 2 + 1];
                const int32_t* k = weights.data() + static_cast<size_t>(xx) * ksize;
                const uint8_t* p = src_row + static_cast<size_t>(xmin) * 3;
                int32_t ss0 = 1 << (PRECISION_BITS - 1);
                int32_t ss1 = ss0;
                int32_t ss2 = ss0;
                for (int x = 0; x < xcnt; ++x)
                {
                    ss0 += p[0] * k[x];
                    ss1 += p[1] * k[x];
                    ss2 += p[2] * k[x];
                    p += 3;
                }
                dst_row[xx * 3] = clip8(ss0 >> PRECISION_BITS);
                dst_row[xx * 3 + 1] = clip8(ss1 >> PRECISION_BITS);
                dst_row[xx * 3 + 2] = clip8(ss2 >> PRECISION_BITS);
            }
        }
        src = &temp;
    }
    if (height == src->height)
    {
        if (src == &temp)
        {
            out.width = temp.width;
            out.height = temp.height;
            out.rgb.swap(temp.rgb);
            return true;
        }
        out.width = in.width;
        out.height = in.height;
        return out.rgb.resize_uninitialized(in.rgb.size()) && (std::memcpy(out.rgb.data(), in.rgb.data(), in.rgb.size()), true);
    }
    Array<int32_t> bounds;
    Array<int32_t> weights;
    int ksize = 0;
    const size_t row_elems = static_cast<size_t>(src->width) * 3;
    Array<int32_t> acc;
    if (!weights_for(src->height, height, bounds, weights, ksize) || !acc.resize_uninitialized(row_elems) ||
        !out.rgb.resize_uninitialized(row_elems * height))
    {
        error.set("out of memory");
        return false;
    }
    out.width = src->width;
    out.height = height;
    for (uint32_t yy = 0; yy < height; ++yy)
    {
        const int ymin = bounds[yy * 2];
        const int ycnt = bounds[yy * 2 + 1];
        const int32_t* k = weights.data() + static_cast<size_t>(yy) * ksize;
        for (size_t i = 0; i < row_elems; ++i)
        {
            acc[i] = 1 << (PRECISION_BITS - 1);
        }
        for (int y = 0; y < ycnt; ++y)
        {
            const uint8_t* src_row = src->rgb.data() + static_cast<size_t>(ymin + y) * row_elems;
            const int32_t w = k[y];
            for (size_t i = 0; i < row_elems; ++i)
            {
                acc[i] += src_row[i] * w;
            }
        }
        uint8_t* dst_row = out.rgb.data() + static_cast<size_t>(yy) * row_elems;
        for (size_t i = 0; i < row_elems; ++i)
        {
            dst_row[i] = clip8(acc[i] >> PRECISION_BITS);
        }
    }
    return true;
}

}
