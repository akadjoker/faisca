#pragma once

#include "faisca/array.h"
#include "faisca/error.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// 8 bit RGB, rows top to bottom, 3 bytes per pixel; alpha holds one byte per pixel for
// PNG files with an alpha channel (colour types 4 and 6) and is empty otherwise
struct Image
{
    uint32_t width = 0;
    uint32_t height = 0;
    Array<uint8_t> rgb;
    Array<uint8_t> alpha;
};

// PNG (8 and 16 bit, every colour type, no interlace, alpha kept apart) or baseline JPEG
// (Huffman, 1 or 3 components, any sampling, restart markers; progressive refused), decoded
// as stb_image does, so the reference sees the same pixels
bool read_image(const char* path, Image& out, Error& error);
bool decode_image(const uint8_t* data, size_t size, Image& out, Error& error);

enum class ResizeFilter
{
    Bilinear, // support 1
    Bicubic // support 2, a = -0.5 (Pillow's value, mtmd's default)
};

// Pillow's resampling in fixed point, as mtmd resizes: separable, 22 fractional bits
bool resize_image(const Image& in, uint32_t width, uint32_t height, ResizeFilter filter, Image& out, Error& error);

}
