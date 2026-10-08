#pragma once

#include "faisca/array.h"
#include "faisca/error.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

// 8 bit RGB, rows top to bottom, 3 bytes per pixel
struct Image
{
    uint32_t width = 0;
    uint32_t height = 0;
    Array<uint8_t> rgb;
};

// PNG (8 and 16 bit, every colour type, no interlace, alpha dropped) or baseline JPEG
// (Huffman, 1 or 3 components, any sampling, restart markers; progressive refused), decoded
// as stb_image does, so the reference sees the same pixels
bool read_image(const char* path, Image& out, Error& error);
bool decode_image(const uint8_t* data, size_t size, Image& out, Error& error);

// Pillow's bilinear resampling in fixed point, as mtmd resizes: separable, 22 fractional bits
bool resize_bilinear(const Image& in, uint32_t width, uint32_t height, Image& out, Error& error);

}
