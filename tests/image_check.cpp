// Decodes images with faisca and compares the pixels with a raw RGB dump (u32 width, u32
// height, then width x height x 3 bytes) written by stb_image, the decoder of the reference.
//
// usage: image_check <image> <dump.rgb> [<image> <dump.rgb> ...]

#include "faisca/image.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv)
{
    int failures = 0;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        faisca::Error error;
        faisca::Image image;
        if (!faisca::read_image(argv[i], image, error))
        {
            std::printf("FAIL %s: %s\n", argv[i], error.c_str());
            ++failures;
            continue;
        }
        FILE* f = std::fopen(argv[i + 1], "rb");
        uint32_t w = 0;
        uint32_t h = 0;
        if (f == nullptr || std::fread(&w, 4, 1, f) != 1 || std::fread(&h, 4, 1, f) != 1)
        {
            std::printf("FAIL %s: cannot read %s\n", argv[i], argv[i + 1]);
            ++failures;
            continue;
        }
        std::vector<uint8_t> ref(static_cast<size_t>(w) * h * 3);
        const size_t got = std::fread(ref.data(), 1, ref.size(), f);
        std::fclose(f);
        if (w != image.width || h != image.height || got != ref.size())
        {
            std::printf("FAIL %s: %u x %u, reference %u x %u\n", argv[i], image.width, image.height, w, h);
            ++failures;
            continue;
        }
        size_t differing = 0;
        int worst = 0;
        for (size_t k = 0; k < ref.size(); ++k)
        {
            const int d = std::abs(static_cast<int>(image.rgb[k]) - static_cast<int>(ref[k]));
            differing += d != 0;
            worst = d > worst ? d : worst;
        }
        std::printf("%s %s: %u x %u, %zu of %zu bytes differ, worst %d\n", worst == 0 ? "ok  " : "FAIL", argv[i], w, h,
                    differing, ref.size(), worst);
        failures += worst != 0;
    }
    return failures == 0 ? 0 : 1;
}
