#include "faisca/audio.h"

#include "faisca/mapped_file.h"

#include <cmath>
#include <cstring>

namespace faisca
{

namespace
{

constexpr size_t SIN_COS_N = WHISPER_N_FFT;

struct Tables
{
    float sin_vals[SIN_COS_N];
    float cos_vals[SIN_COS_N];
    float hann[WHISPER_N_FFT];

    Tables()
    {
        for (size_t i = 0; i < SIN_COS_N; ++i)
        {
            const double theta = (2 * M_PI * static_cast<double>(i)) / static_cast<double>(SIN_COS_N);
            sin_vals[i] = sinf(static_cast<float>(theta));
            cos_vals[i] = cosf(static_cast<float>(theta));
        }
        for (size_t i = 0; i < WHISPER_N_FFT; ++i)
        {
            hann[i] = static_cast<float>(0.5 * (1.0 - cosf(static_cast<float>((2.0 * M_PI * static_cast<double>(i)) / static_cast<double>(WHISPER_N_FFT)))));
        }
    }
};

const Tables& tables()
{
    static const Tables instance;
    return instance;
}

// the same recursion as whisper.cpp: in holds 2N floats of scratch, out 8N
void dft(const float* in, int n, float* out)
{
    const Tables& t = tables();
    const int step = static_cast<int>(SIN_COS_N) / n;
    for (int k = 0; k < n; ++k)
    {
        float re = 0;
        float im = 0;
        for (int i = 0; i < n; ++i)
        {
            const int idx = (k * i * step) % static_cast<int>(SIN_COS_N);
            re += in[i] * t.cos_vals[idx];
            im -= in[i] * t.sin_vals[idx];
        }
        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

void fft(float* in, int n, float* out)
{
    if (n == 1)
    {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    const int half = n / 2;
    if (n - half * 2 == 1)
    {
        dft(in, n, out);
        return;
    }
    float* even = in + n;
    for (int i = 0; i < half; ++i)
    {
        even[i] = in[2 * i];
    }
    float* even_fft = out + 2 * n;
    fft(even, half, even_fft);

    float* odd = even;
    for (int i = 0; i < half; ++i)
    {
        odd[i] = in[2 * i + 1];
    }
    float* odd_fft = even_fft + n;
    fft(odd, half, odd_fft);

    const Tables& t = tables();
    const int step = static_cast<int>(SIN_COS_N) / n;
    for (int k = 0; k < half; ++k)
    {
        const int idx = k * step;
        const float re = t.cos_vals[idx];
        const float im = -t.sin_vals[idx];
        const float re_odd = odd_fft[2 * k + 0];
        const float im_odd = odd_fft[2 * k + 1];
        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

uint32_t read_u32(const uint8_t* p)
{
    uint32_t v = 0;
    std::memcpy(&v, p, 4);
    return v;
}

uint16_t read_u16(const uint8_t* p)
{
    uint16_t v = 0;
    std::memcpy(&v, p, 2);
    return v;
}

}

bool read_wav(const char* path, Array<float>& samples, Error& error)
{
    MappedFile file;
    if (!file.open(path, error))
    {
        return false;
    }
    const uint8_t* bytes = file.data();
    const size_t size = file.size();
    if (size < 12 || std::memcmp(bytes, "RIFF", 4) != 0 || std::memcmp(bytes + 8, "WAVE", 4) != 0)
    {
        error.set("'%s' is not a WAV file", path);
        return false;
    }
    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t rate = 0;
    uint16_t bits = 0;
    size_t pos = 12;
    while (pos + 8 <= size)
    {
        const uint32_t chunk_size = read_u32(bytes + pos + 4);
        const uint8_t* chunk = bytes + pos + 8;
        const size_t available = size - pos - 8;
        if (std::memcmp(bytes + pos, "fmt ", 4) == 0 && chunk_size >= 16 && available >= 16)
        {
            format = read_u16(chunk);
            channels = read_u16(chunk + 2);
            rate = read_u32(chunk + 4);
            bits = read_u16(chunk + 14);
        }
        else if (std::memcmp(bytes + pos, "data", 4) == 0)
        {
            if (format != 1 || bits != 16 || channels == 0)
            {
                error.set("'%s': only 16 bit PCM WAV is read (format %u, %u bits)", path, format, bits);
                return false;
            }
            if (rate != WHISPER_SAMPLE_RATE)
            {
                error.set("'%s': %u Hz, the models take 16000 Hz (resample first)", path, rate);
                return false;
            }
            const size_t data_size = chunk_size < available ? chunk_size : available;
            const size_t frames = data_size / (2 * channels);
            if (!samples.resize_uninitialized(frames))
            {
                error.set("out of memory");
                return false;
            }
            const float scale = 1.0f / (32768.0f * static_cast<float>(channels));
            for (size_t i = 0; i < frames; ++i)
            {
                int32_t sum = 0;
                for (uint16_t c = 0; c < channels; ++c)
                {
                    sum += static_cast<int16_t>(read_u16(chunk + (i * channels + c) * 2));
                }
                samples[i] = static_cast<float>(sum) * scale;
            }
            return true;
        }
        pos += 8 + static_cast<size_t>(chunk_size) + (chunk_size & 1);
    }
    error.set("'%s': no data chunk", path);
    return false;
}

size_t log_mel_spectrogram(const float* samples, size_t n_samples, const float* filters, size_t n_mel,
                           Array<float>& mel, Error& error)
{
    const size_t frame_size = WHISPER_N_FFT;
    const size_t frame_step = WHISPER_HOP;
    const size_t stage_1_pad = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SECONDS;
    const size_t stage_2_pad = frame_size / 2;

    if (n_samples > (static_cast<size_t>(1) << 31))
    {
        error.set("audio too long (%zu samples)", n_samples);
        return 0;
    }
    Array<float> padded;
    if (!padded.resize_uninitialized(n_samples + stage_1_pad + stage_2_pad * 2))
    {
        error.set("out of memory");
        return 0;
    }
    std::memset(padded.data(), 0, padded.size() * sizeof(float));
    std::memcpy(padded.data() + stage_2_pad, samples, n_samples * sizeof(float));
    const size_t n_reflect = n_samples > 0 ? (stage_2_pad < n_samples - 1 ? stage_2_pad : n_samples - 1) : 0;
    for (size_t i = 0; i < n_reflect; ++i)
    {
        padded[stage_2_pad - n_reflect + i] = samples[n_reflect - i];
    }

    const size_t n_len = (padded.size() - frame_size) / frame_step;
    if (!mel.resize_uninitialized(n_mel * n_len))
    {
        error.set("out of memory");
        return 0;
    }
    const size_t n_valid = n_samples + stage_2_pad;
    const Tables& t = tables();
    float fft_in[WHISPER_N_FFT * 2];
    float fft_out[WHISPER_N_FFT * 8];
    size_t i = 0;
    const size_t frames = n_valid / frame_step + 1 < n_len ? n_valid / frame_step + 1 : n_len;
    for (; i < frames; ++i)
    {
        const size_t offset = i * frame_step;
        const size_t count = n_valid - offset < frame_size ? n_valid - offset : frame_size;
        for (size_t j = 0; j < count; ++j)
        {
            fft_in[j] = t.hann[j] * padded[offset + j];
        }
        for (size_t j = count; j < frame_size * 2; ++j)
        {
            fft_in[j] = 0.0f;
        }
        fft(fft_in, static_cast<int>(frame_size), fft_out);
        for (size_t j = 0; j < WHISPER_N_FREQ; ++j)
        {
            fft_out[j] = fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }
        for (size_t j = 0; j < n_mel; ++j)
        {
            const float* f = filters + j * WHISPER_N_FREQ;
            double sum = 0.0;
            size_t k = 0;
            for (; k + 3 < WHISPER_N_FREQ; k += 4)
            {
                sum += fft_out[k + 0] * f[k + 0] + fft_out[k + 1] * f[k + 1] + fft_out[k + 2] * f[k + 2] + fft_out[k + 3] * f[k + 3];
            }
            for (; k < WHISPER_N_FREQ; ++k)
            {
                sum += fft_out[k] * f[k];
            }
            sum = log10(sum > 1e-10 ? sum : 1e-10);
            mel[j * n_len + i] = static_cast<float>(sum);
        }
    }
    const float silence = static_cast<float>(log10(1e-10));
    for (; i < n_len; ++i)
    {
        for (size_t j = 0; j < n_mel; ++j)
        {
            mel[j * n_len + i] = silence;
        }
    }

    double mmax = -1e20;
    for (size_t k = 0; k < mel.size(); ++k)
    {
        if (mel[k] > mmax)
        {
            mmax = mel[k];
        }
    }
    mmax -= 8.0;
    for (size_t k = 0; k < mel.size(); ++k)
    {
        if (mel[k] < mmax)
        {
            mel[k] = static_cast<float>(mmax);
        }
        mel[k] = static_cast<float>((mel[k] + 4.0) / 4.0);
    }
    return n_len;
}

}
