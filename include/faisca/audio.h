#pragma once

#include "faisca/array.h"
#include "faisca/error.h"

#include <cstddef>
#include <cstdint>

namespace faisca
{

constexpr size_t WHISPER_SAMPLE_RATE = 16000;
constexpr size_t WHISPER_N_FFT = 400;
constexpr size_t WHISPER_HOP = 160;
constexpr size_t WHISPER_N_FREQ = WHISPER_N_FFT / 2 + 1;
constexpr size_t WHISPER_CHUNK_SECONDS = 30;

// 16 bit PCM WAV at 16 kHz, channels averaged, samples in [-1, 1)
bool read_wav(const char* path, Array<float>& samples, Error& error);

// mono 16 bit PCM WAV, samples clamped to [-1, 1]
bool write_wav(const char* path, const float* samples, size_t count, uint32_t sample_rate, Error& error);

// log mel spectrogram as whisper.cpp computes it, bit for bit: filters is n_mel x 201 (from the
// model file), mel gets n_mel rows of n_len frames (mel[j * n_len + i]); returns n_len
size_t log_mel_spectrogram(const float* samples, size_t n_samples, const float* filters, size_t n_mel,
                           Array<float>& mel, Error& error);

}
