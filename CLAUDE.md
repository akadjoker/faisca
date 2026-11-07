# faisca

LLM runner for small models, written from scratch in C++17. Library first, CLI on
top. No third party dependencies. Owner: Luís (akadjoker). "faisca" is a working name.

## Rules (set by the owner, follow them without being asked again)

- Opening brace always on its own line.
- Minimal or no comments. None at the top of files. A comment only when it
  explains something specific that the code does not show.
- No smart pointers. No std containers, std::string or exceptions in the library
  and CLI. Use `faisca::Array`, `faisca::StringMap`, `faisca::IntMap`,
  `faisca::Arena`, `faisca::Error`. Tests and benchmarks may use the standard library.
- Speed matters: write own data structures when they are faster, and prove it
  with a real benchmark on real data against the alternatives (see bench/).
- When investigating, never fill gaps with assumptions. Say "precisamos de mais
  dados" or "precisamos de mais investigação" instead.
- When there is a work plan, follow it. No shortcuts to finish a task sooner.
- Windows support is required (MinGW tested under Wine; MSVC not built yet).
- Talk to the owner in European Portuguese, without em dashes. Code, comments
  and README in English.
- Every piece is validated against a reference before moving on.

## Plan

1. GGUF loader: done (validated against gguf-py and ggml, 16x faster than ggml).
2. Tokenizer: done, see below.
3. CPU runner in float32: done, validated layer by layer against llama.cpp,
   see below.
4. CPU optimization: in progress. Done: Q8_0, Q4_0, Q5_0, Q4_K and Q6_K
   packed VNNI kernels (AVX-512), thread pool, batched prompt, tiled
   attention, KV cache in f16, software prefetch in the generation kernels.
   AMX for the prompt was measured (scratch bench_amx) and set aside: the
   per block float epilogue bounds it at 1.25x over VNNI on the GEMM, about
   10% of the prompt (bench/README.md). AVX-VNNI (256-bit) kernels for
   Q5_0, Q4_K and Q6_K: done, the AVX-VNNI build is bit identical to the
   AVX-512 one on every model (the softmax denominator of the attention is
   16 partial sums reduced by one fixed tree in both; before, the two
   builds differed by ulps there, invisible under f16 scales and visible
   under the float32 scale of Q8_K). The 256-bit kernels are GEMM
   templates like the AVX-512 ones: each chunk unpacked once per group
   block and reused for GEMM_TOKENS_AVX = 4 inputs (16 ymm registers:
   two halves of the sums for 4 inputs plus the chunk), the tail and
   generation through the same template with 1 input: the prompt is 1.2x
   (Q4_0) to 2x (1.7B) over the per token kernels and 1.5 to 1.7x below
   AVX-512, generation unchanged; 8 inputs spill and lose 10 to 25%
   (bench/README.md).
   Speculative decoding with SmolLM2-135M as draft: done (`Speculative`,
   `--draft`), output identical to plain greedy, gain 1.3 to 1.4x on one
   thread with a Q4_0 draft and about 1.1x with four, 0.9x at a 512 token
   context; the draft costs 0.3 to 0.4 of the target per token, which caps
   it (bench/README.md). Then Vulkan; then the cache format.
5. Vulkan compute backend (the owner chose Vulkan over CUDA and OpenGL): in
   progress. Done: VulkanDevice (loader opened at run time with dlopen /
   LoadLibrary, instance, compute queue, buffers with staging, pipelines,
   recorded dispatches with a memory barrier after each), VulkanKernels
   with matvec_q8_0 and matvec_f32 (GLSL in src/vulkan/shaders, SPIR-V
   embedded by tools/embed_spirv.py, generated file committed),
   tests/vulkan_check (1.3e-7 to 1.8e-7 against double precision, 1 and 16
   inputs), the other ops (embed, rms_norm, rope, kv_store, attention,
   swiglu, add) and VulkanRunner (include/faisca/vulkan_runner.h: weights
   uploaded from the GgufFile: Q8_0, Q4_0 and Q5_0 split into packed f16
   scales and uints of quants, Q4_K as the file's 36 uint blocks, Q6_K as
   53 uint blocks (2 bytes of padding), each with its own matvec shader
   that dequantizes as it reads, F16 as float32; the embedding gather
   reads Q8_0 or float32 only, so other embedding types convert; float32
   activations and KV cache; one command buffer per batch, submitted and
   downloaded after every traced step when a trace is set; `--vulkan` in
   the CLI and runner_dump). Validated with compare_layers
   (`--ours-flags=--vulkan`, `--reference-model` for a different file on
   the reference side): float32 model 1.8e-6 from llama.cpp and 3.4e-6
   from the CPU runner, Q8_0 / Q4_0 / Q4_K_M models 2.0e-6 / 1.9e-6 /
   2.3e-6 from llama.cpp on their float32 copies (the GPU does not quantize
   activations, so it is closer to float32 than the CPU quantized paths;
   against llama.cpp Q8_0 it is 8e-2 and the near tie token flips), 16
   tokens identical; vulkan_check per matrix 1.2e-7 to 5e-7 for every
   type (`--all-down` covers the Q4_K layers). Only llvmpipe (Mesa CPU, Vulkan
   1.4, subgroup 8) is available here, so correctness only; speed needs the
   owner's GPU. Next: speed on a real GPU (the kernels are simple: one
   workgroup per row, scalar loads; expected first optimizations: several
   rows per workgroup, vectorized loads, f16 cache, fewer dispatches),
   Windows loading path untested.
6. Own cache format with offset tables, KV cache kept between turns and saved to
   disk, longest common prefix reuse, snapshots: done. `Runner::state_bytes /
   export_state / import_state` (positions 0..pos: per layer the key tiles
   covering them, then the value rows), `Session` (include/faisca/session.h:
   tokens by position, `feed` with longest common prefix reuse capped at
   count - 1 so the last token always runs, `push`, `truncate`, `snapshot /
   restore` in memory, `save / load`: "FSCA" header, version 1, n_layer,
   n_head_kv, head_dim, kv_f16, n_vocab, n_tokens, model name, offset table
   of {kind, layer, offset, size} with the tokens then keys and values per
   layer, sections 8 byte aligned; another model, cache type or version is
   refused), `faisca chat` with `--session`. tests/session_check: prefix
   reuse, cut at a divergence, restore and load in a fresh runner all give
   logits bit identical to an uninterrupted run (Q8_0 and Q4_K_M, ASan
   clean). A 19 token snapshot of SmolLM2-360M in f16 is 1 MiB (whole key
   tiles of 16 positions are stored). The Vulkan runner has no state
   export yet.

7. Whisper (speech to text), the owner's priority over more CPU work. Done
   in float32: `WhisperModel` (include/faisca/whisper_model.h: whisper.cpp's
   ggml file, f16 converted at load, vocabulary as raw bytes, special ids
   derived from n_vocab as whisper.cpp does, 99 language codes), `read_wav`
   and `log_mel_spectrogram` (audio.h, bit identical to whisper.cpp: same
   Hann, sin/cos tables, FFT recursion, double filter sums, clamp),
   float32 ops (layer_norm, gelu tanh form, conv1d_k3, attention_rows),
   `WhisperRunner` (encoder, cross K scaled by head_dim^-0.25 and V per
   decoder layer, decoder with self KV cache, Q and K scaled by
   head_dim^-0.25, prompt [sot, lang, transcribe, no_timestamps], the
   logit rules of whisper_process_logits without timestamps, greedy up to
   n_text_ctx / 2 - 4 tokens, 30 s windows), `faisca transcribe`. Reference:
   /home/user/ref/whisper.cpp 1.9.5 patched (whisper_set_eval_callback on
   every sched, named tensors e_*/d_*/cross_*/conv1/logits, and a
   WHISPER_F32_REFERENCE build: itype F32, no GELU f16 table, conv im2col
   F32), tests/reference/whisper_layer_dump.cpp, tests/make_f32_whisper.py,
   tests/whisper_dump.cpp, tests/compare_whisper.py. Validated on jfk.wav
   and FLEURS pt_br clips (/home/user/ref/audio): tiny / base / small, every
   traced tensor at worst 1.4e-5 / 1.8e-5 / 2.2e-5 rel L2 from the f32
   reference, tokens identical in all runs, 1 and 4 threads bit identical,
   ASan/UBSan clean; stock whisper.cpp (f16 cache, GELU table) is 3e-3 from
   ours with the same tokens. The matrices are quantized to Q8_0 and packed
   at load (`WhisperModel::load(path, error, quantize)`, `--f32` keeps the
   file's values): noise 1.6e-1 at the last encoder GELU and 8e-2 at the
   logits, small and base keep the tokens, tiny flips a near tie on jfk
   ("you, ask"), the same flip whisper-cli makes with its own Q8_0 file
   (/home/user/models/whisper/ggml-*-q8_0.bin). The encoder and cross
   attention run through attention_head on keys tiled per head (15 of
   20.7 s before on small, 1 thread); the causal decoder self attention
   keeps attention_rows. Speed on jfk.wav (11 s), 1 / 4 threads: tiny 1.47
   / 0.48 s, base 4.35 / 1.14, small 9.45 / 3.25 (whisper-cli f16 2.09 /
   0.85, 4.29 / 1.47, 13.1 / 4.03; its Q8_0 files 1.68 / 0.72, 3.19 /
   1.11, 8.69 / 2.89); the load (f16 to float32, quantize, pack, one
   thread) is 0.26 to 1.6 s. Phases in bench/README.md: next the scalar
   conv1d (1.5 s of 9.5 on small) and the tanhf GELU (1.1 s). Not done:
   timestamp tokens and cutting windows at pauses (now every 30 s), the
   initial prompt from the previous window, language detection, medium
   and large files (should load, not tried), WAV only (ffmpeg converts).
   Quality (Whisper paper, LibriSpeech WER): tiny 7.6, base 5.0, small 3.4,
   medium 2.9; on the three Portuguese clips only small gets pt_0 exactly,
   and the training data is mostly Brazilian, so European Portuguese needs
   our own measurement with the owner's recordings.
8. Text to speech: done with Pocket TTS (Kyutai, 100M, English and
   Portuguese packs; the owner heard samples of Pocket, OuteTTS and Piper
   and the Portuguese voices are all Brazilian, so English quality decided).
   Reference: /home/user/ref/tts/pockettts.cpp (ggml port with `--dump`),
   built stock in `build` and as `build-f32` with PT_F32_REFERENCE (no GELU
   f16 table, im2col in float32, conv kernels kept F32) plus a patch that
   names the step and prefill nodes (dbg_attn, dbg_attn_out, dbg_res_attn,
   dbg_norm2, dbg_up, dbg_down, dbg_res_ffn per layer, set as outputs) and
   writes them with PT_DUMP_NODES=<dir>. Packs in /home/user/models/tts
   (pocket-tts-english_2026-09 and pocket-tts-portuguese, F32, Q8_0 and
   voices GGUF). Library: `PocketTokenizer` (SentencePiece unigram, Viterbi
   per space delimited segment, byte pieces), `pocket_prepare_text` and
   `pocket_chunk_text` (text_chunking.py: rules from the GGUF keys,
   chunks of at most 50 tokens cut at sentence then clause boundaries),
   `PocketModel` (include/faisca/pocket_model.h: flow LM, flow head, Mimi
   decoder and encoder weights; Q8_0 matrices packed, F32 ones quantized
   unless `quantize` is false, which also dequantizes a Q8_0 pack;
   transposed conv kernels stored per output phase as 2 tap causal convs),
   `PocketVoice` (the `<name>.k` / `.v` rows of the voices GGUF),
   `TorchRandom` (MT19937, torch's uniform from 24 bits and normal_ by
   Box-Muller over blocks of 16), `PocketRunner` (pocket_runner.h: voice
   rows written to a float32 KV cache, text prefill, one step per 80 ms
   frame: input_linear, 6 layers, out_norm, EOS logit, flow head with the
   two averaged time embeddings and n LSD steps, noise seeded once per
   synthesis with one discarded draw after each prefill; the chunk's
   latents decoded at once by Mimi: quantizer projection, depthwise
   transposed conv, 2 windowed transformer layers (window 250, layer
   scales), SEANet decoder with ELU; first 1920 samples of every chunk
   dropped), `write_wav`, `faisca speak`. Validated with tests/pocket_dump
   and tests/compare_pocket.py against the f32 reference on "The quick
   brown fox jumps over the lazy dog.", voice alba, seed 1: prompt ids and
   text embeddings identical, step 0 hidden 5.9e-7, EOS logits 2.9e-7,
   37 latents 5.1e-6, audio 4.0e-5 rel L2, same 38 steps; a three chunk
   text gives the same 51 / 74 / 52 frames; 1 and 4 threads bit
   identical; ASan/UBSan clean; the plain AVX2 build gives the same
   float32 numbers (the AVX-VNNI build cannot run on this VM, no avx_vnni
   flag). With the Q8_0 pack both runners are 1.3e-2 to 1.5e-2 from
   float32 at step 0 and 1.2e-2 from each other: the activations differ
   by 1 ulp and one Q8_0 byte of the layer 0 attention output flips at a
   rounding tie (x times id = 72.5), after which the autoregressive loop
   keeps the two apart (the matmuls agree to 1e-7 on identical input, 1
   and 14 tokens); the reference's own stock and f32 builds are as far
   apart on the Q8_0 pack. Speed (bench/README.md): 13.9 s of audio in
   6.8 / 3.7 / 2.2 s with 1 / 2 / 4 threads, pocket-tts 6.9 s with its 2
   threads; the SEANet convolutions run through `conv1d_rows`
   (src/ops_conv.cpp, 4 rows x 64 columns in AVX-512 registers, 4 x 32 in
   AVX2, scalar edges), which took the decode from 108 to 22 ms per frame
   on one thread. Not done: voice cloning from a recording (the Mimi
   encoder weights load, the encoder does not run), streaming output,
   `--temperature` and the other pocket-tts knobs, `pt_PT` quality
   (precisamos de gravações do owner para medir).
9. Qwen2.5, Qwen3 and Gemma 3: done. `LlamaHparams::arch` (llama, qwen2,
   qwen3, gemma3) with the flags the runner reads: rope_neox (pairs i and
   i + n_rot / 2, `rope_neox` in ops), qk_norm (RMS norm of every Q and K
   head before rope, qwen3 and gemma3), post_norms (gemma3's
   post_attention_norm and post_ffw_norm before the residual adds),
   gelu_ffn (`geglu`), embd_scale (sqrt(n_embd)), attn_scale (gemma3: Q
   scaled by 1 / sqrt(head_dim) after rope, traced before, scores not
   scaled), n_swa / swa_pattern (gemma3: 512, layer il local when il % 6
   != 5, rope base 10000 on the local layers), final_softcap (0 on the 1B
   file); head_dim from attention.key_length (qwen3: 128 with n_embd 1024,
   gemma3: 256); Q, K and V biases on qwen2 (`LlamaLayer::bq`); the keys
   are read under the architecture's prefix. The sliding window reuses
   the tiled attention kernels: the window starts at a whole key tile and
   a `skip` argument (0 to 15) masks the positions before the start inside
   it, with the kernels bit identical when skip is 0 (ops_unit checks the
   skip against a double precision softmax). Tokenizers: `qwen2`
   pre-tokenizer (the llama 3 regex with one digit per piece, clean_spaces
   off; `split_llama3(max_digits)`), llama.cpp's promotion of the end of
   generation texts (<|im_end|>, </s>, <end_of_turn>, <eos>, ...) to
   control tokens whatever the file says, and SentencePiece
   (`tokenizer.ggml.model` llama: one symbol per UTF-8 character, the
   neighbouring pair whose joined text is the piece of highest score
   merged first, ties to the leftmost, what no piece covers as <0xXX>
   bytes; spaces as U+2581, add_space_prefix from the file, BOS by
   default, pieces unescaped on decode; `Tokenizer::Model`). The CLI
   picks the chat template by architecture: ChatML with Qwen2.5's default
   system prompt, plain ChatML for Qwen3 (it thinks before answering),
   <start_of_turn> for Gemma. The Vulkan runner refuses the new
   architectures. Models in /home/user/models/qwen and gemma
   (Qwen2.5-0.5B-Instruct Q8_0 from Qwen, Qwen3-0.6B Q8_0 from Qwen,
   gemma-3-1b-it Q8_0 from unsloth, float32 copies by make_f32_gguf.py).
   Reference llama.cpp b86d2f0 (2026-10-07) with the GELU f16 table
   commented out in ggml-cpu/vec.h (the stock build would put 1e-3 of
   noise in Gemma's FFN); `llama_layer_dump` also keeps inp_scaled,
   Qcur_normed, Kcur_normed, attn_post_norm, sa_out and ffn_post_norm.
   Validation: tokenizers 0 mismatches over 40062 cases (2.86 M tokens on
   the Qwen vocabulary, 3.26 M on Gemma's), SmolLM2 and Llama 3.2 still
   0; float32 path against llama.cpp: Qwen2.5-0.5B 266 tensors at worst
   5.5e-6, Qwen3-0.6B 366 at 1.9e-6, Gemma 3 1B 367 at 4.1e-6 on a 16
   token chat prompt and 3.6e-6 on a 600 token prompt (the window active
   on the local layers), 16 greedy tokens identical in every run; the
   Q8_0 files give the same 16 tokens as the float32 reference, 1 and 4
   threads bit identical, ASan/UBSan clean. llama.cpp's own Q8_0 path
   cannot run on this VM any more (its AMX repack aborts with "AMX is not
   ready to be used", SIGILL), so the quantized comparison is against the
   float32 reference only, and the speed table uses a llama.cpp built
   with GGML_AMX_INT8 off (bench/README.md): faisca is 1.1 to 1.7x faster
   on generation and prompts on the three models. The rest of the Gemma 3
   family (the owner wants to work with Gemma): linear rope scaling
   (`rope.scaling.type` linear with a factor gives `rope_freq_scale` =
   1 / factor, applied to the angle of every pair like ggml's rope_yarn
   with ext_factor 0, on the global layers only; the local layers keep
   1), so Gemma 3 4B (window 1024, factor 8) loads; gemma-3-270m-it and
   gemma-3-4b-it Q8_0 from unsloth in /home/user/models/gemma. 270M
   validated like the 1B: tokenizer 0 mismatches, float32 255 tensors at
   5.3e-6, 16 tokens identical, Q8_0 the same tokens as llama.cpp's Q8_0
   run, 1 and 4 threads identical, ASan clean. 4B has no float32 copy (17
   GB on a 15 GB machine), so it is compared Q8_0 against Q8_0: 479
   tensors at worst 6.1e-2 (the 1B gives 8.9e-2 the same way, the
   activation quantization noise seen on every model), logits 3.6e-2, 16
   tokens identical, 1 and 4 threads identical, ASan clean; its window of
   1024 was not exercised (a 1100 token comparison needs 2 GB of dumps
   per side, no disk), the 1B's 512 window at 600 tokens covers the code.
   The reference dump tools of build-bench link llama.cpp's shared libs,
   and the AMX ones abort on Q8_0 here: run them with
   LD_LIBRARY_PATH=/home/user/ref/llama.cpp/build-noamx/bin (the build
   with GGML_AMX_INT8, GGML_AMX_TILE and GGML_AMX_BF16 off). Speed,
   bench_runner against llama-bench without AMX, 1 / 4 threads: 270M
   generation 34 / 77 (llama.cpp 24 / 67), prompt 262 / 641 (221 / 643);
   4B generation 2.3 / 7.2 (1.7 / 6.0), prompt 18 / 59 (10 / 38); the 4B
   reads 3.9 GiB per token and its load packs them in 25 s on one thread,
   1.3 s with four. Not done: Gemma 3 12B and 27B (13 and 28 GB in Q8_0,
   beyond this machine; Q4_K_M files should load), Qwen larger than 0.6B
   (should load, not tried), the Vulkan runner for these architectures.
   Next: Mistral / Llama 2 with the SentencePiece tokenizer (needs the
   add_space_prefix path, which Gemma does not exercise).
10. Gemma 3 vision (the owner wants to work with Gemma; the 4B's SigLIP
   encoder and projector from the mmproj file): done. Image decoders
   written from scratch (include/faisca/image.h, src/image.cpp): PNG (own
   inflate, every colour type and bit depth, no interlace, alpha dropped)
   and baseline JPEG (Huffman, any sampling, restart markers, progressive
   refused) replicating stb_image's integer IDCT, fixed point YCbCr and
   fancy upsampling, bit identical to stb dumps on the test images
   (tests/image_check, /home/user/ref/images); `resize_image` is Pillow's
   separable fixed point resampling (22 bits, bilinear or bicubic),
   identical byte for byte to mtmd's img_tool::resize (checked with a
   harness that includes mtmd-image.cpp). `VisionModel`
   (vision_model.h: clip.* keys of the mmproj GGUF, F16 matrices
   quantized to Q8_0 and packed unless `quantize` is false, n_in padded to
   the Q8_0 block: the patch kernel 588 to 608, n_ff 4304 to 4320),
   `VisionRunner` (vision_runner.h: mtmd's fixed size preprocessing for
   gemma3, bilinear, PAD_CEIL, centred on black, mean and std 0.5, planar
   R G B; im2col patches in (channel, row, col) order; 27 pre-LN layers
   with biases, GELU, bidirectional attention over the 4096 patches with
   the keys tiled and the values stored per head, post LN, 4x4 average
   pool to 256 tokens, RMS norm with mm.soft_emb_norm, projection to
   2560; the padded FFN tail stays zero and the up projection is written
   with the padded stride, which was the first bug found), `Runner::
   forward_embeddings` (rows of input embeddings as one non causal block:
   every row sees the whole block and the positions before it, a local
   layer's window starts n_swa before each row, no embedding scale, the
   mask llama.cpp builds with llama_set_causal_attn(false) and
   LLAMA_NON_CAUSAL_TYPE_ALL; `reserve_batch` grows the token buffers to
   the block), `faisca describe <model> <mmproj> <image> [-p]` with the
   image before the user text as mtmd lays it out (<start_of_image>, 256
   embeddings, <end_of_image>). Reference: tests/reference/
   llama_vision_dump.cpp over libmtmd (build-noamx/bin/libmtmd.so, run
   with that LD_LIBRARY_PATH): the embeddings, `--trace` of the encoder
   through the eval callback with clip.cpp's names (`--layers N`, a whole
   trace is 8 GB), `--lm` for the language model on an image prompt
   (inp_embd, l_out, the greedy tokens); ours: tests/vision_dump
   (`--trace`, `--layers`, `--f32`, `--f16-patches`), tests/describe_dump,
   tests/compare_vision.py (cuts the padded FFN columns). Pitfalls met:
   ggml_conv_2d runs its im2col in f16 whatever the kernel type, so the
   reference rounds the image patches to f16 and its patch embedding is
   6e-5 from an exact one (`--f16-patches` rounds ours the same way for
   the comparison only); mtmd's default resize is bicubic but clip.cpp
   sets bilinear for gemma3; llama.cpp's image rows are not scaled by
   sqrt(n_embd) (scale_rows marks token rows only). Validation on
   test-1.png (the NYT moon landing page, 640 x 488): float32 encoder on
   a float32 copy of the mmproj (make_f32_gguf.py works on it) against
   the reference, f16 patches, layer 0 at worst 2.3e-6 (kqv_out) and the
   256 x 2560 embeddings at 1.7e-5 rel L2 after 27 layers (the f16 file
   with its f16 activations is 2.2e-3 away); Q8_0 encoder 7.7e-2 from the
   float32 reference (the quantization noise of 27 layers; llama.cpp's
   own quantized mmproj was not measured, precisamos de mais dados), 1
   and 4 threads bit identical, ASan/UBSan clean; language model on
   "<start_of_turn>user\n<image>Describe this image." against mtmd (Q8_0
   both, float32 KV cache, no flash attention): the 16 greedy tokens
   identical, the text rows before the image at 1.3e-6 (layer 0) to
   9.9e-3 (layer 32) as the Q8_0 vs Q8_0 noise of the text comparison,
   the image rows 3.9e-2 to 1.4e-1 (the vision noise on top), ASan/UBSan
   clean on the whole path. Speed (bench/README.md): the encoder was
   374 s on one thread: attention_head is bound by the L3 on 4096 keys
   (every query streams 1.2 MB of key tiles and, with V strided by
   n_embd, 20k cache lines: 2.5 GMAC/s, 5.0 with V per head), so
   `attention_block` runs 16 queries per pass over the keys and the
   values with each query's FMAs in attention_head's order (bit
   identical, ops_unit; 24.5 GMAC/s, 20 or 24 queries spill to 10), the
   FFN bias and GELU moved from the main thread to the pool and gelu is
   vectorized as x / (1 + exp(-2y)) (gelu_scalar keeps tanhf, which loses
   its digits below x = -4; within 1.2e-7 of each other relative to the
   input). Encoder Q8_0 74 / 20.5 s with 1 / 4 threads, llama.cpp F16
   with flash attention 115 / 31; faisca describe with 4 threads: load
   9.5 s, encode 20.9, prompt 271 tokens 4.1 s, 7.7 tok/s. Phases at 4
   threads before the GELU work: attention 26 s, matmuls 18, GELU 21 on
   one thread. Not done: the remaining encoder speed (the block kernel
   is bound by the broadcast loads, a key tiled flash attention would
   keep K and V in L2; the matmuls are 1.7 TMAC per image against 1.0 of
   the attention), several images or tiles per prompt, Gemma 3 12B and
   27B vision (same mmproj layout, files beyond this machine), Whisper's
   encoder through attention_block (same shape of problem), the Vulkan
   runner.
11. CUDA runner: initial Llama backend implemented, --cuda in generate,
  runner_dump and bench_runner. CudaDevice loads the driver at runtime;
  CudaKernels has F32/Q8_0 matvec, embed, RMS norm, RoPE, KV store,
  causal GQA attention, SwiGLU and add, with embedded PTX. CudaRunner
  uploads F32/F16/Q8_0 weights, keeps activations and KV in float32,
  uses one stream and batches of 32. Q8_0 is uploaded in the GGUF layout,
  without CPU repacking; four inputs reuse each matrix weight load.
  Linux/RTX 4060: 387 tensors against CPU float32 at worst 5.44e-6 on
  16 repeated token-42 inputs and 5.47e-6 on 65, 16 greedy tokens
  identical; CUDA Q8_0 and --f32 dumps exactly identical. IDs 1..65
  give 5.76e-5 (Vulkan 8.71e-5) with identical tokens, documented as
  a remaining strict-tolerance limitation, not silently waived.
  Compute Sanitizer and ASan/UBSan/LSan clean on generate (ASan needs
  protect_shadow_gap=0). SmolLM2 Q8_0, 512 prompt then 32 tokens:
  mean runs 2..4 about 1027 / 144 tok/s, load 0.23 s, 366 MiB weights.
  Windows untested; Q4/Q5/K-quants, Qwen/Gemma, f16 KV, tensor cores,
  CUDA Graphs, speculative decoding and GPU sessions remain pending.
  Detailed phases and validation gates: [docs/cuda_plan.md](docs/cuda_plan.md).
  The previous VM had no GPU/toolkit; the current machine has CUDA 12.8.
  Phase 0 measured SmolLM2 Q8_0 on RTX 4060: Vulkan 82 / 60 tok/s prompt /
  generation, llama.cpp CUDA 5839 / 348 (different prompts and f32 / f16
  cache, so indicative only). On generation (memory bound) a well made
   Vulkan and a well made CUDA kernel land close to each other; CUDA
   earns its place with int8 tensor cores for the prompt (mma.sync /
   WMMA, the GPU version of our VNNI GEMM), the tooling, and the async
   host to device copies an expert cache needs (Strata's idea, see the
   MoE note below). Rules for whoever works on it (the owner wants the
   mechanical parts done by a cheaper model): never touch the CPU path;
   one kernel per commit; a commit goes in only when its check passes;
   when a comparison fails, stop and write down what was seen instead
   of guessing (precisamos de mais investigação); the expensive model
   does phase 0, the CudaDevice design, the first matvec kernel as the
   template, the phase 4 GEMM, and any comparison that fails without an
   obvious cause; the cheap model ports the remaining kernels from the
   template, writes cuda_check and CudaRunner from the Vulkan ones, runs
   compare_layers and cmp, updates the docs. The owner runs the GPU
   measurements and pastes the numbers.
   Phase 0, measure first (owner's GPU, one day): bench_runner with
   --vulkan on SmolLM2-360M and Gemma 3 4B, generation and prompt,
   against the card's bandwidth (nvidia-smi) and llama-bench with CUDA.
   If Vulkan reaches about 70% of the bandwidth on generation, CUDA is
   justified by the prompt and the MoE only; decide then.
   Phase 1, device without third parties: CudaDevice like VulkanDevice,
   the driver API opened at run time (dlopen libcuda.so.1 / LoadLibrary
   nvcuda.dll: cuInit, context, streams, cuMemAlloc, async copies from
   pinned host memory, modules from embedded PTX). Kernels in .cu files,
   nvcc -ptx once, tools/embed_ptx.py embeds the PTX as embed_spirv.py
   does, generated file committed: building faisca needs no CUDA
   toolkit. No CUDA runtime, no cuBLAS, no NVRTC.
   Phase 2, kernels one by one against a reference: the 14 shaders in
   vulkan_check's order (matvec_f32, matvec_q8_0, Q4_0, Q5_0, Q4_K,
   Q6_K, embed, rms_norm, rope, kv_store, attention, swiglu, add), each
   validated by tests/cuda_check (same as vulkan_check: 1 and 16 inputs
   against a double precision product, thresholds 1e-7 to 5e-7).
   Designed for generation from the start: one warp per row with 16
   byte loads, 4 rows per block, the weights in the packed layout the
   CPU already uses (same bytes, no second copy), float32 activations.
   Phase 3, CudaRunner with VulkanRunner's interface (init(device, file,
   n_ctx), forward, set_trace), one stream per batch, f16 KV cache on
   the device, --cuda in the CLI and runner_dump. Validation:
   compare_layers --ours-flags=--cuda against llama.cpp in float32 on
   the four architectures, 16 tokens identical, cmp of runner_dump
   across block configurations.
   Phase 4, prompt on tensor cores: int8 GEMM by mma.sync or WMMA over
   packed Q8_0 weights and Q8_0 activations, validated with
   matmul_check (zero error against llama.cpp's inputs), then measured
   against llama-bench CUDA on the same card.
   Phase 5, after MoE exists on the CPU: experts resident in VRAM,
   async copies of the missing ones, hit rate measured on our corpus.
   Size: phases 1 to 3 are the size of the Vulkan work already done;
   phase 4 is the hard part.
   MoE note: the loader rejects MoE today. The CPU MoE (router with
   top-k and weight normalization, experts as groups of packed
   matrices read only when chosen, shared expert, validated layer by
   layer against llama.cpp) comes before phase 5; this VM has 15 GB of
   RAM, so the development model must be a small MoE in Q8_0 (OLMoE-
   1B-7B or Qwen1.5-MoE-A2.7B class), to be confirmed with llama.cpp
   running on it here before the loader is designed.
12. TripoSR (single image to 3D mesh, stabilityai/TripoSR): in progress.
   Reference: /media/ctw04578/data/projects/ref/triposr (code/ is
   VAST-AI-Research/TripoSR, hf/ holds config.yaml and model.ckpt;
   dump_ref.py <rgba.png> <out> [mc_res] writes every stage as .f32 plus
   shapes.txt and the torchmcubes mesh; dumps/chair and dumps/chair256 done).
   Python only produces references, with /media/ctw04578/data/pytorch.
   Inputs are cut-out RGBA images, rembg is not ported.
   Done: `CkptFile` (torch_ckpt.h), the owner's choice over a GGUF
   conversion: torch zip checkpoints (zip64, stored entries, pickle protocol
   2 to 5 subset, F32/F16/BF16 contiguous tensors pointing into the mapped
   file, nested "state_dict"). tests/ckpt_dump + tests/compare_ckpt.py: the
   549 TripoSR tensors match torch.load in names, order, dtype, shape and
   bytes; open 20 ms. Small F16/BF16/Parameter/view/scalar/nested
   checkpoints match too, a non contiguous tensor is rejected; 3000 random
   mutations under ASan/UBSan: no crash. MinGW not checked (no toolchain on
   this machine).
   Done: DINO ViT-B/16 image encoder (`TripoModel`, `TripoRunner::encode_image`
   in triposr.h): CLS token, position embeddings resampled as
   ViTEmbeddings.interpolate_pos_encoding of transformers 4.35.0 does
   (scale_factor (32 + 0.1) / 14, so the source step is 14 / 32.1, through
   PyTorch v2.11.0 UpSample.h bicubic, A = -0.75, kept in
   /media/ctw04578/data/projects/ref/pytorch-v2.11.0-upsample), pre-norm
   layers with eps 1e-12 and the erf GELU. tests/triposr_dump (reads the
   reference rgb_cond.f32) + tests/compare_triposr.py on dumps/chair: rel L2
   7.3e-7 after the embeddings, 8.4e-7 after layer 0, 1.9e-6 on the final
   tokens; bit identical with 22 and 5 threads. 2.4 s with 22 threads, 2.0 s
   with 5 (F32 Weight::matmul, one token at a time; not optimized yet).
   Next: triplane transformer (Transformer1D: GroupNorm 32, proj_in, 16
   BasicTransformerBlocks with self and cross attention and GEGLU,
   proj_out), upsample, NeRF MLP density grid, marching cubes, then the
   preprocessing (resize_foreground, antialiased bilinear to 512).

OmniDreams (NVIDIA video world model) is an exploration, not a plan step:
read docs/omnidreams_handoff.md first, then docs/omnidreams.md.

Target model: SmolLM2-360M-Instruct GGUF, then 1B to 3B models. SmolLM2-1.7B
Q4_K_M (/home/user/models/1.7b/, official file, its float32 copy and the
llama dump /home/user/ref/llama_17b_dump.bin) validated: matmul_check zero
error on 145 matrices, float32 path 1.8e-5 from llama.cpp (block 22 is
sensitive: the quantized runs of both runners are 0.9 to 1.0 from float32
there, tokens identical), 16 tokens identical; 9.0 / 28.2 tok/s generation
and 45 / 125 prompt with 1 / 4 threads (llama.cpp 5.4 / 16.5 and 55 /
169). Load takes 10 s (packing 1 GB into huge pages). Llama 3.x files
need rope scaling, which the loader still rejects.

## Tokenizer status

Byte level BPE (`tokenizer.ggml.model` = gpt2) with the `gpt-2`, `phi-2`,
`smollm` and `llama-bpe` pre-tokenizers, matching llama.cpp token for token.
`llama-bpe` (Llama 3): `split_llama3` mirrors llama.cpp's hand written
`unicode_regex_split_custom_llama3`; `ignore_merges` (a word whose byte level
form is a vocabulary token is that token, looked up as the mapped string,
not the raw bytes); `clean_spaces` on; `add_bos` on unless the file says
otherwise. Validated on Llama 3.2 1B: 0 mismatches over 39941 cases, 2.69 M
tokens, the text corpus in /home/user/ref/texts.

- Done: load from GGUF, special token partition, both splitters, BPE merges,
  decode with llama.cpp's clean_spaces passes, unit tests
  (`tests/tokenizer_unit.cpp`), ASan/UBSan clean, word cache, piece table for
  decode, linear scan merges for short words, benchmark against llama.cpp
  (`bench/bench_tokenizer.cpp`, results in bench/README.md: encode 16x to
  27x faster, decode 1.3x, load 5.8x).
- Validated against llama.cpp (commit b86d2f0), 0 mismatches, on the final
  code: the real SmolLM2-360M-Instruct vocabulary (59k cases, 4.2 M tokens,
  chat template tokens and words longer than 64 bytes included), GPT-2 in six
  variants (both pre-tokenizers, with and without extra special tokens and
  BOS/EOS flags, 295k cases, 26.1 M tokens), the unit test vocabulary (49k
  cases, 6.6 M tokens). Every optimization step was re-validated the same way.
- Not done: Windows build (MinGW not available in the last sessions), CLI
  `tokenize` command, a run with Clang.
- Known difference: code points above U+10FFFF from malformed UTF-8 make
  llama.cpp throw; faisca encodes them as U+FFFD.
- Reference setup: clone llama.cpp at b86d2f0, build `libllama` only
  (`-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF`), compile
  `tests/reference/llama_tokenizer_dump.cpp` against it. The GPT-2 files come
  from huggingface.co/openai-community/gpt2 (vocab.json, merges.txt), the model
  from huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF (q8_0).

## Runner status

`LlamaModel` + `Runner` (include/faisca/model.h, runner.h, ops.h, thread.h,
memory.h): the `llama` architecture. F32 in place, F16 converted to float32,
Q8_0, Q4_0, Q5_0, Q4_K and Q6_K packed at load into groups of 16 interleaved
rows (ops.h, "Packed Q8_0" and the K-quant layouts) in anonymous huge page
memory; `--f32` / `--dequantize` convert everything to float32 for the
reference comparison. Activations quantized like llama.cpp on x86: Q8_0
(scale amax/127 as f16, 127/amax, round to even) for the 32-block types, Q8_K
(blocks of 256, float32 scale max/-127 with max the signed extreme, ggml's
nearest_int, sums of 16) for Q4_K and Q6_K; `Weight::activation()` says
which. K-quant kernels: per super block the sub block scales are applied in
int32 (Q6_K subtracts 32 x sum16 first), then fmadd(sumi, d*dx) and, for
Q4_K, fnmadd(summ, dmin*dx), the structure of llama.cpp's AMX tile and VNNI
kernels, so matmul_check gives zero error; the scalar kernels use std::fma
and are bit identical to the AVX-512 ones (ops_unit checks it). Kernels:
AVX-512 VNNI, AVX-VNNI (256-bit, all five types, runner output bit
identical to the AVX-512 build: ops_unit on build-avxvnni, runner_dump cmp
on Q8_0 / Q4_0 / Q4_K_M / 1.7B, matmul_check zero error on the K-quants),
scalar reference; without the kernel for a type the loader converts it to
float32 (Q8_0 without VNNI stays in the file layout with an AVX2 kernel).
Prompt
in batches of 32 (Runner::BATCH), weights read once per batch. Attention over
keys stored in tiles of 16 positions (key_index), own vectorized exp (1 ulp of
expf). KV cache in f16 by default (`Runner::init(..., kv_f16)`, `--kv-f32` in
the CLI and runner_dump): halves converted with F16C on read, arithmetic in
float32, q and softmax weights never rounded; the kernels are templates
shared with the float32 cache, so attention_head_f16 is bit identical to
attention_head on the rounded values (ops_unit checks it). Thread pool over pthreads/Win32; every row group, head and token step
whole on one thread, so output is bit identical for 1, 2 or 4 threads (cmp of
runner_dump files is the test). -ffp-contract=off keeps Debug and Release
identical. No rope scaling, no biases, no MoE yet: rejected with an error.

- Validation: float32 path against llama.cpp with compare_layers.py: 386
  tensors, worst rel L2 4.1e-6 (16 tokens) and 5.3e-6 (1045 tokens), greedy
  tokens identical (both caches float32, `--ours-flags=--kv-f32`). f16 cache
  against the same float32 reference: 1.9e-3 and 2.59e-3, tokens identical;
  llama.cpp's own f16 cache (`--reference-flags=--kv-f16`) is at 2.2e-3 and
  2.50e-3. The 1045 token numbers were 3.66e-6 and 2.49e-3 up to 9532ba4
  and moved at 4611b8f (vectorized SwiGLU and quantizers, which only
  re-measured the 16 token prompt); bisected with worktrees of both
  commits, tokens identical in all. Pitfall: /home/user/ref/long_tokens.txt
  has no "E " prefix, so `cut -c3-` drops its first token (the BOS) and
  gives 3.46e-3 on a 1044 token prompt. On Q8_0 the 8th greedy token
  of the chat prompt is a near tie (gap 0.03 to 0.04 logits between 7042 and
  EOS) that flips with any change of rounding, on both runners. Quantized kernels with matmul_check against llama.cpp's
  own inputs: 193 matrices, zero error, Q8_0 and Q4_0. Full quantized pass:
  inherent noise, both runners 4.4e-2 from float32 logits, same argmax at
  every position, 16 greedy tokens identical. ASan, UBSan, TSan clean with 4
  threads.
- Speed, SmolLM2-360M, 2.1 GHz Sapphire Rapids VM, 1 / 4 threads, 512 token
  prompt then 32 tokens (bench/README.md). Generation: Q8_0 33.8 / 98 tok/s
  (llama.cpp 24.9 / 71), Q4_0 45 / 144 to 183 (llama.cpp 38.8 / 70), Q4_K_M
  40.3 / 118 (llama.cpp 19.2 / 54). Prompt: Q8_0 212 / 603, Q4_0 207 / 570
  to 599, Q4_K_M 204 / 550 (llama.cpp pp512 255 / 143, 249 / 147, 83 /
  266). Four thread numbers move up to 25% between runs on this VM. Before
  the software prefetch generation was Q8_0 24.6 / 76.5, Q4_0 30.8 / 89.7,
  Q4_K_M 23.6 / 78; the vectorized SwiGLU and quantizers then took the Q8_0
  prompt from 178 / 494 to 212 / 603. At a context of 1045 positions, Q8_0: f16 cache 21.5 / 65 tok/s
  against 20.4 / 56 with the float32 cache, prompt 132 / 417 against 122 /
  393, cache 80 MiB instead of 160 for 2048 positions (measured before the
  prefetch). The 512 token file is the first 512 ids of
  /home/user/ref/long_tokens.txt; the older 633 token prompt text was not
  kept. Q5_0 experiment (scratch bench_q5, before the prefetch): expanding
  Q5_0 to packed Q8_0 at load was 6% faster for one token on one thread but
  1.55x the bytes and slower with 16 tokens; packed Q5_0 kept. float32 path:
  about 5 tok/s on one thread.
- What explained the speed (the investigation is in the commit messages): the
  row by row kernel was compute bound at 6 GB/s of weights, ggml's own
  vec_dot too when called from our harness; llama.cpp reaches 10 GB/s because
  on this CPU it repacks Q8_0/Q4_0 into an AMX buffer and runs 512-bit VNNI
  over 16 interleaved rows. Attention was 12 GMAC per 633 token prompt
  (forgetting the 32 layers once cost an hour): tiles, broadcast FMAs and the
  vectorized exp fixed it. Huge pages: +25% on the row kernel, so the packed
  copy asks for them; load takes 0.5 to 1.2 s from synchronous compaction.
  Generation kernels one token at a time were bound by the misses in flight,
  not by the unpack (in cache Q4_0 runs at 39 GB/s, streaming at 7.9): a
  `prefetcht0` per line 8 KiB ahead of the group block took Q4_0 from 7.9 to
  13.8 GB/s on a 400 MiB stream (the L3 is 260 MiB, so streaming tests need
  more than that) and Q8_0 from 12.3 to 16.0; 2 or 4 integer accumulators
  changed nothing. Measure micro benchmarks at -O3: at -O2 the compiler
  spills the unpacked chunks and the in-cache numbers are 3x off. Prompt
  phases, Q8_0, 512 tokens, 1 thread, 3.14 s: matmul 1.86 s, attention
  0.65, token steps 0.37, quantize 0.25 (instrumented copy of runner.cpp in
  the scratchpad, runner_phases.cpp). Token steps were std::exp in swiglu
  (82k calls per token): now exp_values, 0.18 s; quantizers vectorized with
  bit identical bytes (x * id rounded to nearest even, as nearbyint), 0.11
  s. The VNNI GEMM is bound by one zmm vpdpbusd per cycle (64 MAC/cycle,
  131 GMAC/s at 2.1 GHz). swiglu now differs from std::exp by a few ulp, so
  dumps changed; the float32 path against llama.cpp is 4.1e-6 (was 4.5e-6)
  and Q8_0 keeps the same 16 tokens as llama.cpp.
- Speculative decoding: `Runner::truncate(pos)`, `forward(..., all_logits)`
  (count rows of logits) and `init(model, n_ctx, ThreadPool&, ...)` so two
  runners share one pool (two pools spinning on four cores cut the speed to
  a tenth). `Speculative::round(last, out)`: the draft catches up with the
  tokens the target kept (m_tokens by position), proposes n_draft tokens,
  the target checks them in one pass, kept = longest agreeing prefix, out
  gets the kept proposals plus the target's token at the first disagreement.
  Draft files: /home/user/models/draft/ (bartowski Q8_0 and f16, the f32 copy,
  and q4_0-pure made with llama-quantize; the community Q4_0 mixes Q4_1).
  The draft on our runner is at 1.6e-6 from llama.cpp in float32 and 1.7e-2
  in Q8_0, 16 tokens identical. Costs behind the gain: target token 25 ms
  (1 thread), K + 1 token pass 29 to 52 ms, draft token 10.8 (Q8_0) / 7
  (Q4_0) ms; with 4 threads the draft (86 to 136 MiB, in L3) is bound by
  the ~450 fork/join tasks per token.
- Reference tools: `tests/reference/llama_layer_dump.cpp` and
  `tests/matmul_check.cpp` (see README). Q4_0 and Q4_K_M files of SmolLM2 were
  made with llama-quantize from the float32 copy; the Q4_K_M one is mostly
  Q5_0 with Q4_K/Q6_K only where n_in is a multiple of 256 (SmolLM2 has 960).
- Reference setup: `python3 tests/make_f32_gguf.py q8.gguf f32.gguf` (both
  sides read the same float32 weights), `llama_layer_dump` is built by the bench
  configuration (`-DFAISCA_BUILD_BENCH=ON -DFAISCA_LLAMA_DIR=...`). The tools
  take token ids; get them with `tokenizer_dump` on a cases file
  (`<u32 flags=3><u32 length><text>`).
- Pitfalls met: llama.cpp names both the projected and the rotated Q and K
  "Qcur"/"Kcur" and both the FFN down output and the residual sum "ffn_out";
  the dump tool suffixes repeats with #2, #3 and the script matches the best
  variant. llama.cpp defaults to an f16 KV cache and flash attention; the
  reference tool forces f32 and disables it. `llama_decode` takes
  `llama_batch`, `llama_process` takes `llama_batch_ext`.

## Commands

```
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
tests/run_all.sh                    # release + ASan/UBSan
python3 tests/compare_tokenizer.py build/tokenizer_dump <llama_tokenizer_dump> model.gguf --files text.txt
```

`tests/reference/llama_tokenizer_dump.cpp` links against a built llama.cpp.
