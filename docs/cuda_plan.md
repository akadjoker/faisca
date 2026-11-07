# CUDA backend plan

Status: initial Llama CUDA inference runs end to end on Linux/RTX 4060.
F32/Q8_0 kernels and CudaRunner validated; remaining formats, architectures,
optimizations and Windows validation are pending.

## Decision and scope

Build an optional backend inside faisca, not a separate project. Reuse the
GGUF loader, tokenizer, CLI, error handling and reference tools. Keep the
CPU kernels and execution path unchanged. Use C++17 and faisca containers,
with no CUDA Runtime, cuBLAS, NVRTC or other third-party libraries.

The owner approved continuing on 2026-11-07 with the phase 0 comparison
limitations recorded below. DeepInfer is a design reference, not a runtime
dependency: study prompt/decode-specific kernels, fusion, asynchronous
execution and batch-invariant results after the correctness baseline.
Its SM100/cluster kernels and approximate DeepSeek-specific SWA replay
are not directly portable to this RTX 4060 or to Gemma.

The owner's subsequent request prioritized running a complete model on CUDA
now. The Llama/F32/Q8_0 vertical slice therefore proceeds through the needed
phase 2 and 3 operations while other formats and the Windows gate remain
explicitly pending. Each implemented operation still passes its reference
before integration; this does not declare all of phases 2 and 3 complete.

Preserve the phase order in ../CLAUDE.md, step 11. Each phase advances only
after its reference checks pass. Commit and push each validated chunk right
away; each kernel commit must include passing checks.

## Hardware and open decisions

Verified locally on 2026-11-07:

- Linux 7.0.0-42-generic, x86_64.
- NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB total VRAM; 7649 MiB free
  at the time of the probe, not a model memory measurement.
- NVIDIA driver 580.196.07.
- CUDA Toolkit 12.8, nvcc V12.8.93.
- Compute capability 8.9, reported by nvidia-smi; use compute_89 as the
  initial PTX target for this device, with unsupported-device errors.

Select tensor-core instructions during phase 4. Host RAM has not been
checked in this session. The VM
limitations recorded in CLAUDE.md describe the previous environment.

SmolLM2-360M Q8_0 is the first model. Gemma 3 4B Q8_0 is a later target,
subject to measured peak VRAM including weights, KV cache and scratch.
No GPU result means unvalidated, even when host compilation passes.

## Phase 0: baseline on the owner's GPU

- Preparation done: bench/bench_runner.cpp now supports --vulkan and CPU
  --kv-f32. Both execution paths passed a two-token smoke test; builds with
  and without FAISCA_VULKAN passed strict syntax checks.
- Run Vulkan bench_runner and CUDA llama-bench on the same model files:
  SmolLM2-360M first, then Gemma 3 4B if memory permits.
- Match prompt length, generated tokens, context, cache precision and
  offloaded layers. Record any comparison that cannot match these settings.
- Record load time, prompt tokens/s, generation tokens/s, peak VRAM,
  GPU/driver versions, build flags and reference revision. Use warm-up and
  repeated runs; keep raw results as well as their summary.
- Estimate generation weight bandwidth from the bytes read per token and
  measured tokens/s. Record the source of the card's rated bandwidth;
  nvidia-smi identifies the device but is not a memory-bandwidth benchmark.

Gate: measurements recorded and the owner confirms the CUDA priorities.
Keep step 11's approximately 70% bandwidth figure as a decision heuristic,
not a correctness threshold or a promised result.

### Initial measurements, 2026-11-07

Model: the official HuggingFaceTB/SmolLM2-360M-Instruct-GGUF Q8_0 file,
revision 593b5a2e04c8f3e4ee880263f93e0bd2901ad47f. Downloaded into
$HOME/.cache/faisca/models/smollm2-360m-instruct-q8_0.gguf.
SHA-256: 48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201.

Vulkan matrix checks: eight matrices against double precision, with a
5e-7 limit. Worst relative L2: 1.619e-7 with one input and 1.265e-7 with
16 inputs, zero failures. This validates those products, not the whole
runner's layer outputs or greedy tokens against llama.cpp.

The full Vulkan runner was also compared with the existing CPU runner in
float32 (--dequantize --kv-f32), using the same Q8_0 file, a 16-token
prompt of token 42 and 16 greedy steps. compare_layers.py passed:
387 tensors, worst relative L2 8.220e-6 at l_out-31, 16 identical generated
tokens, zero failures. This reference is faisca CPU, not llama.cpp.

Initial Vulkan benchmark: 512 repetitions of token 42, then 32 greedy
generation steps; batch size 32, float32 KV, four runs on the RTX 4060.
The first run is treated as warm-up when interpreting the remaining runs.

| Run | Prompt tok/s | Generation tok/s |
| --- | --- | --- |
| 1, warm-up | 81.8 | 59.5 |
| 2 | 85.3 | 65.1 |
| 3 | 87.4 | 61.8 |
| 4 | 85.0 | 64.6 |

Load time: 2.37 s. Uploaded weights: 384618240 bytes. Raw output is in
build/cuda-baseline-vulkan.log, a local ignored build artifact. The runs
were not thermally controlled; no peak VRAM or rated-bandwidth comparison
has been recorded yet. These numbers are provisional.

A second run with 100 ms nvidia-smi monitoring, while the CUDA reference
was compiling on the CPU, measured prompt 67.5 / 70.6 / 74.1 / 65.2 tok/s
and generation 45.3 / 51.2 / 50.6 / 50.2 tok/s. Load: 1.06 s. The monitor
recorded 15 MiB initially used, 904 MiB peak total GPU memory used,
51 to 67 C and peak power 54.34 W. Total GPU memory includes any other
processes; it is not an exact allocator accounting for faisca.
Raw artifacts: build/cuda-baseline-vulkan-monitored.log and
build/cuda-baseline-gpu-monitor.csv. Repeat without compilation or other
concurrent work before comparing against CUDA.

Reproduce from the workspace root:

```sh
seq 512 | awk '{print 42}' | build/bench_runner --vulkan --runs 4 --generate 32 "$HOME/.cache/faisca/models/smollm2-360m-instruct-q8_0.gguf" /dev/stdin
```

Reference: the prebuilt Unsloth llama.cpp libraries report build 11160,
commit a3c12db9d, GNU 11.4.0. Their CUDA 13 dependencies are installed in
$HOME/.unsloth/studio/unsloth_studio/lib/python3.13/site-packages/nvidia/cu13/lib,
but were not on the loader's search path. The CUDA 12.8 source rebuild in
build/llama-reference was cancelled once these existing libraries were
located and the prebuilt CUDA backend loaded successfully.

build/llama-bench-reference was compiled from the archive's original
tools/llama-bench/main.cpp against libllama-bench-impl.so. Run it with
the CUDA 13 directory in LD_LIBRARY_PATH and the working directory set to
$HOME/.unsloth/llama.cpp/build/bin for automatic backend discovery.
The source archive has no Git metadata; the reported prebuilt commit does
not establish the archive's revision. Failed earlier invocations are not
performance measurements.

CUDA reference settings: same file, 361821120 parameters and 384618240
weight bytes, full GPU offload (-ngl 99), one host thread, batch/ubatch 32,
flash attention off, four measured runs after llama-bench's own warm-up.
Prompt test: -p 512 -n 0 -d 0. Generation test: -p 0 -n 32 -d 512.
This version rejects f32 KV, so both reference caches use f16 instead.
Its JSON labels this file "llama 3B Q8_0" despite reporting 361821120
parameters; identify the model by file, hash and parameter count instead.

| Backend | Prompt tok/s | Generation tok/s | KV |
| --- | --- | --- | --- |
| faisca Vulkan, mean of runs 2 to 4 | 82.4 | 59.9 | f32 |
| llama.cpp CUDA, mean of four runs | 5839.2 | 348.0 | f16 |

The final Vulkan run had no reference compilation running: prompt
80.8 / 83.5 / 81.2 / 82.5 tok/s, generation 61.4 / 61.3 / 59.8 / 58.7
tok/s, load 0.35 s. Raw output: build/cuda-baseline-vulkan-final.log.
CUDA standard deviations: prompt 237.9 tok/s, generation 28.3 tok/s.
Raw reference results and settings: build/cuda-baseline-llama-prompt.json
and build/cuda-baseline-llama-generation.json; GPU discovery is recorded
in the corresponding .stderr files.

These are not identical-workload timings: the reference uses a different
prompt and generation policy, and its cache precision differs. Reference
load time and peak VRAM were not measured. NVIDIA's laptop comparison page
confirms a 128-bit memory interface but does not specify its data rate:
https://www.nvidia.com/en-us/geforce/laptops/compare/. The rated-bandwidth
ratio remains unverified; do not infer it from desktop RTX 4060 specs.

Recommendation for the owner: proceed with CUDA for both prompt and
generation on this GPU. The observed gap is substantial, but is not proof
that Vulkan intrinsically cannot reach similar generation performance;
the current faisca Vulkan kernels are unoptimized. Confirm this priority
before phase 1. Gemma measurements remain blocked by Vulkan architecture
support, not by a demonstrated VRAM limit.

## Phase 1: device and build integration

Planned files: include/faisca/cuda.h, src/cuda/device.cpp,
src/cuda/kernels/, tools/embed_ptx.py and generated embedded PTX sources.

- Load libcuda.so.1 on Linux and nvcuda.dll on Windows at runtime.
- Implement explicit context ownership, device buffers, pinned host buffers,
  streams, events, copies, module loading and kernel launches in CudaDevice.
- Report device name, compute capability and free/total device memory.
- Add FAISCA_CUDA to CMake. Normal builds use committed PTX and need no
  toolkit or driver. Regenerating PTX requires nvcc and a recorded version.
- Establish how build-time driver declarations are provided without
  requiring installed toolkit headers. Keep that boundary private.
- Return a faisca::Error for a missing driver, unsupported PTX target,
  allocation failure or asynchronous execution error. Release partial state.

Gate: host builds with the backend on/off; absent-driver failure is clean;
allocation, upload/download byte equality and cleanup pass on a real GPU.
Verify Linux and Windows loading on their respective systems, not Wine alone.

### Implementation and checks, 2026-11-07

- include/faisca/cuda.h and src/cuda/device.cpp implement CudaDevice using
  a private driver ABI table and runtime dlopen/LoadLibrary. No SDK headers
  or CUDA libraries are included or linked by the normal build.
- Dedicated contexts preserve the caller's current context via push/pop.
  Device and pinned host buffers, nonblocking streams, events, PTX modules
  and kernel launches use explicit ownership and tracked resource ids.
- release synchronizes pending work and frees tracked resources. Copies
  of freed handles and handles from another CudaDevice are rejected.
  Explicit destruction synchronizes the context before freeing resources.
- Async copies accept tracked pinned buffers and offsets. A successful
  enqueue does not mean the host data is ready: synchronize a stream/event
  before consuming it. Serialize allocation, destruction and release;
  concurrent resource mutation is not supported.
- FAISCA_CUDA defaults to ON for the 64-bit driver backend; OFF excludes
  it. The CPU-only CLI built successfully with CUDA and Vulkan OFF.
  The original CPU implementation was not changed.
- tests/cuda_check.cpp checks missing-driver recovery, failed allocations,
  byte equality, offsets/bounds, events between two streams, malformed PTX,
  missing symbols, exact output of a 32-thread integer kernel, stale
  resources, external-context preservation and release with pending work.
  --require-device makes missing GPU/driver initialization a failure.
  The embedded test PTX targets sm_89; other devices run transfer tests only.
- Compute Sanitizer memcheck: zero errors and zero bytes leaked. Intentional
  API failures are excluded with --report-api-errors no.
- ASan/UBSan with leak detection passed with protect_shadow_gap=0.
  With default ASan options this driver returned CUDA_ERROR_OUT_OF_MEMORY
  from cuInit before resource creation; the compatibility option is required
  for this environment and is not a backend code change.
- tools/embed_ptx.py follows embed_spirv.py: nvcc generates PTX with
  --fmad=false, then the tool writes C++ char arrays and declarations.
  A temporary fill_words.cu fixture in build/cuda-embed-fixture passed the
  full nvcc -> embedded C++ -> driver JIT -> exact GPU output check.
  Production embedded modules are now in src/cuda/kernels_ptx.h and .cpp,
  regenerated from the .cu sources in src/cuda/kernels.
- CMake Tools/CTest could not configure this workspace through the editor
  extension. Direct CMake builds and cuda_check executions passed; the
  cuda_check test is registered with CTest and skips unavailable GPUs (77).
- No MinGW compiler or Windows NVIDIA environment was available here.
  Windows implementation is present but its build/loading gate is pending.

Focused checks from the workspace root:

```sh
build/cuda_check --require-device
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --report-api-errors no --leak-check full --error-exitcode 1 build/cuda_check --require-device
ASAN_OPTIONS=detect_leaks=1:protect_shadow_gap=0 build/cuda_check_sanitize --require-device
python3 tools/embed_ptx.py --nvcc /usr/local/cuda-12.8/bin/nvcc
```

The embedding command regenerates all implemented kernels in src/cuda/kernels.
Use --source-dir/--output-dir for temporary fixtures.

### Runnable Llama CUDA slice

Implemented: matvec_f32, matvec_q8_0, embedding for both formats, RMS norm,
interleaved RoPE with optional frequency factors, KV store, causal GQA
attention, SwiGLU and residual add. The GPU reads native GGUF Q8_0 blocks;
this is not the proposed CPU interleaved packing. F32 and Q8_0 matvec process
four inputs per row/warp to reuse loads. No tensor-core GEMM or CUDA Graphs
are used yet. The native layout is the measured initial implementation,
not a claim that GPU/CPU packing alternatives have been benchmarked.

CudaRunner uploads weights once, keeps activations and float32 KV on the
device, uses one ordered stream and batches of 32, and downloads final
logits. Without tracing/all_logits, only the last row runs the output head.
F16 weights convert to F32; --f32 converts Q8_0 too. Measured model: the
official SmolLM2-360M Q8_0 file identified above, not an F16 model file.
Other architectures, quantization types and unsupported scaling/window
flags are rejected. Only generate, runner_dump and bench_runner expose
--cuda; chat/session, speculative decoding and speech/vision are not ported.

Kernel references passed: eight real matrices with 1, 3 and 16 inputs
(F32 worst relative L2 below 1.9e-7; Q8_0 bit identical to the F32 result),
embedding/add/KV exact, RMS norm 6.25e-8, RoPE at most 4.20e-8, SwiGLU
4.13e-8, causal GQA attention 1.52e-7 against double precision. A matrix
check initially exposed a pageable-upload/nonblocking-stream race (0.196
relative error). Completing synchronous uploads before returning fixed it;
the unchanged matrix check then passed. Async pinned copies remain ordered
by the caller's stream/events and do not use this global upload barrier.

Full-model reference: compare_layers.py against the existing faisca CPU
runner with --dequantize --kv-f32, not directly against llama.cpp:

- 16 repetitions of token 42: 387 tensors, worst relative L2 5.441e-6,
  16 identical greedy tokens, zero failures at the default 1e-5 limit.
- 65 repetitions of token 42: 387 tensors, worst relative L2 5.467e-6,
  16 identical greedy tokens, zero failures; covers two full batches and
  a tail, with cache positions crossing both batch boundaries.
- CUDA --dequantize versus CUDA Q8_0: all 387 tensors exactly equal,
  16 identical tokens at zero relative/absolute tolerance.
- IDs 1..65 are a remaining strict-tolerance case: eight tensors exceed
  1e-5, CUDA worst 5.761e-5 at ffn_out-30. The existing Vulkan path fails
  the same eight tensors at 8.708e-5; both keep 16 tokens identical. This
  discrepancy has not been fully attributed, and no threshold was silently
  widened. Do not claim 1e-5 accuracy for arbitrary input sequences.
- CLI response to "What is the capital of France?" is byte identical to
  the CPU float32 response: "The capital of France is Paris."

Compute Sanitizer on full generate: zero memory errors and zero leaked
allocations. Whole-library Debug ASan/UBSan with leak detection also passed
generate, with ASAN_OPTIONS=detect_leaks=1:protect_shadow_gap=0. The build
with CUDA OFF still compiles and returns a clear error for --cuda.

RTX 4060 benchmark, same dimensions and fixed prompt as Vulkan above:
512 repetitions of token 42 then 32 greedy steps, context 545, batch 32,
float32 KV, four runs. Mean of runs 2 to 4: prompt 1026.7 tok/s, generation
143.6 tok/s. Samples: prompt 1058.4 / 1019.1 / 1040.1 / 1020.8, generation
146.1 / 145.5 / 144.6 / 140.6. Load 0.23 s, weights 384618240 bytes.
Raw output: build/cuda-faisca-baseline.log. Mobile power/thermal state is
not controlled. The earlier llama.cpp 5839 / 348 reference remains
indicative because its prompt/generation policy and cache type differ.

```sh
build/faisca generate "$HOME/.cache/faisca/models/smollm2-360m-instruct-q8_0.gguf" --cuda -p "What is the capital of France?" -n 32
seq 512 | awk '{print 42}' | build/bench_runner --cuda --runs 4 --generate 32 "$HOME/.cache/faisca/models/smollm2-360m-instruct-q8_0.gguf" /dev/stdin
```

### Generation against llama.cpp, 2026-11-07

Goal: CUDA generation faster than llama.cpp on the RTX 4060 Laptop with
SmolLM2-360M-Instruct Q8_0. Fair comparison: the same depth, 512 cached
tokens then 128 generated (`llama-bench -p 0 -n 128 -d 512 -r 5 -ngl 99 -t 1`
against `bench_runner --cuda --runs 5 --generate 128` on 512 repetitions of
token 42, the mean of faisca runs 2 to 5 against llama.cpp's mean). The two
alternate round by round: this laptop's GPU drifts about 5% between minutes
(an antivirus scan raised the load to 62 during the work), so only
alternating rounds compare. llama-bench's plain tg32 starts at depth 0 and is
not the same measurement.

Steps, each measured with nsys (graph nodes traced) and kept only if it paid:

| Step | Generation tok/s | Note |
| --- | --- | --- |
| Start | 147 | matvec_q8_0 102 GB/s, attention 2.2 ms per token |
| Q8_0 packed as quants then f16 scales, one warp per row, 16 byte loads | 210 | |
| Split attention (flash-decoding, 128 positions per block) | 320 | |
| CUDA graph per decode step, position read on the device | 349 | |
| f16 KV cache, the CPU runner's default (`--kv-f32` keeps float32) | 352 | |
| q/k/v in one launch, residual add in the o and down matvecs, gate+up+swiglu in one kernel | 341-352 | kernel time 2606 to 2464 us per token |
| rope on q and k plus the KV store in one kernel | | |
| logits copied inside the graph into pinned memory | 375-377 | tie with llama.cpp |
| attention specialised for head_dim 64 and 128 (loads unrolled) | 377 | |
| combine merged into the last split block (atomic counter) | | |
| four rows per warp (gate/up: two each), the input loaded once for them | 389-392 | llama.cpp 376-385 in the same rounds |

Final: faisca ahead in 4 of 4 alternating rounds, about 2.5%. Kernel time
2.37 ms per token; the rest is the gap between dependent graph nodes
(about 1.2 us each, 8 nodes per layer).

Tried and reverted (slower): loads of four chunks before the arithmetic in
the matvec (lm_head 205 to 228 us); rms_norm recomputed per warp inside the
matvecs (swiglu 24 to 97 us); rms_norm per block into shared memory, with or
without a weight prefetch before it (qkv 9.6 to 16 us); the next layer's
weights prefetched into L2 on a side stream (matvecs unchanged, rms_norm
slower beside it); rms_norm in the last block of the o and down matvecs
(those 10 to 14 us, more than the norm they removed). Programmatic dependent
launch needs sm_90, this GPU is sm_89. ncu has no counter permission here.

Checks on the final state:

- cuda_check: pass.
- compare_layers.py against the CPU runner, `--kv-f32` both sides
  (`--dequantize` on the CPU): 16, 65 and 300 repetitions of token 42, 387
  tensors, worst 9.2e-6, 16 identical tokens, zero failures. The fused path
  only runs without a trace; its fusions use the same float operations as the
  separate kernels.
- IDs 1..65, float32 KV: 7 tensors above 1e-5, worst 5.48e-5 at ffn_out-30
  (before: 8 at 5.76e-5), 16 identical tokens. Still not attributed.
- f16 KV against the CPU f16 KV: 2 failures, worst 4.4e-5 at attn_out-31, 16
  identical tokens. The CPU computes in float32 over the same halves; the
  likely cause is a 1e-7 difference before rounding flipping a half by one
  ulp (1e-3 relative). Not proven; recorded, the limit not widened.
- `faisca generate --cuda` against `--f32` on the CPU, 4 prompts, 300 tokens,
  f16 and float32 KV: byte identical text, "The capital of France is Paris."

Prompt is not done: about 1150 tok/s against llama.cpp's 5800 (batch 32) and
24700 (default batch); that is phase 4 (tensor-core GEMM).

### Prompt on the tensor cores, 2026-11-07

Batches of more than one token with Q8_0 weights quantize their activations
as the CPU runner does (quantize_q8_0, after llama.cpp) and run
`gemm_q8_0`: mma.sync m16n8k32 s8, one per Q8_0 block, so every block's
integer sum is exact, then block by block in order
`acc = fma(float(sum), d_w * d_x, acc)`, the epilogue of the CPU's VNNI
kernels. One token keeps the float32 matvec; `--dequantize` keeps every
product in float32 (the strict reference path).

- cuda_check `check_gemm`: random weights and inputs, shapes 960x320,
  2560x96, 64x16, 1/7/13/32/77 tokens: quantized bytes and every product
  bit identical to `quantize_q8_0` + `matmul_q8_0_packed` (the VNNI kernel
  this CPU runs). The scalar fallback `matvec_q8_0_packed_scalar` is not
  bit identical to the VNNI kernels: it is built with -ffp-contract=off and
  rounds the product and the sum apart (differences up to 1.4e-4 relative
  on small results).
- compare_layers, IDs 1..65 + 16 generated: CUDA quantized against CPU
  quantized: attn_norm-0 7e-8, Qcur-0 2.2e-4 (the 1e-7 differences change
  roundings), logits 3.2e-2, 16 identical tokens. Distance to the float32
  CPU: CUDA 4.97e-2, CPU 5.02e-2. With `--dequantize`: 0 failures, as before.
- The batch is now 512 tokens (llama.cpp's ubatch); prompt attention is a
  flash-attention kernel (FlashAttention-2 tiling, float32, K/V tiles of 64
  in shared memory, 64 queries per block, online softmax per tile):
  compare_layers float32 path 0 failures at 16, 65 and 300 tokens.

Prompt of 512 tokens, tok/s (llama.cpp 6600 with batch 32, 25000 default):

| Step | Prompt |
| --- | --- |
| float matvec, batch 32 | 1150 |
| int8 GEMM, one warp per row tile | 3320 |
| one warp per 8 tokens, loads first | 5450 |
| batch 512 | 8160 |
| warp tile 32 rows x 32 tokens | 11300 |
| flash-attention prompt kernel | 16900 |
| attention on the TF32 tensor cores, 3xTF32 (Ootomo and Yokota 2022) | 23050 |
| GEMM: next block's fragments loaded during this one | 23820 |
| GEMM: quants staged in shared memory with cp.async, two stages | 24140 |
| GEMM: a block major copy of the scales, staged with the quants | 27320 |

3xTF32: every operand split into tf32(x) and tf32(x - tf32(x)), each
product as small*big + big*small + big*big on mma.sync m16n8k8; keys of
each group of 8 taken as 0,2,4,6,1,3,5,7 for P V so P's C fragment is its
A fragment. Float32 path after it: 0 failures at 16, 65 and 300 tokens
(worst 5.4e-6). Head_dim 128 keeps the float32 CUDA core kernel. Measured
without the scale loads the GEMM went 84 to 49 us: hence the second, block
major copy of the weight scales (6% more weight memory) and block major
activation scales from quantize_q8_0. GEMM still bit identical to the CPU
in all 15 cases.

Alternated with llama.cpp (llama-bench -p 512 -n 128 -r 5; tg at depth
512; faisca mean of runs 2 to 5), 4 rounds:

| | llama.cpp | faisca |
| --- | --- | --- |
| prompt 512 (tok/s) | 24040-25260 | 26320-27090 |
| generation at depth 512 (tok/s) | 384-387 | 390-393 (one round 367) |

Text: with the prompt quantized and generation in float32, `--cuda` text
is no longer byte identical to either CPU mode on long answers (example:
the 13th word of "Explain how a car engine works."); the CPU quantized and
CPU float32 texts also differ from each other. Short answers still match the
CPU float32 text. Per-layer checks above hold.

## Phase 2: kernels, one at a time

First implement matvec_f32 as the template, then matvec_q8_0. Continue with
Q4_0, Q5_0, Q4_K, Q6_K, embedding, RMS norm, RoPE, KV store, attention,
SwiGLU and add, covering both embedding shaders in the Vulkan inventory.

- Create tests/cuda_check.cpp using vulkan_check's fixtures and references.
- Start with one warp per matrix row, four rows per block and aligned
  16-byte loads. Verify alignment, padded rows and tails explicitly.
- Use the existing CPU packed weight layout as the planned starting point.
  Measure coalescing and bandwidth before claiming it is GPU-efficient;
  propose a layout change to the owner if measurements require one.
- Matrix checks use identical decoded weights and inputs against a double
  precision product, with 1 and 16 inputs and the corresponding Vulkan
  relative-error limits of 1e-7 to 5e-7. Handle zero-norm references explicitly.
- Other operations use their own scalar/double references and justified
  tolerances; a matrix-product threshold does not apply to every operation.
- Check repeatability. Preserve the reduction order across supported launch
  configurations so the planned dump cmp check remains meaningful.

Gate: each kernel passes before the next begins. On an unexplained failure,
record inputs, error metrics and launch configuration, then stop that phase.

## Phase 3: runner and architecture coverage

Planned files: include/faisca/cuda_ops.h, include/faisca/cuda_runner.h,
src/cuda/ops.cpp and src/cuda/runner.cpp. Integrate --cuda into the CLI,
runner_dump and bench_runner; preserve existing backend behaviour.

- Reuse VulkanRunner's interface and tracing pattern, not its architecture
  restrictions. Begin with SmolLM2, then Qwen2, Qwen3 and Gemma 3.
- Keep weights, activations and KV on the device. Use one ordered stream
  for batch execution; download logits at the batch boundary and tensors
  only when tracing. Synchronize before consuming host results.
- First use float32 activations and KV as a debugging checkpoint. Validate
  against the float32 llama.cpp reference with identical decoded weights.
- Cover biases, NeoX RoPE, QK norms, GELU, post-norms, embedding/attention
  scaling, linear RoPE scaling and sliding-window boundaries explicitly.
- Then implement the planned f16 KV cache and measure its error separately.
- Compare every traced layer with compare_layers.py, require 16 identical
  greedy tokens on the agreed fixtures, and cmp dumps across supported
  launch configurations. CPU/GPU bit equality is not a general guarantee.
- Test batch tails, reset, truncate, context exhaustion and VRAM exhaustion.

Gate: all four architectures pass their references, including a prompt that
crosses Gemma's sliding-window boundary. Record peak memory and speed.

## Phase 4: tensor-core prompt

- Select mma.sync or WMMA after confirming hardware support.
- Implement Q8_0 activation quantization and int8 GEMM; check quantized
  bytes and integer accumulations separately from the float epilogue.
- Extend matmul_check for CUDA on llama.cpp's identical input tensors.
  Step 11 requires zero final error; if this fails, investigate rather than
  silently weakening the gate or attributing it to quantization noise.
- Re-run layer/token checks and benchmark against CUDA llama-bench on the
  same card. Consider fusion or CUDA Graphs only after profiling launch cost.

Gate: reference checks pass and measured prompt performance is recorded.
Retain the validated generation kernels and a path for unsupported hardware.

## Phase 5 and exclusions

MoE GPU work waits for a CPU MoE implementation validated against llama.cpp.
Then add resident experts and asynchronous missing-expert transfers, measuring
cache hit rate, transfer cost and memory use on a fixed corpus.

Whisper, Pocket TTS, vision, session export and multi-GPU are outside this
first CUDA plan. Do not report those workflows as supported by --cuda.

## Next action

Generation is ahead of llama.cpp by a small margin (see the 2026-11-07
section); the prompt is far behind, so phase 4 (tensor-core GEMM) comes next. Remaining formats and Qwen/Gemma need their own reference gates before
support is enabled. Windows remains untested, and the varied-ID strict
accuracy case remains recorded. Do not silently remove the pending Gemma,
reference peak-memory/load-time or rated-bandwidth measurements from phase 0.