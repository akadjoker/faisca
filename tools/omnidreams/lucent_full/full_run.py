import glob
import os
import shutil
import subprocess
import time
import zipfile

OUT = "/kaggle/working"
WORK = "/tmp/work"
os.makedirs(WORK, exist_ok=True)
LOG = open(os.path.join(OUT, "demo.txt"), "w")
DRIVER_SOURCE = 'import glob\nimport json\nimport os\nimport sys\nimport time\n\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\nfrom PIL import Image\n\nOUT = os.environ["STEP_OUT"]\nCHECKPOINT = os.environ["STEP_CHECKPOINT"]\nSCENE = os.environ["STEP_SCENE"]\nN_CHUNKS = int(os.environ.get("STEP_CHUNKS", "3"))\nCAMERA = "camera_front_wide_120fov"\nWIDTH, HEIGHT = 1280, 704\n\n\ndef say(text):\n    print(text, flush=True)\n\n\ndef stamp():\n    torch.cuda.synchronize()\n    return time.time()\n\n\nsay(f"python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda}")\nsay(f"bf16 supported: {torch.cuda.is_bf16_supported()}")\nprobe = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)\nsay(f"bf16 matmul probe: {(probe @ probe).float().abs().mean().item():.4f}")\n\nimport flashdreams.core.attention.cp as cp_module\n\n\nQUERY_BLOCK = int(os.environ.get("STEP_QUERY_BLOCK", "1024"))\n\n\ndef plain_sdpa(query, key, value, return_lse=False):\n    if query.shape[2] <= QUERY_BLOCK:\n        return F.scaled_dot_product_attention(query, key, value), None\n    parts = [\n        F.scaled_dot_product_attention(query[:, :, start:start + QUERY_BLOCK], key, value)\n        for start in range(0, query.shape[2], QUERY_BLOCK)\n    ]\n    return torch.cat(parts, dim=2), None\n\n\ncp_module.torch_sdpa_cudnn = plain_sdpa\ncp_module.torch_sdpa_flash = plain_sdpa\nsay("attention: cudnn/flash replaced by F.scaled_dot_product_attention")\n\nfrom flashdreams.infra.config import derive_config\nfrom omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG\n\nconfig = derive_config(\n    OMNIDREAMS_PIPELINE_CONFIG,\n    name="omnidreams-t4-reference",\n    text_encoder=dict(run_on_cpu=True),\n    image_encoder=dict(use_cuda_graph=False),\n    encoder=dict(use_cuda_graph=False),\n    decoder=dict(use_cuda_graph=False),\n    diffusion_model=dict(transformer=dict(use_cuda_graph=False, checkpoint_path=CHECKPOINT)),\n)\n\nstarted = time.time()\npipeline = config.setup().to("cuda").eval()\nsay(f"pipeline setup {time.time() - started:.1f} s, gpu allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")\n\nif torch.cuda.device_count() > 1 and os.environ.get("STEP_ENCODER_GPU1", "1") == "1":\n    encoder = pipeline.encoder\n    encoder.to("cuda:1")\n    encoder_forward = encoder.forward\n\n    def forward_on_second_gpu(*args, **kwargs):\n        if "input" in kwargs:\n            kwargs["input"] = kwargs["input"].to("cuda:1")\n        args = tuple(item.to("cuda:1") if isinstance(item, torch.Tensor) else item for item in args)\n        with torch.cuda.device(1):\n            result = encoder_forward(*args, **kwargs)\n        return result.to("cuda:0")\n\n    encoder.forward = forward_on_second_gpu\n    say("hdmap encoder moved to cuda:1")\n\nnetwork = None\nfor name, module in pipeline.named_modules():\n    if type(module).__name__ == "CosmosDiTNetwork":\n        network = module\n        say(f"network at {name}")\n        break\n\ndumps = {}\ndump_enabled = {"on": True}\n\n\ndef make_hook(label):\n    def hook(module, inputs, output):\n        if not dump_enabled["on"] or label in dumps:\n            return\n        value = output[0] if isinstance(output, tuple) else output\n        dumps[label] = value.detach().to("cpu", torch.float32).clone()\n        if label.endswith("in") is False and inputs and isinstance(inputs[0], torch.Tensor) and (label + ".input") not in dumps:\n            dumps[label + ".input"] = inputs[0].detach().to("cpu", torch.float32).clone()\n    return hook\n\n\nif network is not None:\n    network.x_embedder.register_forward_hook(make_hook("x_embedder"))\n    network.blocks[0].register_forward_hook(make_hook("block0"))\n    network.blocks[-1].register_forward_hook(make_hook("block27"))\n    network.final_layer.register_forward_hook(make_hook("final_layer"))\n\nsay("\\n== conditioning (Lucent first frame, ASTAS hdmap drawn by ludus)")\nframe_counts = [pipeline.get_num_output_frames(i) for i in range(N_CHUNKS)]\ntotal_frames = sum(frame_counts)\nstored = np.load(os.path.join(SCENE, "hdmap_frames.npz"))\nhdmap_uint8 = stored["frames"][:total_frames]\nif len(hdmap_uint8) < total_frames:\n    raise SystemExit(f"hdmap has {len(hdmap_uint8)} frames, {total_frames} needed")\nhdmap_unit = torch.from_numpy(hdmap_uint8).permute(0, 3, 1, 2).float().div(255.0)\nsay(f"hdmap {hdmap_uint8.shape}, frames per chunk {frame_counts}")\nwith Image.open(os.path.join(SCENE, "first_frame.png")) as image:\n    first_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)\nif first_rgb.shape != (HEIGHT, WIDTH, 3):\n    raise SystemExit(f"first frame {first_rgb.shape}, expected {(HEIGHT, WIDTH, 3)}")\noverlay = first_rgb.copy()\nmask = hdmap_uint8[0].max(axis=2) > 0\noverlay[mask] = (0.4 * overlay[mask] + 0.6 * hdmap_uint8[0][mask]).astype(np.uint8)\nImage.fromarray(overlay).save(os.path.join(OUT, "overlay_0000.png"))\n\nsay("\\n== text and first frame")\nprompt = open(os.path.join(SCENE, "prompt.txt")).read().strip()\nsay(f"prompt: {prompt[:200]}")\nimage = torch.from_numpy(first_rgb).permute(2, 0, 1).float().div(127.5).sub(1.0)\nimage = image.reshape(1, 1, 1, 3, HEIGHT, WIDTH).to("cuda", torch.bfloat16)\nstarted = time.time()\ntext_embeddings = torch.stack([pipeline.text_encoder([prompt])], dim=0)\nsay(f"text encoded {tuple(text_embeddings.shape)} {text_embeddings.dtype} in {time.time() - started:.1f} s")\ntorch.save(text_embeddings.detach().cpu(), os.path.join(OUT, "text_embeddings.pt"))\nstarted = stamp()\nimage_embeddings = pipeline.image_encoder(image)\nsay(f"image encoded {tuple(image_embeddings.shape)} in {stamp() - started:.2f} s")\ntorch.save(image_embeddings.detach().float().cpu(), os.path.join(OUT, "image_latent.pt"))\ncache = pipeline.initialize_cache_from_embeddings(text_embeddings=text_embeddings.to("cuda"), image_embeddings=image_embeddings)\npipeline.release_oneshot_encoders()\nsay(f"cache ready, gpu allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")\n\nsay("\\n== generation")\nframes = []\noffset = 0\nfor index in range(N_CHUNKS):\n    count = frame_counts[index]\n    chunk = hdmap_unit[offset:offset + count].mul(2).sub(1).reshape(1, 1, count, 3, HEIGHT, WIDTH).to("cuda", torch.bfloat16)\n    offset += count\n    started = stamp()\n    output = pipeline.generate(autoregressive_index=index, cache=cache, input=chunk)\n    middle = stamp()\n    pipeline.finalize(autoregressive_index=index, cache=cache)\n    finished = stamp()\n    dump_enabled["on"] = False\n    say(f"chunk {index}: {count} frames, generate {middle - started:.2f} s, finalize {finished - middle:.2f} s, "\n        f"output {tuple(output.shape)} range {output.min().item():.3f}..{output.max().item():.3f}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")\n    pixels = output[0, 0].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()\n    frames.append(pixels)\n    Image.fromarray(pixels[-1]).save(os.path.join(OUT, f"generated_chunk{index}_last.png"))\n\nvideo = np.concatenate(frames)\nnp.savez_compressed(os.path.join(OUT, "generated_frames.npz"), frames=video)\ntry:\n    import imageio.v3 as iio\n    iio.imwrite(os.path.join(OUT, "generated.mp4"), video, fps=30, codec="libx264", quality=9)\n    side = np.concatenate([video, hdmap_uint8[:len(video)]], axis=1)\n    iio.imwrite(os.path.join(OUT, "generated_with_hdmap.mp4"), side, fps=30, codec="libx264", quality=9)\n    say("mp4 written")\nexcept Exception as error:\n    say(f"mp4 failed: {error!r}")\n\nsummary = {}\nfor label, value in dumps.items():\n    torch.save(value.to(torch.bfloat16) if value.numel() > 4_000_000 else value, os.path.join(OUT, f"dit_{label}.pt"))\n    summary[label] = {"shape": list(value.shape), "mean": value.mean().item(), "std": value.std().item()}\njson.dump(summary, open(os.path.join(OUT, "dit_dumps.json"), "w"), indent=1)\nsay(f"dumps: {json.dumps(summary)}")\nsay("driver done")\n'
COMMIT = "34ce7b7ff0750d43aadd9a4198f046b1948b4b53"


def say(text):
    print(text, flush=True)
    LOG.write(str(text) + "\n")
    LOG.flush()


def run(title, command, timeout=3600, keep=120):
    say(f"\n===== {title}\n$ {command}")
    started = time.time()
    process = subprocess.Popen(command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line.rstrip())
            if time.time() - started > timeout:
                process.kill()
                lines.append(f"[killed after {timeout} s]")
                break
        process.wait()
    finally:
        if len(lines) > keep:
            lines = lines[:20] + ["... (%d lines cut) ..." % (len(lines) - keep)] + lines[-(keep - 20):]
        say("\n".join(lines))
        say(f"[exit {process.returncode}, {time.time() - started:.1f} s]")
    return process.returncode


checkpoint = glob.glob("/kaggle/input/**/2b_res720p_30fps_i2v_hdmap_distilled.pt", recursive=True)[0]
say(f"checkpoint {checkpoint} {os.path.getsize(checkpoint)}")

inputs = glob.glob("/kaggle/input/**/first_frame.png", recursive=True)
scene = os.path.dirname(inputs[0])
say(f"inputs at {scene}: {sorted(os.listdir(scene))}")

src = os.path.join(WORK, "flashdreams")
run("clone", f"git clone -q https://github.com/NVIDIA/flashdreams.git {src} && git -C {src} checkout -q {COMMIT} && git -C {src} rev-parse HEAD")
run("apt", "apt-get -qq update >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get -qq install -y yasm nasm >/dev/null 2>&1; yasm --version | head -1; nasm -v")
run("uv", "pip install -q uv && uv --version")
code = run("uv sync", f"cd {src} && UV_LINK_MODE=copy uv sync --python 3.12 --package flashdreams-omnidreams --inexact 2>&1 | grep -v -E '^ [+~-] ' | tail -60; exit ${{PIPESTATUS[0]}}", timeout=3000)
if code != 0 or not os.path.exists(os.path.join(src, ".venv", "bin", "python")):
    say("uv sync failed, stopping")
    LOG.close()
    raise SystemExit(1)
run("env check", f"cd {src} && uv run --no-sync python -c \"import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); import omnidreams.config; print('omnidreams import ok')\"")

os.makedirs("/tmp/cudalink", exist_ok=True)
if not os.path.exists("/tmp/cudalink/libcuda.so"):
    os.symlink("/usr/local/cuda-13.0/compat/libcuda.so.1", "/tmp/cudalink/libcuda.so")
with open(os.path.join(WORK, "driver.py"), "w") as handle:
    handle.write(DRIVER_SOURCE)
env = (
    f"STEP_OUT={OUT} STEP_CHECKPOINT={checkpoint} STEP_SCENE={scene} STEP_CHUNKS=52 "
    "LIBRARY_PATH=/tmp/cudalink:$LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)
run("driver", f"cd {src} && {env} uv run --no-sync python {WORK}/driver.py 2>&1 | grep -v -E -e '-MMD|-MD -MF'", timeout=7200, keep=200)
say("\ndemo done")
LOG.close()
