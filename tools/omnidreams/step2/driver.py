import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

OUT = os.environ["STEP_OUT"]
CHECKPOINT = os.environ["STEP_CHECKPOINT"]
SCENE = os.environ["STEP_SCENE"]
N_CHUNKS = int(os.environ.get("STEP_CHUNKS", "3"))
CAMERA = "camera_front_wide_120fov"
WIDTH, HEIGHT = 1280, 704


def say(text):
    print(text, flush=True)


def stamp():
    torch.cuda.synchronize()
    return time.time()


say(f"python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda}")
say(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
probe = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
say(f"bf16 matmul probe: {(probe @ probe).float().abs().mean().item():.4f}")

import flashdreams.core.attention.cp as cp_module


QUERY_BLOCK = int(os.environ.get("STEP_QUERY_BLOCK", "1024"))


def plain_sdpa(query, key, value, return_lse=False):
    if query.shape[2] <= QUERY_BLOCK:
        return F.scaled_dot_product_attention(query, key, value), None
    parts = [
        F.scaled_dot_product_attention(query[:, :, start:start + QUERY_BLOCK], key, value)
        for start in range(0, query.shape[2], QUERY_BLOCK)
    ]
    return torch.cat(parts, dim=2), None


cp_module.torch_sdpa_cudnn = plain_sdpa
cp_module.torch_sdpa_flash = plain_sdpa
say("attention: cudnn/flash replaced by F.scaled_dot_product_attention")

from flashdreams.infra.config import derive_config
from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG

config = derive_config(
    OMNIDREAMS_PIPELINE_CONFIG,
    name="omnidreams-t4-reference",
    text_encoder=dict(run_on_cpu=True),
    image_encoder=dict(use_cuda_graph=False),
    encoder=dict(use_cuda_graph=False),
    decoder=dict(use_cuda_graph=False),
    diffusion_model=dict(transformer=dict(use_cuda_graph=False, checkpoint_path=CHECKPOINT)),
)

started = time.time()
pipeline = config.setup().to("cuda").eval()
say(f"pipeline setup {time.time() - started:.1f} s, gpu allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

if torch.cuda.device_count() > 1 and os.environ.get("STEP_ENCODER_GPU1", "1") == "1":
    encoder = pipeline.encoder
    encoder.to("cuda:1")
    encoder_forward = encoder.forward

    def forward_on_second_gpu(*args, **kwargs):
        if "input" in kwargs:
            kwargs["input"] = kwargs["input"].to("cuda:1")
        args = tuple(item.to("cuda:1") if isinstance(item, torch.Tensor) else item for item in args)
        with torch.cuda.device(1):
            result = encoder_forward(*args, **kwargs)
        return result.to("cuda:0")

    encoder.forward = forward_on_second_gpu
    say("hdmap encoder moved to cuda:1")

network = None
for name, module in pipeline.named_modules():
    if type(module).__name__ == "CosmosDiTNetwork":
        network = module
        say(f"network at {name}")
        break

dumps = {}
dump_enabled = {"on": True}


def make_hook(label):
    def hook(module, inputs, output):
        if not dump_enabled["on"] or label in dumps:
            return
        value = output[0] if isinstance(output, tuple) else output
        dumps[label] = value.detach().to("cpu", torch.float32).clone()
        if label.endswith("in") is False and inputs and isinstance(inputs[0], torch.Tensor) and (label + ".input") not in dumps:
            dumps[label + ".input"] = inputs[0].detach().to("cpu", torch.float32).clone()
    return hook


if network is not None:
    network.x_embedder.register_forward_hook(make_hook("x_embedder"))
    network.blocks[0].register_forward_hook(make_hook("block0"))
    network.blocks[-1].register_forward_hook(make_hook("block27"))
    network.final_layer.register_forward_hook(make_hook("final_layer"))

say("\n== conditioning")
from omnidreams.impl.conditioning.renderer import LudusRenderer, load_and_attach_ludus_scene
from omnidreams.impl.conditioning.world_scenario.data_loaders import load_scene
from omnidreams.impl.conditioning.world_scenario.settings import SETTINGS

from pathlib import Path

from omnidreams.impl.scenes import prepare_clipgt_dir

clipgt = os.path.join(SCENE, "clipgt")
staged, staged_holder = prepare_clipgt_dir(Path(clipgt))
say(f"clipgt staged at {staged}")
started = time.time()
scene = load_scene(
    str(staged),
    camera_names=[CAMERA],
    max_frames=-1,
    input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
    resize_resolution_hw=[HEIGHT, WIDTH],
)
scene = load_and_attach_ludus_scene(clipgt, scene, device=torch.device("cuda"))
renderer = LudusRenderer(scene_data=scene, camera_models={CAMERA: scene.camera_models[CAMERA]}, device=torch.device("cuda"))
say(f"scene loaded {time.time() - started:.1f} s, ego poses {scene.num_frames}, frame rate {scene.frame_rate}")

first_frames = sorted(glob.glob(os.path.join(SCENE, "frames", CAMERA, "*.jpeg")))
first_timestamp = int(os.path.splitext(os.path.basename(first_frames[0]))[0])
timestamps = scene.timestamps
start = int(np.searchsorted(timestamps, first_timestamp))
say(f"first frame ts {first_timestamp}, ego ts range {timestamps[0]}..{timestamps[-1]}, start index {start} (ts {timestamps[start]})")

frame_counts = [pipeline.get_num_output_frames(i) for i in range(N_CHUNKS)]
total_frames = sum(frame_counts)
indices = list(range(start, start + total_frames))
extrinsic = np.asarray(scene.camera_extrinsics[CAMERA], dtype=np.float32)
poses = np.stack([scene.ego_poses[i].transformation_matrix @ extrinsic for i in indices]).astype(np.float32)
started = time.time()
hdmap = renderer.render_all_frames_and_cameras(
    camera_names=[CAMERA],
    camera_poses_per_camera={CAMERA: torch.from_numpy(poses).cuda()},
    frame_timestamps_us=[int(timestamps[i]) for i in indices],
)
say(f"hdmap rendered {tuple(hdmap.shape)} {hdmap.dtype} min {hdmap.min().item()} max {hdmap.max().item()} in {time.time() - started:.1f} s")
hdmap = hdmap[0]
if hdmap.dtype == torch.uint8:
    hdmap_unit = hdmap.float() / 255.0
elif hdmap.max() > 1.5:
    hdmap_unit = hdmap.float() / 255.0
else:
    hdmap_unit = hdmap.float().clamp(0, 1)
hdmap_uint8 = (hdmap_unit * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
np.savez_compressed(os.path.join(OUT, "hdmap_frames.npz"), frames=hdmap_uint8, timestamps=np.array([timestamps[i] for i in indices]), poses=poses)
Image.fromarray(hdmap_uint8[0]).save(os.path.join(OUT, "hdmap_0000.png"))

with Image.open(first_frames[0]) as image:
    first_rgb = np.asarray(image.convert("RGB").resize((WIDTH, HEIGHT), resample=Image.Resampling.BILINEAR), dtype=np.uint8)
Image.fromarray(first_rgb).save(os.path.join(OUT, "first_frame_1280x704.png"))
overlay = first_rgb.copy()
mask = hdmap_uint8[0].max(axis=2) > 0
overlay[mask] = (0.4 * overlay[mask] + 0.6 * hdmap_uint8[0][mask]).astype(np.uint8)
Image.fromarray(overlay).save(os.path.join(OUT, "overlay_0000.png"))

say("\n== text and first frame")
prompt = open(os.path.join(SCENE, "prompt1.txt")).read().strip()
say(f"prompt: {prompt[:200]}")
image = torch.from_numpy(first_rgb).permute(2, 0, 1).float().div(127.5).sub(1.0)
image = image.reshape(1, 1, 1, 3, HEIGHT, WIDTH).to("cuda", torch.bfloat16)
started = time.time()
text_embeddings = torch.stack([pipeline.text_encoder([prompt])], dim=0)
say(f"text encoded {tuple(text_embeddings.shape)} {text_embeddings.dtype} in {time.time() - started:.1f} s")
torch.save(text_embeddings.detach().cpu(), os.path.join(OUT, "text_embeddings.pt"))
started = stamp()
image_embeddings = pipeline.image_encoder(image)
say(f"image encoded {tuple(image_embeddings.shape)} in {stamp() - started:.2f} s")
torch.save(image_embeddings.detach().float().cpu(), os.path.join(OUT, "image_latent.pt"))
cache = pipeline.initialize_cache_from_embeddings(text_embeddings=text_embeddings.to("cuda"), image_embeddings=image_embeddings)
pipeline.release_oneshot_encoders()
say(f"cache ready, gpu allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

say("\n== generation")
frames = []
offset = 0
for index in range(N_CHUNKS):
    count = frame_counts[index]
    chunk = hdmap_unit[offset:offset + count].mul(2).sub(1).reshape(1, 1, count, 3, HEIGHT, WIDTH).to("cuda", torch.bfloat16)
    offset += count
    started = stamp()
    output = pipeline.generate(autoregressive_index=index, cache=cache, input=chunk)
    middle = stamp()
    pipeline.finalize(autoregressive_index=index, cache=cache)
    finished = stamp()
    dump_enabled["on"] = False
    say(f"chunk {index}: {count} frames, generate {middle - started:.2f} s, finalize {finished - middle:.2f} s, "
        f"output {tuple(output.shape)} range {output.min().item():.3f}..{output.max().item():.3f}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    pixels = output[0, 0].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    frames.append(pixels)
    Image.fromarray(pixels[-1]).save(os.path.join(OUT, f"generated_chunk{index}_last.png"))

video = np.concatenate(frames)
np.savez_compressed(os.path.join(OUT, "generated_frames.npz"), frames=video)
try:
    import imageio.v3 as iio
    iio.imwrite(os.path.join(OUT, "generated.mp4"), video, fps=30, codec="libx264", quality=9)
    side = np.concatenate([video, hdmap_uint8[:len(video)]], axis=1)
    iio.imwrite(os.path.join(OUT, "generated_with_hdmap.mp4"), side, fps=30, codec="libx264", quality=9)
    say("mp4 written")
except Exception as error:
    say(f"mp4 failed: {error!r}")

summary = {}
for label, value in dumps.items():
    torch.save(value.to(torch.bfloat16) if value.numel() > 4_000_000 else value, os.path.join(OUT, f"dit_{label}.pt"))
    summary[label] = {"shape": list(value.shape), "mean": value.mean().item(), "std": value.std().item()}
json.dump(summary, open(os.path.join(OUT, "dit_dumps.json"), "w"), indent=1)
say(f"dumps: {json.dumps(summary)}")
say("driver done")
