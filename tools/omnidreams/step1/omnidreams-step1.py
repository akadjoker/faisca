import glob
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile

OUT = "/kaggle/working"
TEMP = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp/work"
os.makedirs(TEMP, exist_ok=True)
LOG = open(os.path.join(OUT, "step1.txt"), "w")
EXPECTED_SIZE = 4118900683
EXPECTED_SHA = "c7cb6f0ce1ca88a721c9ca9e72c690e447dad6efa9aebac22f405477d11dd62f"


def say(text):
    print(text, flush=True)
    LOG.write(str(text) + "\n")
    LOG.flush()


def run(title, command, timeout=1800):
    say(f"\n===== {title}\n$ {command}")
    started = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        lines = (result.stdout + result.stderr).strip().splitlines()
        if len(lines) > 120:
            lines = lines[:30] + ["... (%d lines cut) ..." % (len(lines) - 110)] + lines[-80:]
        say("\n".join(lines))
        say(f"[exit {result.returncode}, {time.time() - started:.1f} s]")
        return result.returncode
    except subprocess.TimeoutExpired:
        say(f"[timeout after {timeout} s]")
        return -1


say("===== 1. checkpoint")
candidates = glob.glob("/kaggle/input/**/2b_res720p_30fps_i2v_hdmap_distilled.pt", recursive=True)
say(f"found: {candidates}")
checkpoint = candidates[0]
size = os.path.getsize(checkpoint)
say(f"size {size} expected {EXPECTED_SIZE} -> {'ok' if size == EXPECTED_SIZE else 'MISMATCH'}")
started = time.time()
digest = hashlib.sha256()
with open(checkpoint, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 24), b""):
        digest.update(block)
say(f"sha256 {digest.hexdigest()} -> {'ok' if digest.hexdigest() == EXPECTED_SHA else 'MISMATCH'} ({time.time() - started:.0f} s)")

say("\n===== 2. tensors")
import torch

with zipfile.ZipFile(checkpoint) as archive:
    names = archive.namelist()
    say(f"zip entries: {len(names)}; first: {names[:5]}")
    pickles = [name for name in names if name.endswith(".pkl")]
    say(f"pickles: {pickles}")

started = time.time()
try:
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    say(f"weights_only load ok in {time.time() - started:.1f} s")
except Exception as error:
    say(f"weights_only load failed: {error!r}")
    state = None


def flatten(prefix, value, out):
    if isinstance(value, torch.Tensor):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


if state is not None:
    say(f"top level type {type(state).__name__}, keys: {list(state.keys())[:20] if isinstance(state, dict) else ''}")
    flat = {}
    flatten("", state, flat)
    tensors = {key: value for key, value in flat.items() if isinstance(value, torch.Tensor)}
    others = {key: repr(value)[:200] for key, value in flat.items() if not isinstance(value, torch.Tensor)}
    total = sum(value.numel() for value in tensors.values())
    dtypes = {}
    for value in tensors.values():
        dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + value.numel()
    say(f"tensors {len(tensors)}, parameters {total} ({total / 1e9:.3f} B), by dtype {dtypes}")
    say(f"non tensor entries: {json.dumps(others)[:2000]}")
    listing = [{"name": key, "shape": list(value.shape), "dtype": str(value.dtype)} for key, value in tensors.items()]
    with open(os.path.join(OUT, "tensors.json"), "w") as handle:
        json.dump(listing, handle, indent=0)
    for item in listing:
        if ".blocks." not in item["name"] or ".blocks.0." in item["name"]:
            say(f"{item['name']} {item['shape']} {item['dtype']}")
    del state, flat, tensors

say("\n===== 3. scene")
scene_dir = os.path.join(TEMP, "scene")
archive_path = os.path.join(TEMP, "scene.usdz")
started = time.time()
from huggingface_hub import hf_hub_download

downloaded = hf_hub_download(
    repo_id="nvidia/omni-dreams-scenes",
    repo_type="dataset",
    filename="scenes/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz",
    local_dir=TEMP,
)
say(f"scene downloaded {os.path.getsize(downloaded)} bytes in {time.time() - started:.0f} s")
with zipfile.ZipFile(downloaded) as archive:
    archive.extractall(scene_dir)
run("scene tree", f"cd {scene_dir} && find . -maxdepth 3 | sort | head -80; echo; du -sh .; find . -name '*.txt' -size -4k -exec sh -c 'echo == $1; cat $1; echo' _ {{}} \\;")
first_frames = sorted(glob.glob(f"{scene_dir}/**/frames/*front*/*", recursive=True))[:1]
say(f"first front frame: {first_frames}")
for path in first_frames:
    shutil.copy(path, os.path.join(OUT, "first_frame" + os.path.splitext(path)[1]))

say("\n===== 4. ludus")
src = os.path.join(TEMP, "src")
run("clone", f"git clone -q --depth 1 https://github.com/NVIDIA/flashdreams.git {src} && git -C {src} rev-parse HEAD")
ludus = f"{src}/integrations_v2/omnidreams/impl/ludus-renderer"
run("install", f"pip install -q 'cmake>=4.0' 'ninja>=1.13.0' nvidia-nvcomp-cu12 loguru && pip install -q --no-deps --no-build-isolation {ludus} 2>&1 | tail -5")
os.makedirs("/tmp/cudalink", exist_ok=True)
if not os.path.exists("/tmp/cudalink/libcuda.so"):
    os.symlink("/usr/local/cuda-13.0/compat/libcuda.so.1", "/tmp/cudalink/libcuda.so")
env = "LIBRARY_PATH=/tmp/cudalink:$LIBRARY_PATH"
clipgt_candidates = [os.path.dirname(path) for path in glob.glob(f"{scene_dir}/**/*.parquet", recursive=True)]
clipgt = sorted(set(clipgt_candidates), key=len)[0] if clipgt_candidates else scene_dir
say(f"clipgt dir guess: {clipgt}")
render_out = os.path.join(TEMP, "render")
os.makedirs(render_out, exist_ok=True)
run("list cameras", f"cd {ludus} && {env} python3 examples/render_hdmap_scene.py --scene {clipgt} --list-cameras 2>&1 | grep -v -E -e '-MMD|-MD -MF'")
run("help", f"cd {ludus} && {env} python3 examples/render_hdmap_scene.py --help 2>&1 | head -60")
run("render frame 0", f"cd {render_out} && {env} python3 {ludus}/examples/render_hdmap_scene.py --scene {clipgt} --frame 0 --msaa 4 2>&1 | tail -20")
run("render sequence", f"cd {render_out} && {env} timeout 900 python3 {ludus}/examples/render_hdmap_scene.py --scene {clipgt} --sequence --fps 30 --msaa 4 --output-format mp4 2>&1 | tail -20")
run("render outputs", f"find {render_out} {ludus} -newer {archive_path if os.path.exists(archive_path) else downloaded} \\( -name '*.png' -o -name '*.mp4' -o -name '*.jpg' \\) 2>/dev/null | head -20")
for path in glob.glob(f"{render_out}/**/*", recursive=True) + glob.glob(f"{ludus}/output*/**/*", recursive=True):
    if os.path.isfile(path) and path.endswith((".png", ".mp4")) and os.path.getsize(path) < 200 * 2**20:
        shutil.copy(path, os.path.join(OUT, "render_" + os.path.basename(path)))

say("\nstep1 done")
LOG.close()
