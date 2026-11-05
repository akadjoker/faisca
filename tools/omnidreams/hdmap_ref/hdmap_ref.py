"""Reference hdmap frames from ludus alone, for the alignment check and for Lucent.

Draws the conditioning video of a ClipGT scene the way the OmniDreams pipeline does
(conditioning/renderer.py LudusRenderer settings), with the camera pose built as
apps/interactive_drive rasterizer.py does: rig to world from the ludus ego track times
ludus' sensor_to_rig, FLU, inverted for ludus. Writes the frames, the overlay of frame 0
on the real first image, and an npz with frames, timestamps and poses.

--poses-npz renders the camera to world poses and timestamps stored in an npz instead
(the step2 output), to reproduce an earlier drawing.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from ludus_renderer import load_scene
from ludus_renderer.render_utils import SceneAdapter
from ludus_renderer.torch import LudusCudaTimestampedContext
from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="extracted scene folder with clipgt/ and frames/")
parser.add_argument("--out", required=True)
parser.add_argument("--camera", default="camera:front:wide:120fov")
parser.add_argument("--frames", type=int, default=21)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=704)
parser.add_argument("--poses-npz", default=None)
parser.add_argument("--start-us", type=int, default=None, help="first frame time, default the first photo")
parser.add_argument("--fps", type=float, default=30.0)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device("cuda")

started = time.time()
clipgt = load_scene(
    os.path.join(args.scene, "clipgt"),
    device=device,
    target_resolution=(args.width, args.height),
    include_ego_trajectory=False,
    include_ego_obstacle=False,
    include_dynamic_obstacles=True,
    simplify_dual_lane_lines=False,
)
scene = SceneAdapter(clipgt)
print(f"scene loaded in {time.time() - started:.1f} s, cameras {list(clipgt.camera_name_to_id)}")

ctx = LudusCudaTimestampedContext(device=device)
ctx.set_depth_scaling(True)
ctx.set_msaa_samples(4)
ctx.set_max_tessellation_levels(cube=0)
ctx.upload_cameras(clipgt.cameras)
scene_id = ctx.upload_scene(clipgt.timestamped_scene)
camera_id = clipgt.camera_name_to_id[args.camera]
camera = clipgt.cameras[camera_id]
print(f"camera {args.camera}: size {camera.image_size.tolist()} centre {camera.principal_point.tolist()} "
      f"max ray angle {float(camera.max_ray_angle):.4f} fw_poly {camera.fw_poly.tolist()}")

frame_dir = os.path.join(args.scene, "frames", args.camera.replace(":", "_"))
images = sorted(glob.glob(os.path.join(frame_dir, "*.jpeg")))
first_image = images[0] if images else None
if args.start_us is not None:
    first_timestamp = args.start_us
elif first_image:
    first_timestamp = int(os.path.splitext(os.path.basename(first_image))[0])
else:
    first_timestamp = int(scene.ego_tracks._ego_track.timestamps[0].item())

if args.poses_npz:
    stored = np.load(args.poses_npz)
    timestamps = [int(t) for t in stored["timestamps"]]
    camera_to_world = torch.from_numpy(stored["poses"]).float().to(device)
else:
    timestamps = [first_timestamp + round(k * 1e6 / args.fps) for k in range(args.frames)]
    rig_to_world = scene.ego_tracks.get_transforms_at_timestamp(torch.tensor(timestamps, dtype=torch.int64))[:, 0]
    sensor_to_rig = scene.cameras[args.camera].sensor_to_rig.to(rig_to_world.device)
    camera_to_world = (rig_to_world @ sensor_to_rig).float().to(device)
count = len(timestamps)
print(f"first image ts {first_timestamp}, rendering {count} frames from ts {timestamps[0]}")
print(f"camera position first {camera_to_world[0, :3, 3].tolist()} last {camera_to_world[-1, :3, 3].tolist()}")

started = time.time()
images = ctx.render(
    torch.full((count,), scene_id, dtype=torch.int32, device=device),
    torch.full((count,), camera_id, dtype=torch.int32, device=device),
    torch.tensor(timestamps, dtype=torch.int64, device=device),
    torch.full((count,), CAMERA_TYPE_REGULAR, dtype=torch.int32, device=device),
    torch.linalg.inv(camera_to_world),
    resolution=(args.height, args.width),
)
torch.cuda.synchronize()
print(f"rendered {tuple(images.shape)} {images.dtype} in {time.time() - started:.3f} s")
rgb = images[:, :, :, :3]
if ctx.needs_vflip:
    rgb = rgb.flip(1)
if rgb.dtype != torch.uint8:
    rgb = (rgb.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
frames = rgb.cpu().numpy()

for index, frame in enumerate(frames):
    Image.fromarray(frame).save(os.path.join(args.out, f"hdmap_{index:04d}.png"))
np.savez_compressed(os.path.join(args.out, "hdmap_frames.npz"), frames=frames,
                    timestamps=np.array(timestamps), poses=camera_to_world.cpu().numpy())

if first_image and args.start_us is None:
    with Image.open(first_image) as image:
        photo = np.asarray(image.convert("RGB").resize((args.width, args.height), resample=Image.Resampling.BILINEAR), dtype=np.uint8)
    overlay = photo.copy()
    mask = frames[0].max(axis=2) > 0
    overlay[mask] = (0.4 * overlay[mask] + 0.6 * frames[0][mask]).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(args.out, "overlay_0000.png"))
print(f"written to {args.out}")
