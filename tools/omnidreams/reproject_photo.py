"""Turns a dashcam photo into what the OmniDreams front camera would see from the same spot.

The model was trained on one camera (camera:front:wide:120fov, f-theta, looking along the car).
A dashcam is a different lens pointing a little elsewhere; drawn as it is, the HD map has to be
tilted to match and the model drifts. Here every pixel of the f-theta camera (backward polynomial
of the ClipGT calibration scaled to the target size as ludus does, as lucent
tools/omnidreams/pinhole_to_ftheta.py) becomes a ray, the ray is turned so that the photo's
vanishing point of the road becomes the f-theta optical axis, and the photo is sampled as a pinhole
of horizontal field of view --hfov. Rays outside the photo are filled from its nearest edge.
Usage: reproject_photo.py <photo> <out.png> --calibration ... --vanish X Y --hfov DEG
"""
import argparse
import json
import math

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("photo")
parser.add_argument("out")
parser.add_argument("--calibration", required=True)
parser.add_argument("--camera", default="camera:front:wide:120fov")
parser.add_argument("--vanish", type=float, nargs=2, required=True, help="vanishing point of the road in the photo, px")
parser.add_argument("--hfov", type=float, required=True, help="horizontal field of view of the photo, degrees")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=704)
args = parser.parse_args()

table = pq.read_table(args.calibration).column("calibration_estimate").to_pylist()
props = next(s["properties"] for s in json.loads(table[0]["rig_json"])["rig"]["sensors"] if s["name"] == args.camera)
scale_x, scale_y = args.width / float(props["width"]), args.height / float(props["height"])
poly_scale = 0.5 * (scale_x + scale_y)
cx, cy = float(props["cx"]) * scale_x, float(props["cy"]) * scale_y
backward = [float(c) for c in props["polynomial"].split()]

v, u = np.mgrid[0:args.height, 0:args.width].astype(np.float64)
dx, dy = u + 0.5 - cx, v + 0.5 - cy
radius = np.hypot(dx, dy)
theta = sum(c * (radius / poly_scale) ** p for p, c in enumerate(backward))
with np.errstate(invalid="ignore", divide="ignore"):
    ux = np.where(radius > 1e-9, dx / radius, 0.0)
    uy = np.where(radius > 1e-9, dy / radius, 0.0)
ray = np.stack([np.sin(theta) * ux, np.sin(theta) * uy, np.cos(theta)], axis=-1)  # RDF

photo = np.asarray(Image.open(args.photo).convert("RGB"), dtype=np.float32)
ph, pw = photo.shape[:2]
focal = 0.5 * pw / math.tan(math.radians(args.hfov) * 0.5)
# the road direction in the photo's camera; turn it onto +z with yaw then pitch (no roll)
d = np.array([(args.vanish[0] - 0.5 * pw) / focal, (args.vanish[1] - 0.5 * ph) / focal, 1.0])
yaw = math.atan2(d[0], d[2])
pitch = math.atan2(d[1], math.hypot(d[0], d[2]))
cy_, sy_ = math.cos(yaw), math.sin(yaw)
cp, sp = math.cos(pitch), math.sin(pitch)
yaw_m = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
pitch_m = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
rotation = yaw_m @ pitch_m  # f-theta ray -> photo camera ray
r = ray @ rotation.T
px = 0.5 * pw + r[..., 0] / r[..., 2] * focal - 0.5
py = 0.5 * ph + r[..., 1] / r[..., 2] * focal - 0.5
outside = (r[..., 2] <= 0) | (px < 0) | (py < 0) | (px > pw - 1) | (py > ph - 1)
print(f"yaw {math.degrees(yaw):.1f} deg, pitch {math.degrees(pitch):.1f} deg, {100 * outside.mean():.1f}% of the "
      f"pixels outside the photo")
px = np.clip(np.where(r[..., 2] > 0, px, 0.5 * pw), 0, pw - 1.001)
py = np.clip(np.where(r[..., 2] > 0, py, 0.5 * ph), 0, ph - 1.001)
x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
fx, fy = (px - x0)[..., None], (py - y0)[..., None]
out = (photo[y0, x0] * (1 - fx) * (1 - fy) + photo[y0, x0 + 1] * fx * (1 - fy)
       + photo[y0 + 1, x0] * (1 - fx) * fy + photo[y0 + 1, x0 + 1] * fx * fy)
Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(args.out)
