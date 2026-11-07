import os
import sys

import numpy as np

ref_dir, out_dir = sys.argv[1], sys.argv[2]
shapes = {}
for line in open(os.path.join(ref_dir, "shapes.txt")):
    name, *dims = line.split()
    shapes[name] = tuple(int(d) for d in dims)


def ref(name):
    return np.fromfile(os.path.join(ref_dir, name + ".f32"), np.float32).reshape(shapes[name])


def ours(name, shape):
    return np.fromfile(os.path.join(out_dir, name + ".f32"), np.float32).reshape(shape)


STAGES = [
    ("rgb_cond", lambda r: r[0, 0], "rgb_cond", (512, 512, 3)),
    ("vit_embed", lambda r: r[0], "vit_embed", (1025, 768)),
    ("vit_layer0", lambda r: r[0], "vit_layer0", (1025, 768)),
    ("vit_tokens", lambda r: r[0, 0].T, "vit_tokens", (1025, 768)),
    ("bb_proj_in", lambda r: r[0], "bb_proj_in", (3072, 1024)),
    ("bb_block0", lambda r: r[0], "bb_block0", (3072, 1024)),
    ("bb_out", lambda r: r[0].T, "bb_out", (3072, 1024)),
    ("scene_codes", lambda r: r[0], "scene_codes", (3, 40, 64, 64)),
]

worst = 0.0
for ref_name, view, our_name, shape in STAGES:
    if not os.path.exists(os.path.join(out_dir, our_name + ".f32")):
        continue
    r = view(ref(ref_name)).astype(np.float64)
    o = ours(our_name, shape).astype(np.float64)
    diff = np.abs(o - r)
    rel = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)
    worst = max(worst, rel)
    print(f"{ref_name:12s} max abs {diff.max():.3e} at {np.unravel_index(diff.argmax(), diff.shape)}"
          f"  rel L2 {rel:.3e}  ref max {np.abs(r).max():.3e}")
print(f"worst rel L2 {worst:.3e}")
