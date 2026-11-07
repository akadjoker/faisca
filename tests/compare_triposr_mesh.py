import os
import sys

import numpy as np

ref_dir, out_dir, res = sys.argv[1], sys.argv[2], int(sys.argv[3])
R = 0.87

d_ref = np.fromfile(os.path.join(ref_dir, "density_act.f32"), np.float32).astype(np.float64)
d_our = np.fromfile(os.path.join(out_dir, "density_act.f32"), np.float32).astype(np.float64)
rel = np.abs(d_our - d_ref) / np.maximum(np.abs(d_ref), 1e-6)
print(f"density {d_ref.size} points: max rel {rel.max():.3e}, rel L2 {np.linalg.norm(d_our - d_ref) / np.linalg.norm(d_ref):.3e}")
side = (d_ref > 25.0) != (d_our > 25.0)
print(f"points on a different side of the threshold: {side.sum()}")

def load(d, faces_name, faces_type):
    v = np.fromfile(os.path.join(d, "mc_vertices.f32"), np.float32).reshape(-1, 3)
    f = np.fromfile(os.path.join(d, faces_name), faces_type).reshape(-1, 3).astype(np.int64)
    c = np.fromfile(os.path.join(d, "mc_colors.f32"), np.float32).reshape(-1, 3)
    return v, f, c

v_ref, f_ref, c_ref = load(ref_dir, "mc_faces.i32", np.int32)
v_our, f_our, c_our = load(out_dir, "mc_faces.u32", np.uint32)
print(f"vertices {len(v_our)} (reference {len(v_ref)}), faces {len(f_our)} (reference {len(f_ref)})")

# torchmcubes joins its z slabs in OpenMP order: compare geometry, not order. A vertex
# key is its grid space position rounded to 1e-3 of a cell
def keys(v):
    g = (v.astype(np.float64) + R) / (2 * R) * (res - 1)
    return [tuple(k) for k in np.round(g * 1000).astype(np.int64)]

k_ref, k_our = keys(v_ref), keys(v_our)
index_ref = {k: i for i, k in enumerate(k_ref)}
matched = [index_ref.get(k, -1) for k in k_our]
unmatched = sum(1 for m in matched if m < 0)
print(f"our vertices without a reference vertex at the same key: {unmatched}")
pairs = np.array([(i, m) for i, m in enumerate(matched) if m >= 0])
if len(pairs):
    dv = np.abs(v_our[pairs[:, 0]].astype(np.float64) - v_ref[pairs[:, 1]])
    dc = np.abs(c_our[pairs[:, 0]].astype(np.float64) - c_ref[pairs[:, 1]])
    print(f"matched vertices: position max diff {dv.max():.3e}, exact {np.all(dv == 0)}; color max diff {dc.max():.3e}")

def triangles(f, k):
    out = []
    for a, b, c in f:
        t = (k[a], k[b], k[c])
        i = t.index(min(t))
        out.append(t[i:] + t[:i])
    return sorted(out)

same = triangles(f_our, k_our) == triangles(f_ref, k_ref)
print(f"triangles (oriented, any order) identical: {same}")
