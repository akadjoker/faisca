import mmap
import sys

import torch

ckpt_path, dump_path = sys.argv[1], sys.argv[2]
DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}

state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
if "state_dict" in state and isinstance(state["state_dict"], dict):
    state = state["state_dict"]

rows = [line.rstrip("\n").split("\t") for line in open(dump_path)]
failures = 0
if [r[0] for r in rows] != list(state.keys()):
    print("FAIL: names or order differ")
    failures += 1

with open(ckpt_path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for name, dtype, shape, offset, n_bytes in rows:
        t = state.get(name)
        if t is None:
            print(f"FAIL {name}: not in torch.load")
            failures += 1
            continue
        want_shape = ",".join(str(d) for d in t.shape)
        if dtype != DTYPES.get(t.dtype) or shape != want_shape:
            print(f"FAIL {name}: {dtype} {shape} vs {t.dtype} {want_shape}")
            failures += 1
            continue
        raw = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        offset, n_bytes = int(offset), int(n_bytes)
        if n_bytes != len(raw) or mm[offset : offset + n_bytes] != raw:
            print(f"FAIL {name}: bytes differ")
            failures += 1

print(f"{len(rows)} tensors checked against torch {torch.__version__}, {failures} failures")
sys.exit(1 if failures else 0)
