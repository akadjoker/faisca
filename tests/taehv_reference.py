"""Reference for the TAEHV decoder OmniDreams uses (lighttaew2_1.pth) in float64 on the CPU.

FlashDreams flashdreams/recipes/taehv: impl.py Decoder.forward / MemBlock.cache_step (whole
chunks, the last input frame of each MemBlock carried to the next chunk), TAEHV.decode (clamp to
[0, 1], the first 3 frames of a rollout dropped) and __init__.py TaehvVAEDecoder.forward (the
lighttae latent scaling z * std + mean). Usage: taehv_reference.py <pth> <out dir> [h w chunks]"""
import os
import sys

import torch
import torch.nn.functional as F

path, folder = sys.argv[1], sys.argv[2]
H, W, CHUNKS = (int(v) for v in sys.argv[3:6]) if len(sys.argv) >= 6 else (22, 40, 2)
T = 2
os.makedirs(folder, exist_ok=True)
sd = {k: v.double() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921], dtype=torch.float64)
STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160], dtype=torch.float64)


def conv(x, name, bias=True):
    w = sd[f"decoder.{name}.weight"]
    return F.conv2d(x, w, sd[f"decoder.{name}.bias"] if bias else None, padding=w.shape[-1] // 2)


state = {}


def memblock(x, i):
    t, c, h, w = x.shape
    prev = state.get(i, torch.zeros(1, c, h, w, dtype=x.dtype))
    past = torch.cat([prev, x[:-1]], 0)
    y = F.relu(conv(torch.cat([x, past], 1), f"{i}.conv.0"))
    y = F.relu(conv(y, f"{i}.conv.2"))
    out = F.relu(conv(y, f"{i}.conv.4") + x)
    state[i] = x[-1:]
    return out


def tgrow(x, i):
    t, c, h, w = x.shape
    return F.conv2d(x, sd[f"decoder.{i}.conv.weight"]).reshape(-1, c, h, w)


def decode(z):
    x = z * STD.view(1, -1, 1, 1) + MEAN.view(1, -1, 1, 1)
    x = torch.tanh(x / 3) * 3
    x = F.relu(conv(x, "1"))
    for i in (3, 4, 5):
        x = memblock(x, i)
    x = tgrow(F.interpolate(x, scale_factor=2, mode="nearest"), 7)
    x = conv(x, "8", bias=False)
    for i in (9, 10, 11):
        x = memblock(x, i)
    x = tgrow(F.interpolate(x, scale_factor=2, mode="nearest"), 13)
    x = conv(x, "14", bias=False)
    for i in (15, 16, 17):
        x = memblock(x, i)
    x = tgrow(F.interpolate(x, scale_factor=2, mode="nearest"), 19)
    x = F.relu(conv(x, "20", bias=False))
    return conv(x, "22").clamp(0, 1)


generator = torch.Generator().manual_seed(99)
with open(f"{folder}/shape.txt", "w") as f:
    f.write(f"{H} {W} {CHUNKS}\n")
for chunk in range(CHUNKS):
    z = torch.randn(T, 16, H, W, generator=generator, dtype=torch.float64)
    out = decode(z)
    if chunk == 0:
        out = out[3:]
    print(f"chunk {chunk}: {out.shape[0]} frames, mean {out.mean().item():.4f}")
    z.float().numpy().tofile(f"{folder}/latents_{chunk}.f32")
    out.numpy().tofile(f"{folder}/frames_{chunk}.f64")
