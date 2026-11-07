"""Reference for the Wan 2.1 VAE encoder OmniDreams uses (lightvaew2_1.pth) in float64 on the CPU.

FlashDreams flashdreams/recipes/wan/autoencoder/vae.py: CausalConv3d.forward / cache_step (lines
153-219), RMS_norm (222-252), Resample downsample2d / downsample3d (268-349), ResidualBlock
(492-520), AttentionBlock (523-552), Encoder3d (716-824, pruning_rate 0.75), WanVAE.encode
(1164-1216: a 1-frame seed, normalize_state_for_body, then 4-frame chunks, conv1, (mu - mean) *
inv_std). Two calls: 1 + 4 frames, then 4 frames. Usage: wanvae_reference.py <pth> <out dir> [h w]"""
import os
import sys

import torch
import torch.nn.functional as F

path, folder = sys.argv[1], sys.argv[2]
H, W = (int(v) for v in sys.argv[3:5]) if len(sys.argv) >= 5 else (64, 96)
os.makedirs(folder, exist_ok=True)
sd = {k: v.double() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921], dtype=torch.float64)
STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160], dtype=torch.float64)
CACHE_T = 2
state = {}


def causal_conv(name, x, k_t, pad):
    # CausalConv3d: cat the cached frames in front, pad what is left of the time padding with zeros
    w, b = sd[f"{name}.weight"], sd[f"{name}.bias"]
    time_pad = 2 * pad
    prev = state.get(name)
    if prev is not None and time_pad > 0:
        x_in = torch.cat([prev, x], dim=2)
        time_pad = max(0, time_pad - prev.shape[2])
    else:
        x_in = x
    x_in = F.pad(x_in, (pad, pad, pad, pad, time_pad, 0))
    return F.conv3d(x_in, w, b)


def causal_step(name, x, pad=1):
    prev = state.get(name)
    out = causal_conv(name, x, 3, pad)
    tail = x[:, :, -CACHE_T:]
    if tail.shape[2] < CACHE_T and prev is not None:
        tail = torch.cat([prev[:, :, -1:], tail], dim=2)
    state[name] = tail
    return out


def rms(x, name):
    g = sd[f"{name}.gamma"]
    return F.normalize(x, dim=1) * x.shape[1] ** 0.5 * g


def residual(prefix, x):
    h = F.conv3d(x, sd[f"{prefix}.shortcut.weight"], sd[f"{prefix}.shortcut.bias"]) \
        if f"{prefix}.shortcut.weight" in sd else x
    y = causal_step(f"{prefix}.residual.2", F.silu(rms(x, f"{prefix}.residual.0")))
    y = causal_step(f"{prefix}.residual.6", F.silu(rms(y, f"{prefix}.residual.3")))
    return y + h


def spatial_down(prefix, x):
    b, c, t, h, w = x.shape
    f = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    f = F.conv2d(F.pad(f, (0, 1, 0, 1)), sd[f"{prefix}.resample.1.weight"], sd[f"{prefix}.resample.1.bias"], stride=2)
    return f.reshape(b, t, c, h // 2, w // 2).permute(0, 2, 1, 3, 4)


def down3d(prefix, x):
    x = spatial_down(prefix, x)
    prev = state.get(prefix)
    tail = x[:, :, -1:]
    if prev is not None:
        x = F.conv3d(torch.cat([prev, x], dim=2), sd[f"{prefix}.time_conv.weight"], sd[f"{prefix}.time_conv.bias"],
                     stride=(2, 1, 1))
    state[prefix] = tail
    return x


def attention(prefix, x):
    b, c, t, h, w = x.shape
    f = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    f = F.normalize(f, dim=1) * c ** 0.5 * sd[f"{prefix}.norm.gamma"]
    q, k, v = F.conv2d(f, sd[f"{prefix}.to_qkv.weight"], sd[f"{prefix}.to_qkv.bias"]) \
        .reshape(b * t, 1, c * 3, h * w).permute(0, 1, 3, 2).chunk(3, dim=-1)
    o = F.scaled_dot_product_attention(q, k, v).squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
    o = F.conv2d(o, sd[f"{prefix}.proj.weight"], sd[f"{prefix}.proj.bias"])
    return o.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4) + x


def encoder(x):
    x = causal_step("encoder.conv1", x)
    for i in (0, 1):
        x = residual(f"encoder.downsamples.{i}", x)
    x = spatial_down("encoder.downsamples.2", x)
    for i in (3, 4):
        x = residual(f"encoder.downsamples.{i}", x)
    x = down3d("encoder.downsamples.5", x)
    for i in (6, 7):
        x = residual(f"encoder.downsamples.{i}", x)
    x = down3d("encoder.downsamples.8", x)
    for i in (9, 10):
        x = residual(f"encoder.downsamples.{i}", x)
    x = residual("encoder.middle.0", x)
    x = attention("encoder.middle.1", x)
    x = residual("encoder.middle.2", x)
    return causal_step("encoder.head.2", F.silu(rms(x, "encoder.head.0")))


def normalize_state_for_body():
    for name, prev in list(state.items()):
        if name.endswith((".conv1", ".residual.2", ".residual.6", ".head.2")) and prev.shape[2] < CACHE_T:
            state[name] = torch.cat([prev.new_zeros(*prev.shape[:2], CACHE_T - prev.shape[2], *prev.shape[3:]),
                                     prev], dim=2)


def encode(x, first):
    outs = []
    if first:
        outs.append(encoder(x[:, :, :1]))
        x = x[:, :, 1:]
        normalize_state_for_body()
    for i in range(0, x.shape[2], 4):
        outs.append(encoder(x[:, :, i:i + 4]))
    mu = F.conv3d(torch.cat(outs, dim=2), sd["conv1.weight"], sd["conv1.bias"])[:, :16]
    return (mu - MEAN.view(1, -1, 1, 1, 1)) / STD.view(1, -1, 1, 1, 1)


generator = torch.Generator().manual_seed(5)
with open(f"{folder}/shape.txt", "w") as f:
    f.write(f"{H} {W}\n")
for call, (count, first) in enumerate(((5, True), (4, False))):
    frames = (torch.rand(1, 3, count, H, W, generator=generator, dtype=torch.float64) * 2 - 1).float().double()
    z = encode(frames, first)
    print(f"call {call}: {count} frames -> {z.shape[2]} latent frames, rms {z.pow(2).mean().sqrt().item():.4f}")
    frames[0].permute(1, 0, 2, 3).float().numpy().tofile(f"{folder}/frames_{call}.f32")
    z[0].permute(1, 0, 2, 3).numpy().tofile(f"{folder}/latents_{call}.f64")
