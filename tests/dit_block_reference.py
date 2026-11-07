"""Reference for one OmniDreams (Cosmos-Predict2.5) DiT block with random weights.

The block follows FlashDreams' integrations_v2/omnidreams/impl/transformer/modules.py:
Block.forward (lines 961-1092) without cross-view attention, MultiHeadAttention.query_kv /
_compute_or_update_kv_cache (lines 317-395) on a first chunk (the self-attention cache holds this
chunk only), GPT2FeedForward (lines 48-66, exact GELU); RoPE as rope_kernel.py's docstring
(non-interleaved pairs (d, d + 64), cos / sin in f32 cast to the activation type) over the
frequencies of rope.py _compute_freqs / _freq_components / _cat_freqs (lines 31-167).

Weights, x, the modulation vectors and the text context are rounded to bf16 once, then the block
runs in float64 (the reference), float32 and bfloat16 (as the model runs in PyTorch).
Usage: dit_block_reference.py <out dir> [T H W]"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

D, HEADS, HEAD, CONTEXT, CONTEXT_DIM, MLP = 2048, 16, 128, 512, 1024, 8192
folder = sys.argv[1]
numbers = [a for a in sys.argv[2:] if not a.startswith("--")]
T, H, W = (int(v) for v in numbers) if len(numbers) == 3 else (2, 44, 80)
L = T * H * W
os.makedirs(folder, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
generator = torch.Generator().manual_seed(1234)


def bf16(t):
    return t.to(torch.bfloat16).to(torch.float64)


def linear(n_out, n_in):
    bound = 1.0 / n_in**0.5
    return bf16((torch.rand(n_out, n_in, generator=generator, dtype=torch.float64) * 2 - 1) * bound)


def normal(*shape, std=1.0, mean=0.0):
    return bf16(torch.randn(*shape, generator=generator, dtype=torch.float64) * std + mean)


weights = {
    "self_q": linear(D, D), "self_k": linear(D, D), "self_v": linear(D, D), "self_o": linear(D, D),
    "self_q_norm": normal(HEAD, std=0.1, mean=1.0), "self_k_norm": normal(HEAD, std=0.1, mean=1.0),
    "cross_q": linear(D, D), "cross_k": linear(D, CONTEXT_DIM), "cross_v": linear(D, CONTEXT_DIM),
    "cross_o": linear(D, D),
    "cross_q_norm": normal(HEAD, std=0.1, mean=1.0), "cross_k_norm": normal(HEAD, std=0.1, mean=1.0),
    "mlp1": linear(MLP, D), "mlp2": linear(D, MLP),
}
x = normal(L, D)
# shift, scale, gate for self attention, cross attention and the MLP (Block.forward 1002-1017)
modulation = torch.stack([normal(D, std=0.5), normal(D, std=0.3), normal(D, std=0.5)] * 3)
context = normal(CONTEXT, CONTEXT_DIM)


def compute_freqs(dim, ratio):
    dim_range = torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim
    theta = 10000.0 * ratio ** (dim / (dim - 2))
    return 1.0 / (theta**dim_range)


dim_hw = HEAD // 6 * 2
dim_t = HEAD - 2 * dim_hw
freqs_t = torch.outer(torch.arange(T, dtype=torch.float32), compute_freqs(dim_t, 1.0))
freqs_h = torch.outer(torch.arange(H, dtype=torch.float32), compute_freqs(dim_hw, 3.0))
freqs_w = torch.outer(torch.arange(W, dtype=torch.float32), compute_freqs(dim_hw, 3.0))
angles = torch.cat([
    freqs_t[:, None, None, :].expand(T, H, W, -1),
    freqs_h[None, :, None, :].expand(T, H, W, -1),
    freqs_w[None, None, :, :].expand(T, H, W, -1)], dim=-1).reshape(L, HEAD // 2)


def rope(t, dtype):
    cos = torch.cos(angles).to(t.device).to(dtype)[:, None, :]
    sin = torch.sin(angles).to(t.device).to(dtype)[:, None, :]
    a, b = t[..., : HEAD // 2], t[..., HEAD // 2:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


def rms_norm(t, gamma):
    return F.rms_norm(t, (HEAD,), gamma, eps=1e-6)


def attention(q, k, v):
    if q.dtype == torch.float64:
        out = torch.empty_like(q)
        for h in range(HEADS):
            s = (q[:, h] @ k[:, h].T) / HEAD**0.5
            out[:, h] = torch.softmax(s, dim=-1) @ v[:, h]
        return out
    return F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                          v.transpose(0, 1)[None])[0].transpose(0, 1)


prepared = {}


def block(dtype, keep=False):
    if dtype not in prepared:
        prepared[dtype] = ({name: value.cuda().to(dtype) for name, value in weights.items()}, x.cuda().to(dtype),
                           modulation.cuda().to(dtype), context.cuda().to(dtype))
    w, h_x, mod, ctx = prepared[dtype]

    def modulate(t, branch):
        return F.layer_norm(t, (D,), eps=1e-6) * (1 + mod[3 * branch + 1]) + mod[3 * branch]

    n = modulate(h_x, 0)
    q = rope(rms_norm((n @ w["self_q"].T).view(L, HEADS, HEAD), w["self_q_norm"]), dtype)
    k = rope(rms_norm((n @ w["self_k"].T).view(L, HEADS, HEAD), w["self_k_norm"]), dtype)
    v = (n @ w["self_v"].T).view(L, HEADS, HEAD)
    h_x = h_x + mod[2] * (attention(q, k, v).reshape(L, D) @ w["self_o"].T)

    n = modulate(h_x, 1)
    q = rms_norm((n @ w["cross_q"].T).view(L, HEADS, HEAD), w["cross_q_norm"])
    k = rms_norm((ctx @ w["cross_k"].T).view(CONTEXT, HEADS, HEAD), w["cross_k_norm"])
    v = (ctx @ w["cross_v"].T).view(CONTEXT, HEADS, HEAD)
    h_x = h_x + mod[5] * (attention(q, k, v).reshape(L, D) @ w["cross_o"].T)

    n = modulate(h_x, 2)
    h_x = h_x + mod[8] * (F.gelu(n @ w["mlp1"].T) @ w["mlp2"].T)
    return h_x if keep else h_x.double().cpu()


if "--time" in sys.argv:
    for _ in range(3):
        block(torch.bfloat16, keep=True)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(5):
        block(torch.bfloat16, keep=True)
    end.record()
    torch.cuda.synchronize()
    print(f"torch bf16 block: {start.elapsed_time(end) / 5:.2f} ms (text keys and values recomputed per call, ~1 ms)")
    sys.exit(0)


reference = block(torch.float64)
for name, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
    out = block(dtype)
    error = ((out - reference).norm() / reference.norm()).item()
    print(f"torch {name}: rel L2 to float64 {error:.3e}")
    out.float().numpy().tofile(f"{folder}/torch_{name}.f32")

for name, value in weights.items():
    value.to(torch.bfloat16).view(torch.int16).numpy().tofile(f"{folder}/{name}.bf16")
x.float().numpy().tofile(f"{folder}/x.f32")
modulation.float().numpy().tofile(f"{folder}/modulation.f32")
context.to(torch.bfloat16).view(torch.int16).numpy().tofile(f"{folder}/context.bf16")
angles.numpy().astype(np.float32).tofile(f"{folder}/angles.f32")
reference.numpy().tofile(f"{folder}/reference.f64")
with open(f"{folder}/shape.txt", "w") as f:
    f.write(f"{T} {H} {W}\n")
