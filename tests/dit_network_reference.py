"""Reference for the OmniDreams DiT network over several chunks, random bf16 weights.

CosmosDiTNetwork.forward (FlashDreams integrations_v2/omnidreams/impl/transformer/network.py
479-568) with Timesteps / TimestepEmbedding / FinalLayer (modules.py 69-252), the blocks as
dit_block_reference.py, the self attention caches with FlashDreams' own BlockKVCache
(flashdreams/core/attention/kvcache.py, loaded from its file) and 3 passes per chunk on the same
chunk index (timesteps 1000, 500, 128 as the distilled sampler, docs/omnidreams.md).
Runs in float64 (the reference) and bfloat16; writes the weights, inputs and outputs. With
--checkpoint <.pt> the weights are the real ones (all 28 blocks, float64 on the CPU, the two
update_parameters_after_loading_checkpoint changes applied) and are not written out.
Usage: dit_network_reference.py <flashdreams dir> <out dir> [T H W blocks window_chunks chunks]
       [--checkpoint <.pt>]"""
import importlib.util
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

checkpoint = None
if "--checkpoint" in sys.argv:
    at = sys.argv.index("--checkpoint")
    checkpoint = sys.argv[at + 1]
    del sys.argv[at:at + 2]
flashdreams, folder = sys.argv[1], sys.argv[2]
T, H, W, BLOCKS, WINDOW, CHUNKS = (int(v) for v in sys.argv[3:9]) if len(sys.argv) >= 9 else (2, 16, 24, 2, 3, 5)
if checkpoint is not None:
    BLOCKS = 28
spec = importlib.util.spec_from_file_location(
    "kvcache", f"{flashdreams}/flashdreams/flashdreams/core/attention/kvcache.py")
kvcache = importlib.util.module_from_spec(spec)
sys.modules["kvcache"] = kvcache
spec.loader.exec_module(kvcache)

D, HEADS, HEAD, LORA, MLP = 2048, 16, 128, 256, 8192
TEXT, TEXT_IN, TEXT_W = 512, 100352, 1024
L = T * H * W
TIMESTEPS = (1000.0, 500.0, 128.0)
os.makedirs(folder, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
generator = torch.Generator().manual_seed(4321)


def bf16(t):
    return t.to(torch.bfloat16).to(torch.float64)


def linear(n_out, n_in):
    bound = 1.0 / n_in**0.5
    return bf16((torch.rand(n_out, n_in, generator=generator, dtype=torch.float64) * 2 - 1) * bound)


def normal(*shape, std=1.0, mean=0.0):
    return bf16(torch.randn(*shape, generator=generator, dtype=torch.float64) * std + mean)


weights = {
    "x_embedder": linear(D, 68), "hdmap_embedder": linear(D, 64),
    "t_linear1": linear(D, D), "t_linear2": linear(3 * D, D), "t_norm": normal(D, std=0.1, mean=1.0),
    "final_mod_down": linear(LORA, D), "final_mod_up": linear(2 * D, LORA), "final_linear": linear(64, D),
    "text_proj": linear(TEXT_W, TEXT_IN), "text_proj_bias": normal(TEXT_W, std=0.02),
}
for b in range(BLOCKS if checkpoint is None else 0):
    weights.update({
        f"b{b}_self_q": linear(D, D), f"b{b}_self_k": linear(D, D), f"b{b}_self_v": linear(D, D),
        f"b{b}_self_out": linear(D, D),
        f"b{b}_self_q_norm": normal(HEAD, std=0.1, mean=1.0), f"b{b}_self_k_norm": normal(HEAD, std=0.1, mean=1.0),
        f"b{b}_cross_q": linear(D, D), f"b{b}_cross_k": linear(D, TEXT_W), f"b{b}_cross_v": linear(D, TEXT_W),
        f"b{b}_cross_out": linear(D, D),
        f"b{b}_cross_q_norm": normal(HEAD, std=0.1, mean=1.0), f"b{b}_cross_k_norm": normal(HEAD, std=0.1, mean=1.0),
        f"b{b}_mlp_up": linear(MLP, D), f"b{b}_mlp_down": linear(D, MLP),
    })
    for branch in range(3):
        weights[f"b{b}_mod_down{branch}"] = linear(LORA, D)
        weights[f"b{b}_mod_up{branch}"] = linear(3 * D, LORA)
if checkpoint is not None:
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    g = lambda name: sd[name].to(torch.float64)
    final = g("final_layer.linear.weight")
    weights = {
        "x_embedder": g("x_embedder.proj.1.weight")[:, :68],
        "hdmap_embedder": g("additional_patch_embedding.proj.1.weight"),
        "t_linear1": g("t_embedder.1.linear_1.weight"), "t_linear2": g("t_embedder.1.linear_2.weight"),
        "t_norm": g("t_embedding_norm.weight"),
        "final_mod_down": g("final_layer.adaln_modulation.1.weight"),
        "final_mod_up": g("final_layer.adaln_modulation.2.weight"),
        # network.py _fuse_shuffle_op_into_last_layer: (kt kh kw c) -> (c kt kh kw)
        "final_linear": final.view(4, 16, D).transpose(0, 1).reshape(64, D),
        "text_proj": g("crossattn_proj.0.weight"), "text_proj_bias": g("crossattn_proj.0.bias"),
    }
    names = {"self_q": "self_attn.q_proj", "self_k": "self_attn.k_proj", "self_v": "self_attn.v_proj",
             "self_out": "self_attn.output_proj", "self_q_norm": "self_attn.q_norm", "self_k_norm": "self_attn.k_norm",
             "cross_q": "cross_attn.q_proj", "cross_k": "cross_attn.k_proj", "cross_v": "cross_attn.v_proj",
             "cross_out": "cross_attn.output_proj", "cross_q_norm": "cross_attn.q_norm",
             "cross_k_norm": "cross_attn.k_norm", "mlp_up": "mlp.layer1", "mlp_down": "mlp.layer2"}
    for b in range(BLOCKS):
        for ours, theirs in names.items():
            weights[f"b{b}_{ours}"] = g(f"blocks.{b}.{theirs}.weight")
        for i, branch in enumerate(("self_attn", "cross_attn", "mlp")):
            weights[f"b{b}_mod_down{i}"] = g(f"blocks.{b}.adaln_modulation_{branch}.1.weight")
            weights[f"b{b}_mod_up{i}"] = g(f"blocks.{b}.adaln_modulation_{branch}.2.weight")
    del sd
text = normal(TEXT, TEXT_IN)
passes = []
for chunk in range(CHUNKS):
    for timestep in TIMESTEPS:
        passes.append((chunk, timestep, normal(L, 68), normal(L, 64)))


def compute_freqs(dim, ratio):
    dim_range = torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim
    theta = 10000.0 * ratio ** (dim / (dim - 2))
    return 1.0 / (theta**dim_range)


dim_hw = HEAD // 6 * 2
dim_t = HEAD - 2 * dim_hw
raw_t, raw_h, raw_w = compute_freqs(dim_t, 1.0), compute_freqs(dim_hw, 3.0), compute_freqs(dim_hw, 3.0)


def angles_for(chunk):
    # RotaryPositionEmbedding3D.shift_t
    freqs_t = torch.outer(torch.arange(T, dtype=torch.float32), raw_t) + (chunk * T) * raw_t
    freqs_h = torch.outer(torch.arange(H, dtype=torch.float32), raw_h)
    freqs_w = torch.outer(torch.arange(W, dtype=torch.float32), raw_w)
    return torch.cat([
        freqs_t[:, None, None, :].expand(T, H, W, -1),
        freqs_h[None, :, None, :].expand(T, H, W, -1),
        freqs_w[None, None, :, :].expand(T, H, W, -1)], dim=-1).reshape(L, HEAD // 2)


half = D // 2
timestep_freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)


def rope(t, angles, dtype):
    cos = torch.cos(angles).to(t.device).to(dtype)[:, None, :]
    sin = torch.sin(angles).to(t.device).to(dtype)[:, None, :]
    a, b = t[..., : HEAD // 2], t[..., HEAD // 2:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


def attention(q, k, v):
    if q.dtype == torch.float64:
        out = torch.empty_like(q)
        for h in range(HEADS):
            s = (q[:, h] @ k[:, h].T) / HEAD**0.5
            out[:, h] = torch.softmax(s, dim=-1) @ v[:, h]
        return out
    return F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                          v.transpose(0, 1)[None])[0].transpose(0, 1)


def run(dtype):
    device = "cpu" if checkpoint is not None and dtype == torch.float64 else "cuda"
    w = {name: value.to(device).to(dtype) for name, value in weights.items()}
    rms = lambda t, g: F.rms_norm(t, (t.shape[-1],), g, eps=1e-6)
    context = F.gelu(text.to(device).to(dtype) @ w["text_proj"].T + w["text_proj_bias"])
    text_kv = []
    caches = []
    for b in range(BLOCKS):
        k = rms((context @ w[f"b{b}_cross_k"].T).view(TEXT, HEADS, HEAD), w[f"b{b}_cross_k_norm"])
        text_kv.append((k, (context @ w[f"b{b}_cross_v"].T).view(TEXT, HEADS, HEAD)))
        caches.append(kvcache.BlockKVCache(k_shape=(WINDOW * L, HEADS, HEAD), v_shape=(WINDOW * L, HEADS, HEAD),
                                           seq_dim=0, chunk_size=L, window_size=WINDOW * L, sink_size=0,
                                           device=device, dtype=dtype))
    outputs = []
    for chunk, timestep, tokens, hdmap in passes:
        angles = angles_for(chunk)
        ts = torch.tensor(timestep * 0.001, dtype=torch.float64)
        arg = (ts * timestep_freqs.double()).to(device).to(dtype)
        sinusoid = torch.cat([torch.cos(arg), torch.sin(arg)])
        lora = F.silu(sinusoid @ w["t_linear1"].T) @ w["t_linear2"].T
        t_emb = rms(sinusoid, w["t_norm"])
        x = tokens.to(device).to(dtype) @ w["x_embedder"].T + hdmap.to(device).to(dtype) @ w["hdmap_embedder"].T
        for cache in caches:
            cache.before_update(chunk)
        for b in range(BLOCKS):
            mods = [(F.silu(t_emb) @ w[f"b{b}_mod_down{i}"].T @ w[f"b{b}_mod_up{i}"].T + lora).chunk(3)
                    for i in range(3)]

            def modulate(t, i):
                return F.layer_norm(t, (D,), eps=1e-6) * (1 + mods[i][1]) + mods[i][0]

            n = modulate(x, 0)
            q = rope(rms((n @ w[f"b{b}_self_q"].T).view(L, HEADS, HEAD), w[f"b{b}_self_q_norm"]), angles, dtype)
            k = rope(rms((n @ w[f"b{b}_self_k"].T).view(L, HEADS, HEAD), w[f"b{b}_self_k_norm"]), angles, dtype)
            v = (n @ w[f"b{b}_self_v"].T).view(L, HEADS, HEAD)
            caches[b].update(k, v)
            a = attention(q, caches[b].cached_k(), caches[b].cached_v())
            x = x + mods[0][2] * (a.reshape(L, D) @ w[f"b{b}_self_out"].T)
            n = modulate(x, 1)
            q = rms((n @ w[f"b{b}_cross_q"].T).view(L, HEADS, HEAD), w[f"b{b}_cross_q_norm"])
            x = x + mods[1][2] * (attention(q, *text_kv[b]).reshape(L, D) @ w[f"b{b}_cross_out"].T)
            n = modulate(x, 2)
            x = x + mods[2][2] * (F.gelu(n @ w[f"b{b}_mlp_up"].T) @ w[f"b{b}_mlp_down"].T)
        for cache in caches:
            cache.after_update(chunk)
        shift, scale = (F.silu(t_emb) @ w["final_mod_down"].T @ w["final_mod_up"].T + lora[: 2 * D]).chunk(2)
        out = (F.layer_norm(x, (D,), eps=1e-6) * (1 + scale) + shift) @ w["final_linear"].T
        outputs.append(out.double().cpu())
    return outputs


reference = run(torch.float64)
low = run(torch.bfloat16)
for i, ((chunk, timestep, _, _), r, o) in enumerate(zip(passes, reference, low)):
    print(f"pass {i} chunk {chunk} t {timestep:g}: torch bf16 rel L2 to float64 "
          f"{((o - r).norm() / r.norm()).item():.3e}")
    r.numpy().tofile(f"{folder}/reference_{i}.f64")
    o.float().numpy().tofile(f"{folder}/torch_bf16_{i}.f32")

for name, value in (weights.items() if checkpoint is None else ()):
    value.to(torch.bfloat16).view(torch.int16).numpy().tofile(f"{folder}/{name}.bf16")
text.to(torch.bfloat16).view(torch.int16).numpy().tofile(f"{folder}/text.bf16")
with open(f"{folder}/passes.txt", "w") as f:
    f.write(f"{T} {H} {W} {BLOCKS} {WINDOW} {len(passes)}\n")
    for i, (chunk, timestep, tokens, hdmap) in enumerate(passes):
        f.write(f"{chunk} {timestep:g}\n")
        tokens.to(torch.bfloat16).view(torch.int16).numpy().tofile(f"{folder}/tokens_{i}.bf16")
        hdmap.to(torch.bfloat16).view(torch.int16).numpy().tofile(f"{folder}/hdmap_{i}.bf16")
