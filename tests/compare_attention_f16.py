"""Compares cuda_attention_f16's output with PyTorch's scaled_dot_product_attention backends on
the same inputs. Usage: compare_attention_f16.py <dir> n_q n_kv heads"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

folder, n_q, n_kv, heads = sys.argv[1], *map(int, sys.argv[2:5])
for name, dtype in (("f16", torch.float16), ("bf16", torch.bfloat16)):
    def load(part, rows):
        raw = np.fromfile(f"{folder}/{name}_{part}.bin", dtype=np.uint16).astype(np.int16)
        return torch.from_numpy(raw).view(dtype).view(rows, heads, 128).transpose(0, 1)[None].cuda()
    q, k, v, ours = load("q", n_q), load("k", n_kv), load("v", n_kv), load("out", n_q)
    reference = F.scaled_dot_product_attention(q.double(), k.double(), v.double())
    for backend in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION"):
        with sdpa_kernel(getattr(SDPBackend, backend)):
            theirs = F.scaled_dot_product_attention(q, k, v)
        same = (theirs.view(torch.int16) == ours.view(torch.int16)).float().mean().item()
        error = lambda x: ((x.double() - reference).norm() / reference.norm()).item()
        print(f"{name} {backend}: identical {same * 100:.2f}%, rel L2 to fp64: theirs {error(theirs):.3e}, "
              f"ours {error(ours):.3e}")
