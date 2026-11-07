"""Text embeddings for OmniDreams from a prompt, on the CPU, as FlashDreams computes them.

flashdreams/infra/encoder/text/cosmos_reason1.py CosmosReason1TextEncoder._encode: nvidia/
Cosmos-Reason1-7B at revision 3210bec0 (the checkpoint Cosmos-Predict 2.5 2B was trained on), the
system prompt "You are a helpful assistant who will provide prompts to an image generator.", the
chat template without a generation prompt, padded with the pad token to 512 tokens, all 28 hidden
states (the embedding layer skipped) each normalized to mean 0 and std 1 per token and concatenated:
[1, 1, 512, 100352] bf16, the shape the Kaggle driver saved.
Needs transformers >= 4.49 (Qwen2.5-VL). Usage: text_embed.py "<prompt>" <out.pt> [threads] [--float32]
--float32 runs the model in float32 and saves float32 (a reference for other implementations).
"""
import sys
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "nvidia/Cosmos-Reason1-7B"
REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
float32 = "--float32" in sys.argv
args = [a for a in sys.argv[1:] if a != "--float32"]
prompt, out = args[0], args[1]
torch.set_num_threads(int(args[2]) if len(args) > 2 else 16)
dtype = torch.float32 if float32 else torch.bfloat16

started = time.time()
processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
tokenizer = processor.tokenizer
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                                           dtype=dtype)
model.eval().requires_grad_(False)
print(f"loaded in {time.time() - started:.1f} s", flush=True)

conversations = [
    {"role": "system", "content": [{"type": "text",
                                    "text": "You are a helpful assistant who will provide prompts to an image generator."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]
formatted = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=False,
                                          add_vision_id=False, return_tensors="pt")
if not torch.is_tensor(formatted):
    formatted = formatted["input_ids"]
if formatted.ndim == 1:
    formatted = formatted.unsqueeze(0)
tokens = formatted.shape[1]
if tokens < 512:
    formatted = torch.cat([formatted, torch.full((1, 512 - tokens), tokenizer.pad_token_id or 0,
                                                 dtype=formatted.dtype)], dim=1)
else:
    formatted = formatted[:, :512]
print(f"{tokens} prompt tokens", flush=True)

started = time.time()
with torch.no_grad():
    hidden = model(input_ids=formatted, output_hidden_states=True, return_dict=True).hidden_states
normalized = [(h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8) for h in hidden[1:]]
embeddings = torch.cat(normalized, dim=-1).unsqueeze(0)
print(f"encoded in {time.time() - started:.1f} s: {tuple(embeddings.shape)} {embeddings.dtype}", flush=True)
torch.save(embeddings.cpu(), out)
