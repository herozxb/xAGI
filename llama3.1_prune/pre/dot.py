#!/usr/bin/env python3
"""
Pipeline: weight-level threshold prune (<0.005), prune redundant neurons by weight-vector angle,
restructure MLP to compact matrices, then generate text.
"""

import time
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Configuration --------------------
MODEL_PATH         = "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"
DEVICE              = torch.device("cpu")
WEIGHT_THRESH       = 0.005   # zero out weights with abs < this in all projections
ANGLE_THRESH_DEG    = 75.0     # prune neurons whose weight vectors differ by < this angle
MAX_NEW_TOKENS      = 5      # generation length
PROMPT              = "I see a fox, whittle you down, on you, how to fight the fox"

# -------------------- Prune & Restructure --------------------
def prune_and_restructure(model: nn.Module,
                          weight_thresh: float,
                          angle_thresh_deg: float):
    """
    1) Zero out weights below weight_thresh in gate/up/down projections.
    2) Identify sets of neurons with nearly parallel gate weight vectors (angle < angle_thresh).
       Keep only one per set.
    3) Rebuild each MLP's gate_proj, up_proj, down_proj as smaller nn.Linear modules.
    4) Print statistics of zeroed weights, angles, and neurons kept.
    """
    d_model   = model.config.hidden_size
    old_d_ffn = model.model.layers[0].mlp.gate_proj.weight.size(0)

    # precompute cosine threshold
    cos_thresh = math.cos(math.radians(angle_thresh_deg))

    for idx, layer in enumerate(model.model.layers):
        mlp    = layer.mlp
        # clone weights & biases
        Wg = mlp.gate_proj.weight.data.clone()
        bg = mlp.gate_proj.bias.data.clone() if mlp.gate_proj.bias is not None else None
        Wu = mlp.up_proj.weight.data.clone()
        bu = mlp.up_proj.bias.data.clone()   if mlp.up_proj.bias   is not None else None
        Wd = mlp.down_proj.weight.data.clone()
        bd = mlp.down_proj.bias.data.clone() if mlp.down_proj.bias is not None else None

        # 1) threshold all small weights
        Wg[Wg.abs() < weight_thresh] = 0.0
        Wu[Wu.abs() < weight_thresh] = 0.0
        Wd[Wd.abs() < weight_thresh] = 0.0

        # report zeroed in gate
        total_g = Wg.numel()
        zero_g  = int((Wg == 0).sum().item())
        print(f"Layer {idx:2d} gate weights zeroed: {zero_g}/{total_g} ({zero_g/total_g*100:.2f}%)")

        # 2) group by angle of gate weight vectors
        # normalize each row
        norms = Wg.norm(dim=1, keepdim=True)
        # avoid division by zero
        norms[norms == 0] = 1.0
        Wg_norm = Wg / norms

        removed = torch.zeros(old_d_ffn, dtype=torch.bool)
        keep_list = []
        for i in range(old_d_ffn):
            if removed[i]:
                continue
            keep_list.append(i)
            # compute cosine similarity with all
            cos_sim = (Wg_norm @ Wg_norm[i])  # [old_d_ffn]
            #angles = torch.acos(cos_sim.clamp(-1+1e-6,1-1e-6)) * (180.0 / math.pi)
            #print(f"Layer {idx}, neuron {i}: cos_sim[:5] = {cos_sim[:5].tolist()}, angles[:5] = {angles[:5].tolist()}")

            # find j > i with cos_sim >= cos_thresh
            mask = (cos_sim >= cos_thresh) & (torch.arange(old_d_ffn) > i)
            removed |= mask
        keep_idx = torch.tensor(keep_list, dtype=torch.long)
        new_d = keep_idx.numel()
        pruned = old_d_ffn - new_d

        print(f"Layer {idx:2d}: pruned {pruned}/{old_d_ffn} redundant neurons (angle < {angle_thresh_deg}°) ({pruned/old_d_ffn*100:.2f}%)")

        # 3) rebuild smaller Linears in fp16
        new_gate = nn.Linear(d_model, new_d, bias=(bg is not None), device=DEVICE, dtype=torch.float16)
        new_up   = nn.Linear(d_model, new_d, bias=(bu is not None), device=DEVICE, dtype=torch.float16)
        new_down = nn.Linear(new_d,   d_model, bias=(bd is not None), device=DEVICE, dtype=torch.float16)

        # 4) copy weights/biases for kept neurons
        new_gate.weight.data.copy_(Wg[keep_idx, :])
        if bg is not None:
            new_gate.bias.data.copy_(bg[keep_idx])
        new_up.weight.data.copy_(Wu[keep_idx, :])
        if bu is not None:
            new_up.bias.data.copy_(bu[keep_idx])
        new_down.weight.data.copy_(Wd[:, keep_idx])
        if bd is not None:
            new_down.bias.data.copy_(bd)

        # swap into model
        mlp.gate_proj = new_gate
        mlp.up_proj   = new_up
        mlp.down_proj = new_down

# -------------------- Generation --------------------
def generate_text(model: nn.Module, prompt: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

# -------------------- Main Script --------------------
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    prune_and_restructure(model, WEIGHT_THRESH, ANGLE_THRESH_DEG)

    start = time.time()
    out = generate_text(model, PROMPT, MAX_NEW_TOKENS)
    print(f"\n--- Generation Output (time: {time.time()-start:.3f}s) ---\n", out)
