#!/usr/bin/env python3
"""
Pipeline: weight-level threshold prune (<0.005), prune neurons based on L1 norm,
restructure MLP to compact matrices, then generate text.
"""

import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Configuration --------------------
MODEL_PATH         = "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"
DEVICE              = torch.device("cpu")
WEIGHT_THRESH       = 0.005   # zero out weights with abs < this in all projections
PRUNE_L1_THRESHOLD  = 30.0    # prune neurons based on L1 norm threshold
MAX_NEW_TOKENS      = 5       # generation length
PROMPT              = "I see a fox, whittle you down, on you, how to fight the fox"

# -------------------- Prune & Restructure --------------------
def prune_and_restructure(model: nn.Module,
                          weight_thresh: float,
                          prune_l1_threshold: float):
    """
    1) Zero out weights below weight_thresh in gate/up/down projections.
    2) For each neuron j, compute the L1 norm of W_gate and prune neurons based on L1 norm.
    3) Rebuild each MLP's gate_proj, up_proj, down_proj as smaller nn.Linear modules.
    4) Print statistics of zeroed weights, neurons kept, and pruned neurons.
    """
    d_model   = model.config.hidden_size
    old_d_ffn = model.model.layers[0].mlp.gate_proj.weight.size(0)

    for idx, layer in enumerate(model.model.layers):
        mlp    = layer.mlp
        # clone weights & biases
        Wg = mlp.gate_proj.weight.data.clone()
        bg = mlp.gate_proj.bias.data.clone() if mlp.gate_proj.bias is not None else None
        Wu = mlp.up_proj.weight.data.clone()
        bu = mlp.up_proj.bias.data.clone()   if mlp.up_proj.bias is not None else None
        Wd = mlp.down_proj.weight.data.clone()
        bd = mlp.down_proj.bias.data.clone() if mlp.down_proj.bias is not None else None

        # 1) threshold small weights to zero
        Wg[Wg.abs() < weight_thresh] = 0.0
        Wu[Wu.abs() < weight_thresh] = 0.0
        Wd[Wd.abs() < weight_thresh] = 0.0

        # report zeroed in gate
        total_g = Wg.numel()
        zero_g  = int((Wg == 0).sum().item())
        print(f"Layer {idx:2d} gate weights zeroed: {zero_g}/{total_g} ({zero_g/total_g*100:.2f}%)")

        # 2) compute per-neuron importance via L1 norm
        norms = Wg.abs().sum(dim=1)  # L1 norm of each neuron
        print(f"Layer {idx:2d} Neuron L1-norms: min = {norms.min().item():.6f}, max = {norms.max().item():.6f}")

        # 3) pick neurons above threshold based on L1 norm
        keep_idx = torch.nonzero(norms >= prune_l1_threshold, as_tuple=True)[0]
        new_d_ffn = keep_idx.numel()
        if new_d_ffn == 0:
            raise ValueError(f"Threshold {prune_l1_threshold} pruned ALL neurons in layer {idx}!")
        print(f"Layer {idx:2d}: keeping {new_d_ffn}/{old_d_ffn} neurons (L1 ≥ {prune_l1_threshold}) ({new_d_ffn/old_d_ffn*100:.2f}%)")

        # 4) rebuild smaller Linear layers in float16
        new_gate = nn.Linear(d_model,    new_d_ffn, bias=(bg is not None),
                             device=DEVICE, dtype=torch.float16)
        new_up   = nn.Linear(d_model,    new_d_ffn, bias=(bu is not None),
                             device=DEVICE, dtype=torch.float16)
        new_down = nn.Linear(new_d_ffn,  d_model,   bias=(bd is not None),
                             device=DEVICE, dtype=torch.float16)

        # 5) copy over only the kept rows/cols
        new_gate.weight.data.copy_(Wg[keep_idx, :])
        if bg is not None:
            new_gate.bias.data.copy_(bg[keep_idx])

        new_up.weight.data.copy_(Wu[keep_idx, :])
        if bu is not None:
            new_up.bias.data.copy_(bu[keep_idx])

        new_down.weight.data.copy_(Wd[:, keep_idx])
        if bd is not None:
            new_down.bias.data.copy_(bd)

        # 6) swap them in
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

    prune_and_restructure(model, WEIGHT_THRESH, PRUNE_L1_THRESHOLD)

    start = time.time()
    out = generate_text(model, PROMPT, MAX_NEW_TOKENS)
    print(f"\n--- Generation Output (time: {time.time()-start:.3f}s) ---\n", out)
