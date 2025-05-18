#!/usr/bin/env python3
"""
Pipeline: weight-level threshold prune (<0.005), prune neurons based on average absolute output weights,
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
OUTPUT_AVG_THRESH   = 0.007   # prune neuron if avg abs W_down < this
MAX_NEW_TOKENS      = 5       # generation length
PROMPT              = "I see a fox, whittle you down, on you, how to fight the fox"

# -------------------- Prune & Restructure --------------------
def prune_and_restructure(model: nn.Module,
                          weight_thresh: float,
                          output_avg_thresh: float):
    """
    1) Zero out weights below weight_thresh in gate/up/down projections.
    2) For each neuron j:
         - compute avg_abs = W_down.abs().sum(dim=0)/d_model
         - keep if avg_abs[j] >= output_avg_thresh AND W_gate row j has any non-zero
    3) Rebuild each MLP's gate_proj, up_proj, down_proj as smaller nn.Linear modules.
    4) Print statistics of zeroed weights and neurons kept.
    """
    d_model   = model.config.hidden_size
    old_d_ffn = model.model.layers[0].mlp.gate_proj.weight.size(0)

    for idx, layer in enumerate(model.model.layers):
        mlp    = layer.mlp
        # clone weights & biases
        W_gate = mlp.gate_proj.weight.data.clone()
        b_gate = mlp.gate_proj.bias.data.clone() if mlp.gate_proj.bias is not None else None
        W_up   = mlp.up_proj.weight.data.clone()
        b_up   = mlp.up_proj.bias.data.clone()   if mlp.up_proj.bias is not None else None
        W_down = mlp.down_proj.weight.data.clone()
        b_down = mlp.down_proj.bias.data.clone() if mlp.down_proj.bias is not None else None

        # 1) threshold all small weights
        W_gate[W_gate.abs() < weight_thresh]   = 0.0
        W_up[W_up.abs()     < weight_thresh]   = 0.0
        W_down[W_down.abs() < weight_thresh]   = 0.0

        # stats for gate weights
        total_gate = W_gate.numel()
        zero_gate  = int((W_gate == 0.0).sum().item())
        print(f"Layer {idx:2d} W_gate zeros: {zero_gate}/{total_gate} ({zero_gate/total_gate*100:.2f}%)")

        # 2) compute output-average per neuron (abs sum over down-proj)
        avg_abs = W_down.abs().sum(dim=0) / d_model    # [old_d_ffn]
        print(avg_abs)

        # print min/max of avg_abs
        min_avg = avg_abs.min().item()
        max_avg = avg_abs.max().item()
        print(f"Layer {idx:2d} avg_abs of W_down: min={min_avg:.6f}, max={max_avg:.6f}")

        # masks: avg threshold AND gate non-zero
        mask_out = avg_abs >= output_avg_thresh
        mask_in  = W_gate.abs().sum(dim=1) > 0
        keep_mask = mask_out & mask_in
        keep_idx  = torch.nonzero(keep_mask, as_tuple=True)[0]
        new_d_ffn = keep_idx.numel()
        if new_d_ffn == 0:
            raise ValueError(f"All neurons pruned in layer {idx} (avg_thresh={output_avg_thresh})")
        print(f"Layer {idx:2d}: kept {new_d_ffn}/{old_d_ffn} neurons (avg_out ≥ {output_avg_thresh})  ({new_d_ffn/old_d_ffn*100:.2f}%)")

        # 3) rebuild smaller Linears in fp16
        new_gate = nn.Linear(d_model, new_d_ffn, bias=(b_gate is not None),
                             device=DEVICE, dtype=torch.float16)
        new_up   = nn.Linear(d_model, new_d_ffn, bias=(b_up   is not None),
                             device=DEVICE, dtype=torch.float16)
        new_down = nn.Linear(new_d_ffn, d_model,   bias=(b_down is not None),
                             device=DEVICE, dtype=torch.float16)

        # 4) copy thresholded weights/biases for kept neurons
        new_gate.weight.data.copy_(W_gate[keep_idx, :])
        if b_gate is not None:
            new_gate.bias.data.copy_(b_gate[keep_idx])

        new_up.weight.data.copy_(W_up[keep_idx, :])
        if b_up is not None:
            new_up.bias.data.copy_(b_up[keep_idx])

        new_down.weight.data.copy_(W_down[:, keep_idx])
        if b_down is not None:
            new_down.bias.data.copy_(b_down)

        # swap into model
        mlp.gate_proj = new_gate
        mlp.up_proj   = new_up
        mlp.down_proj = new_down

# -------------------- Generation --------------------
def generate_text(model: nn.Module, prompt: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    inputs    = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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
    # load & eval
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    # prune & restructure
    prune_and_restructure(model, WEIGHT_THRESH, OUTPUT_AVG_THRESH)

    # generate
    start = time.time()
    output = generate_text(model, PROMPT, MAX_NEW_TOKENS)
    print(f"\n--- Generation Output (time: {time.time()-start:.3f}s) ---\n", output)
