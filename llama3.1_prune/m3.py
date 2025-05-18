#!/usr/bin/env python3
"""
Pipeline: weight-level threshold prune (<0.005), restructure MLP to best-fit mini matrices, then generate text.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

# -------------------- Configuration --------------------
MODEL_PATH    = "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"
DEVICE         = torch.device("cpu")
WEIGHT_THRESH  = 0.005      # zero out weights with abs < this
MAX_NEW_TOKENS = 5          # generation length
PROMPT         = (
    "I see a fox, whittle you down, on you, how to fight the fox"
)

# -------------------- Prune & Restructure --------------------
def prune_and_restructure(model: nn.Module, weight_thresh: float):
    """
    Zero out small weights, then remove any neuron (row) whose entire gate-proj row is zero.
    Rebuild each MLP (gate_proj, up_proj, down_proj) as smaller Linear modules.
    """
    d_model     = model.config.hidden_size
    old_d_ffn   = model.model.layers[0].mlp.gate_proj.weight.size(0)

    for idx, layer in enumerate(model.model.layers):
        mlp    = layer.mlp
        # clone weights & biases
        W_gate = mlp.gate_proj.weight.data.clone()
        b_gate = mlp.gate_proj.bias.data.clone() if mlp.gate_proj.bias is not None else None
        W_up   = mlp.up_proj.weight.data.clone()
        b_up   = mlp.up_proj.bias.data.clone()   if mlp.up_proj.bias   is not None else None
        W_down = mlp.down_proj.weight.data.clone()
        b_down = mlp.down_proj.bias.data.clone() if mlp.down_proj.bias is not None else None


        #print(W_gate.abs())
        #print(W_up.abs())
        #print(W_down.abs())
        # 1) threshold weights
        W_gate[W_gate.abs() < weight_thresh]   = 0.0
        W_up[W_up.abs()     < weight_thresh]   = 0.0
        W_down[W_down.abs() < weight_thresh]   = 0.0

        print("==================")
        print(W_gate.abs())
        print(W_up.abs())
        print(W_down.abs())

        # Print zero-value statistics for W_gate
        total_params = W_gate.numel()
        zero_params  = int((W_gate == 0.0).sum().item())
        print(f"Layer {idx:2d} W_gate zeroed parameters: {zero_params}/{total_params} "
              f"({zero_params/total_params*100:.2f}%)")

        # 2) find neurons with any non-zero gate weights
        keep_idx = torch.nonzero(W_gate.abs().sum(dim=1) > 0, as_tuple=True)[0]
        new_d_ffn = keep_idx.numel()
        if new_d_ffn == 0:
            raise ValueError(f"All neurons pruned in layer {idx} with threshold {weight_thresh}")
        print(f"Layer {idx:2d}: kept {new_d_ffn}/{old_d_ffn} neurons after weight-thresholding")

        # 3) rebuild smaller Linear layers in fp16
        new_gate = nn.Linear(d_model,  new_d_ffn, bias=(b_gate is not None), device=DEVICE, dtype=torch.float16)
        new_up   = nn.Linear(d_model,  new_d_ffn, bias=(b_up   is not None), device=DEVICE, dtype=torch.float16)
        new_down = nn.Linear(new_d_ffn, d_model,   bias=(b_down is not None),device=DEVICE, dtype=torch.float16)

        # 4) copy thresholded weights & biases for kept neurons
        new_gate.weight.data.copy_(W_gate[keep_idx, :])
        if b_gate is not None:
            new_gate.bias.data.copy_(b_gate[keep_idx])

        new_up.weight.data.copy_(W_up[keep_idx, :])
        if b_up is not None:
            new_up.bias.data.copy_(b_up[keep_idx])

        new_down.weight.data.copy_(W_down[:, keep_idx])
        if b_down is not None:
            new_down.bias.data.copy_(b_down)

        # 5) swap into model
        mlp.gate_proj = new_gate
        mlp.up_proj   = new_up
        mlp.down_proj = new_down

# -------------------- Generation --------------------
def generate_text(model: nn.Module, prompt: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
    print("=====llama3.1 running time is :======")    
    print(time.time() - start)  

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
    prune_and_restructure(model, WEIGHT_THRESH)

    # generate
    output = generate_text(model, PROMPT, MAX_NEW_TOKENS)
    print("\n--- Generation Output ---\n", output)
