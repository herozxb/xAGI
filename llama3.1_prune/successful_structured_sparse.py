import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import time

# ——— CONFIG —————————————————————————————————————————————
model_path   = "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"  # your local checkpoint
device       = torch.device("cpu")
prune_ratio  = 0.5                      # keep 50% of neurons
# ————————————————————————————————————————————————————————————

# 1) load the original model in fp32 on CPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=None
).to(device)

d_model = model.config.hidden_size
# derive old FFN dim from the first layer
old_d_ffn = model.model.layers[0].mlp.gate_proj.weight.size(0)
new_d_ffn = int(old_d_ffn * prune_ratio)
print(f"Pruning each FFN from {old_d_ffn} → {new_d_ffn} neurons")

# 2) structured prune every layer
for layer in model.model.layers:
    mlp      = layer.mlp
    # fetch originals
    W_gate   = mlp.gate_proj.weight.data       # [old_d_ffn, d_model]
    W_up     = mlp.up_proj.weight.data         # [old_d_ffn, d_model]
    W_down   = mlp.down_proj.weight.data       # [d_model, old_d_ffn]
    b_gate   = mlp.gate_proj.bias.data         if mlp.gate_proj.bias   is not None else None
    b_up     = mlp.up_proj.bias.data           if mlp.up_proj.bias     is not None else None
    b_down   = mlp.down_proj.bias.data         if mlp.down_proj.bias   is not None else None

    # compute neuron importances (L2‐norm of each input row)
    norms = W_gate.norm(p=2, dim=1)            # [old_d_ffn]
    # select top-k neurons
    topk  = torch.topk(norms, new_d_ffn, largest=True).indices
    topk, _ = torch.sort(topk)                 # optional: maintain original order

    # build new, smaller Linear layers
    #new_gate = nn.Linear(d_model, new_d_ffn, bias=(b_gate is not None))
    #new_up   = nn.Linear(d_model, new_d_ffn, bias=(b_up   is not None))
    #new_down = nn.Linear(new_d_ffn, d_model, bias=(b_down is not None))

    new_gate = nn.Linear(d_model, new_d_ffn, bias=(b_gate is not None)).to(device).half()
    new_up   = nn.Linear(d_model, new_d_ffn, bias=(b_up   is not None)).to(device).half()
    new_down = nn.Linear(new_d_ffn, d_model, bias=(b_down is not None)).to(device).half()


    # copy over pruned weights & biases
    new_gate.weight.data.copy_(W_gate[topk, :])
    if b_gate is not None:
        new_gate.bias.data.copy_(b_gate[topk])
    new_up.weight.data.copy_(W_up[topk, :])
    if b_up is not None:
        new_up.bias.data.copy_(b_up[topk])
    # down_proj: we keep only those columns
    new_down.weight.data.copy_(W_down[:, topk])
    if b_down is not None:
        new_down.bias.data.copy_(b_down)

    # swap them into the model
    mlp.gate_proj = new_gate.to(device)
    mlp.up_proj   = new_up.to(device)
    mlp.down_proj = new_down.to(device)

# 3) Save your structured-pruned model
out_dir = "./llama3.1-structured-pruned"
#model.save_pretrained(out_dir)
print(f"Saved pruned model to {out_dir}")

# —— 4. Quick speed test: compare dense vs sparse MLP ——  
#    we’ll run one forward on a dummy token batch through the first layer’s MLP

# prepare dummy input
batch_size, seq_len, hidden = 1, 8, model.config.hidden_size
# prepare dummy input in half-precision to match the model
x = torch.randn(batch_size, seq_len, hidden, device=device, dtype=torch.float16)

# helper to time a single forward
def time_layer(fwd_fn, x, repeats=10):
    # warmup
    for _ in range(3):
        _ = fwd_fn(x)
    # timed runs
    start = time.time()
    for _ in range(repeats):
        _ = fwd_fn(x)
    return (time.time() - start) / repeats

sparse_mlp = model.model.layers[0].mlp

#t_dense  = time_layer(lambda y: dense_mlp(y), x)
t_sparse = time_layer(lambda y: sparse_mlp(y), x)

#print(f"Dense MLP avg forward:  {t_dense*1000:.2f} ms")
print(f"Sparse MLP avg forward: {t_sparse*1000:.2f} ms")

# Sparse MLP avg forward: 6.25 ms




