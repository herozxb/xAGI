import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

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
threshold = 30.0


for idx, layer in enumerate(model.model.layers):
    mlp    = layer.mlp
    W_gate = mlp.gate_proj.weight.data    # [old_d_ffn, d_model]
    W_up   = mlp.up_proj.weight.data      # [old_d_ffn, d_model]
    W_down = mlp.down_proj.weight.data    # [d_model, old_d_ffn]
    b_gate = mlp.gate_proj.bias.data      if mlp.gate_proj.bias   is not None else None
    b_up   = mlp.up_proj.bias.data        if mlp.up_proj.bias     is not None else None
    b_down = mlp.down_proj.bias.data      if mlp.down_proj.bias   is not None else None

    # 1) compute per‐neuron importance via L1
    norms = W_gate.abs().sum(dim=1)       # [old_d_ffn]
    print(norms)

    # get min/max
    min_norm = norms.min().item()
    max_norm = norms.max().item()

    print(f"Neuron L1‐norms: min = {min_norm:.6f}, max = {max_norm:.6f}")

    # 2) pick neurons above threshold
    threshold = threshold + idx * 0.039
    keep_idx = torch.nonzero(norms >= threshold  , as_tuple=True)[0]
    new_d_ffn = keep_idx.numel()
    if new_d_ffn == 0:
        raise ValueError(f"Threshold {threshold} pruned ALL neurons in layer {idx}!")
    print(f"Layer {idx:2d}: keeping {new_d_ffn}/{old_d_ffn} neurons (L1 ≥ {threshold}) ({new_d_ffn/old_d_ffn*100:.2f}%)")

    # 3) rebuild pruned Linear layers in float16
    new_gate = nn.Linear(d_model,    new_d_ffn, bias=(b_gate is not None),
                         device=device, dtype=torch.float16)
    new_up   = nn.Linear(d_model,    new_d_ffn, bias=(b_up   is not None),
                         device=device, dtype=torch.float16)
    new_down = nn.Linear(new_d_ffn,  d_model,   bias=(b_down is not None),
                         device=device, dtype=torch.float16)

    # 4) copy over only the kept rows/cols
    new_gate.weight.data.copy_(W_gate[keep_idx, :])
    if b_gate is not None:
        new_gate.bias.data.copy_(b_gate[keep_idx])

    new_up.weight.data.copy_(W_up[keep_idx, :])
    if b_up is not None:
        new_up.bias.data.copy_(b_up[keep_idx])

    new_down.weight.data.copy_(W_down[:, keep_idx])
    if b_down is not None:
        new_down.bias.data.copy_(b_down)

    # 5) swap them in
    mlp.gate_proj = new_gate
    mlp.up_proj   = new_up
    mlp.down_proj = new_down

'''
# 1) Zero out all tiny weights in every FFN layer
threshold_w = 0.1
for layer in model.model.layers:
    for proj in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
        w = proj.weight.data
        w[ w.abs() < threshold_w ] = 0.0

'''


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
# Sparse MLP avg forward: 23.80 ms



# 2) Rerun your timing function on the first layer’s MLP
#    (make sure x, time_layer, and sparse_mlp are defined as before)
t_thresh = time_layer(lambda y: model.model.layers[0].mlp(y), x)
print(f"Thresholded MLP avg forward: {t_thresh*1000:.2f} ms")
# Thresholded MLP avg forward: 23.02 ms

# Sparse MLP avg forward: 6.25 ms

# 1) Reload tokenizer (same as before)
tokenizer = AutoTokenizer.from_pretrained(
    "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B",
    use_fast=False
)

# 2) Put model in eval mode
model.eval()

# 3) Prepare your prompt
prompt = "write a python to add two number "
inputs = tokenizer(prompt, return_tensors="pt").to(device)

start = time.time()
# 4) Generate
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=True,
        top_p=0.95,
        temperature=0.8
    )
print("=====llama3.1 running time is :======")    
print(time.time() - start)
# I see a fox, whittle you down, on you, how to fight the fox?
# I can't remember
# 222.12610292434692

# 5) Decode and print
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))




