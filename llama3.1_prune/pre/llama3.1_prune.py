import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ——— 1. Load model & extract FFN weight ———
model = AutoModelForCausalLM.from_pretrained(
    "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B",
    torch_dtype=torch.float16,
    device_map="cpu"
)
wi_weight = model.model.layers[0].mlp.gate_proj.weight.data  # [d_ffn, d_model]

# ——— 2. Magnitude‐based statistics ———
threshold_mag = 0.005
# mask of all absolute‐value< threshold
mask_small = wi_weight.abs() < threshold_mag

num_small      = int(mask_small.sum())             # count of tiny weights
total_weights  = wi_weight.numel()                 # total number of weights
perc_small     = num_small / total_weights * 100   # percentage

print(f"Total weights in FFN layer: {total_weights}")
print(f"Weights with |w| < {threshold_mag}: {num_small} ({perc_small:.2f}%)")

# if you really want the raw values:
# small_values = wi_weight[mask_small]
# print("Example tiny weights:", small_values[:10])

# ——— 3. Prepare normalized neuron vectors ———
# each row of W is one neuron's input‐weight vector# assuming wi_weight is already loaded as a torch.Tensor

# 1. Print the full weight matrix (can be very large!)
print("wi_weight:", wi_weight)

# 2. Compute max and min
max_val = wi_weight.max().item()
min_val = wi_weight.min().item()

# 3. Print them
print(f"Max weight value: {max_val}")
print(f"Min weight value: {min_val}")

# wi_weight: [d_ffn, d_model]
W = F.normalize(wi_weight, p=2, dim=1)  
# Now for every i:  W[i].norm(p=2) == 1

norms = W.norm(p=2, dim=1)   # shape: [d_ffn]
print(norms.mean().item(), norms.min().item(), norms.max().item())
# should all be 1 (or extremely close, given floating-point)



print(W)

# ——— 4. Single‐neuron similarity lookup ———
neuron_idx       = 5       # change to whichever neuron you’re curious about
similarity_thr   = 0.9     # cosine‐threshold for “similar” neurons

v = W[neuron_idx]                  # [d_model]
dots = (W @ v)                     # [d_ffn]   – cosine between neuron_idx and all

print(dots)

# mask out itself
dots[neuron_idx] = -1.0            

# find all indices above threshold
sim_idxs = torch.nonzero(dots > similarity_thr, as_tuple=True)[0].tolist()
sim_vals = dots[sim_idxs].tolist()


print(f"\nNeuron {neuron_idx} has {len(sim_idxs)} partners with cosine > {similarity_thr}:")
for idx, val in zip(sim_idxs, sim_vals):
    print(f"  • neuron {idx} → cosine = {val:.3f}")
