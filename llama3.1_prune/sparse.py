import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ——— 1. Load model & extract FFN weight ———
device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(
    "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B",
    torch_dtype=torch.float16,
    device_map="cpu"
)
# —— 2. Define a SparseLinear that applies a sparse weight matrix ——
class SparseLinear(nn.Module):
    def __init__(self, sparse_weight: torch.sparse_coo_tensor, bias: torch.Tensor = None):
        super().__init__()
        self.weight = sparse_weight.coalesce()       # [out, in]
        self.bias = bias
    def forward(self, x: torch.Tensor):
        # x: [..., in]
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]).T        # [in, N]
        # sparse.mm(weight, x_flat) -> [out, N]
        y_flat_t = torch.sparse.mm(self.weight, x_flat)  # [out, N]
        y_flat   = y_flat_t.T                           # [N, out]
        y = y_flat.view(*orig_shape[:-1], self.weight.shape[0])
        if self.bias is not None:
            y = y + self.bias
        return y

# —— 3. Replace each MLP linear with its sparsified version ——
th = 0.005
for layer in model.model.layers:
    mlp = layer.mlp
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        lin: nn.Linear = getattr(mlp, proj_name)
        W = lin.weight.data.cpu()
        # mask for |w| ≥ th
        mask = W.abs() >= th
        rows, cols = mask.nonzero(as_tuple=True)    # each is a 1D LongTensor of length K
        vals = W[rows, cols]                        # 1D floatTensor of length K
        indices = torch.stack([rows, cols], dim=0)  # shape [2, K]
        # Build the sparse tensor directly on the target device
        W_sparse = torch.sparse_coo_tensor(
            indices,            # 2×K
            vals,               # K
            size=W.shape,       # (out, in)
            device=device
        ).coalesce()

        b = lin.bias.data.to(device) if lin.bias is not None else None
        # build and swap in sparse linear
        setattr(mlp, proj_name, SparseLinear(W_sparse, b))

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

# Sparse MLP avg forward: 1427.68 ms
