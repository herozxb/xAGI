import time
import torch
from transformers import AutoModelForCausalLM

device = torch.device("cpu")

# 1) Load in float32 on CPU
model = AutoModelForCausalLM.from_pretrained(
    "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B",
    torch_dtype=torch.float32,
    device_map=None
).to(device)

# 2) Grab the very first layer's MLP
dense_mlp = model.model.layers[0].mlp

# 3) Prepare dummy input
batch_size, seq_len, hidden = 1, 8, model.config.hidden_size
x = torch.randn(batch_size, seq_len, hidden, device=device)

# 4) Timing helper
def time_layer(fwd_fn, x, repeats=10):
    # warmup
    for _ in range(3):
        _ = fwd_fn(x)
    # timed runs
    start = time.time()
    for _ in range(repeats):
        _ = fwd_fn(x)
    return (time.time() - start) / repeats

# 5) Measure
t_dense = time_layer(lambda y: dense_mlp(y), x)
print(f"Dense MLP avg forward: {t_dense * 1000:.2f} ms")

# Dense MLP avg forward: 30.89 ms