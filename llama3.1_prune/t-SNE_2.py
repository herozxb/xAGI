import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM

import time


import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# ——— CONFIG —————————————————————————————————————————————
model_path   = "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"  # your local checkpoint
device       = torch.device("cpu")
# ————————————————————————————————————————————————————————————


# allowlist the LlamaForCausalLM class so torch.load can unpickle it
torch.serialization.add_safe_globals([LlamaForCausalLM])

# load the full model (architecture + weights)
model = torch.load("./llama3.1-structured-pruned-full.pth", weights_only=False)

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
        max_new_tokens=50,
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
