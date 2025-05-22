import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM

import time


import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# ——— CONFIG —————————————————————————————————————————————
model_path   = "/home/deep/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B"  # your local checkpoint
device       = torch.device("cuda")
# ————————————————————————————————————————————————————————————


# allowlist the LlamaForCausalLM class so torch.load can unpickle it
torch.serialization.add_safe_globals([LlamaForCausalLM])

# load the full model (architecture + weights)
model = torch.load("./llama3.1-structured-pruned-full.pth", weights_only=False).to(device)

# 1) Reload tokenizer (same as before)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/deep/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B",
    use_fast=False
)

# 2) Put model in eval mode
model.eval()

# 3) Prepare your prompt
prompt = "write a python to add two number, give me the python code"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

while 1:
    start = time.time()
    # 4) Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))    
    print("=====llama3.1 running time is :======")    
    print(time.time() - start)
    input()
# I see a fox, whittle you down, on you, how to fight the fox?
# I can't remember
# 222.12610292434692

# 5) Decode and print
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
