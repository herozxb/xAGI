import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM



# ——— CONFIG —————————————————————————————————————————————
model_path   = "/home/deep/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"  # your local checkpoint
device       = torch.device("cuda")
# ————————————————————————————————————————————————————————————

torch.cuda.empty_cache()  # Free unused GPU memory (only useful if you're using a GPU)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=None,
    torch_dtype=torch.float16
).to("cpu")  # if fits entirely in GPU

torch.save(model, "llama3.1-original-full.pth")
