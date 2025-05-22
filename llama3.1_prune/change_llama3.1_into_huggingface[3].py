# Load your model from .pth
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

torch.serialization.add_safe_globals([LlamaForCausalLM])
model = torch.load("./llama3.1-structured-pruned-full.pth", weights_only=False)

tokenizer = AutoTokenizer.from_pretrained(
    "/home/deep/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B",
    use_fast=False
)

model.save_pretrained("llama3.1-structured-pruned-hf/")
tokenizer.save_pretrained("llama3.1-structured-pruned-hf/")
