from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Some matrices hidden dimension is not a multiple of 64")


# ------------------------------------------------------------------
# 1. 4-bit quantization config  (define this *before* you use it)
# ------------------------------------------------------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_compute_dtype=torch.float16,   # or "float16"
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ------------------------------------------------------------------
# 2. load frozen 4-bit backbone
# ------------------------------------------------------------------
base = LlamaForCausalLM.from_pretrained(
    "llama3.1-structured-pruned-hf",   # dir with backbone weights+config
    quantization_config=bnb_cfg,       # <-- use the variable you just made
    device_map="auto",
)

# ------------------------------------------------------------------
# 3. attach LoRA adapter you saved in outputs/
# ------------------------------------------------------------------
model = PeftModel.from_pretrained(base, "outputs/checkpoint-1000")   # adapter dir
model.eval()                                         # inference mode

# ------------------------------------------------------------------
# 4. tokenizer
# ------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(
    "llama3.1-structured-pruned-hf", use_fast=False
)

# quick test
prompt = "Help me split the bill among my friends!"
ids = tok(prompt, return_tensors="pt").to(model.device)
start = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=500)
print("==========================output==============================")
print(time.time() - start)
print(tok.decode(out[0], skip_special_tokens=True))

