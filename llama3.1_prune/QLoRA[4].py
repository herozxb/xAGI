from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch

#torch.cuda.empty_cache()  # Free unused memory on the GPU
device       = torch.device("cuda")
# ————————————————————————————————————————————————————————————

# 1. Set up the bitsandbytes configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit loading
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation
    bnb_4bit_use_double_quant=True,  # Double quantization (optional, can help with performance)
    bnb_4bit_quant_type="nf4"  # Quantization type, can be "nf4" or "fp4"
)

# 2. Load the pruned model from the saved directory with 4-bit quantization
model = LlamaForCausalLM.from_pretrained(
    "llama3.1-structured-pruned-hf/",  # Path to the saved pruned model directory
    quantization_config=bnb_config,   # Apply 4-bit quantization
    device_map="auto"                 # Automatically allocate model layers across available devices
)

model.train()  # Ensure the model is in training mode

tokenizer = AutoTokenizer.from_pretrained(
    "llama3.1-structured-pruned-hf/",
    use_fast=False
)

lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Add layers that are trainable
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
# … after get_peft_model
model = get_peft_model(model, lora_config)

# 1) Freeze all, then unfreeze LoRA
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

# 2) Sanity check
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"🌟 Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")

# 3) Disable cache + enable checkpointing
model.config.use_cache = False

from datasets import load_dataset

dataset = load_dataset("flytech/python-codes-25k", split="train")

# 2) rename "content" → "text"
#dataset = dataset.rename_column("content", "text")

print(dataset[0])

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=False,
        fp16=True,
        warmup_steps=100,
        max_steps=1000,
        logging_steps=10,
        save_steps=500,
        optim="paged_adamw_8bit",
        save_total_limit=1,
    )
)

trainer.train()

