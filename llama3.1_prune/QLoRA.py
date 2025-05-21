from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from datasets import load_dataset

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

#torch.cuda.empty_cache()  # Free unused memory on the GPU
device       = torch.device("cpu")
# ————————————————————————————————————————————————————————————


# allowlist the LlamaForCausalLM class so torch.load can unpickle it
torch.serialization.add_safe_globals([LlamaForCausalLM])

# load the full model (architecture + weights)
model = torch.load("./llama3.1-structured-pruned-full.pth", weights_only=False).to(device)

model.train()  # Ensure the model is in training mode


tokenizer = AutoTokenizer.from_pretrained(
    "/home/deep/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B",
    use_fast=False
)

for param in model.parameters():
    param.requires_grad = True

lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Add layers that are trainable
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

model.config.use_cache = False  # Explicitly set use_cache to False for training

for param in model.parameters():
    param.requires_grad = True

#for name, param in model.named_parameters():
#    if "lora" in name:  # Only LoRA parameters
#        print(f"LoRA layer {name} requires grad: {param.requires_grad}")


from datasets import load_dataset

dataset = load_dataset("codeparrot/codeparrot-clean", split="train")

# 2) rename "content" → "text"
dataset = dataset.rename_column("content", "text")


print(dataset[0])



trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=256,
        gradient_checkpointing=True,
        fp16=True,
        warmup_steps=100,
        max_steps=1000,
        logging_steps=10,
        save_steps=500,
        optim="paged_adamw_8bit",
        save_total_limit=1,
        # deepspeed field removed
        deepspeed="deepspeed_config.json",
    )
)

trainer.train()

