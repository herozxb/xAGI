import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss, MSELoss
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType

# ──────────────── Hyperparameters ────────────────
TEACHER_PATH = "/home/deep/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B"
TEMPERATURE  = 2.0
α, β, γ, δ   = 1.0, 0.5, 0.1, 0.1  # logit, feature, attention, LM weights

# ──────────────── 1) 4-bit Quant Config ────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
)

# ──────────────── 2) Tokenizer ────────────────
tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# ──────────────── 3) Teacher in 4-bit on GPU ────────────────
teacher = LlamaForCausalLM.from_pretrained(
    TEACHER_PATH,
    quantization_config  = bnb_config,
    device_map           = "auto",
    output_hidden_states = True,
    output_attentions    = True,
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# ──────────────── 4) Build & quantize Student ────────────────
student_cfg = LlamaConfig(
    vocab_size              = teacher.config.vocab_size,
    max_position_embeddings = teacher.config.max_position_embeddings,
    hidden_size             = 1024,
    intermediate_size       = 4096,
    num_hidden_layers       = 24,
    num_attention_heads     = 16,
    pad_token_id            = teacher.config.pad_token_id,
    bos_token_id            = teacher.config.bos_token_id,
    eos_token_id            = teacher.config.eos_token_id,
    output_hidden_states    = True,
    output_attentions       = True,
    use_cache               = False,
)

# 4a) Instantiate FP32 student & save
student_fp32 = LlamaForCausalLM(student_cfg)
student_fp32.save_pretrained("student_tmp")
tokenizer.save_pretrained("student_tmp")

# 4b) Reload student in 4-bit
student = LlamaForCausalLM.from_pretrained(
    "student_tmp",
    quantization_config  = bnb_config,
    device_map           = "auto",
    output_hidden_states = True,
    output_attentions    = True,
)
student.train()

# ──────────────── 5) Attach LoRA adapters ────────────────
lora_config = LoraConfig(
    r            = 8,
    lora_alpha   = 32,
    target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0.05,
    bias         = "none",
    task_type    = TaskType.CAUSAL_LM,
)
student = get_peft_model(student, lora_config)

# 5b) Freeze all *base* (quantized) params, unfreeze only adapters
for name, param in student.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total     = sum(p.numel() for p in student.parameters())
print(f"🔑 Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")

# ──────────────── 6) Dataset & Preprocessing ────────────────
raw = load_dataset("flytech/python-codes-25k", split="train")

def preprocess(ex):
    prompt = ex["instruction"]
    if ex["input"].strip():
        prompt += "\n" + ex["input"]
    full = prompt + "\n" + ex["output"]

    toks = tokenizer(
        full,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    labels = [
        (tok if tok != tokenizer.pad_token_id else -100)
        for tok in toks["input_ids"]
    ]
    toks["labels"] = labels
    return toks

ds = raw.map(preprocess, remove_columns=raw.column_names)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ──────────────── 7) DistilTrainer ────────────────
class DistilTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels      = inputs.pop("labels")
        # Student forward
        s_out     = model(**inputs)
        s_logits  = s_out.logits
        s_hiddens = s_out.hidden_states
        s_attns   = s_out.attentions

        # Teacher forward
        with torch.no_grad():
            t_out = teacher(**inputs)
        t_logits  = t_out.logits.float()
        t_hiddens = [h.float() for h in t_out.hidden_states]
        t_attns   = [a.float() for a in t_out.attentions]

        # 1) Logit distillation
        L_logits = KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits.float()/TEMPERATURE, dim=-1),
            F.softmax(   t_logits            /TEMPERATURE, dim=-1),
        ) * (TEMPERATURE**2)

        # 2) Feature‐distillation
        Hs = s_hiddens[0].size(-1)
        feat_losses = []
        for sh, th in zip(s_hiddens, t_hiddens):
            B,T,_ = th.shape
            ratio = th.size(-1) // Hs
            th_ds = th.view(B,T,Hs,ratio).mean(-1)
            feat_losses.append(MSELoss()(sh.float(), th_ds))
        L_feat = torch.stack(feat_losses).mean()

        # 3) Attention‐distillation
        attn_losses = []
        for sa, ta in zip(s_attns, t_attns):
            B,Ht,T1,T2 = ta.shape
            heads      = sa.size(1)
            ta_ds      = ta.view(B,heads,Ht//heads,T1,T2).mean(2)
            attn_losses.append(MSELoss()(sa.float(), ta_ds))
        L_attn = torch.stack(attn_losses).mean()

        # 4) LM loss
        L_lm = F.cross_entropy(
            s_logits.view(-1, s_logits.size(-1)),
            torch.tensor(labels, device=s_logits.device).view(-1),
            ignore_index=-100,
        )

        loss = α*L_logits + β*L_feat + γ*L_attn + δ*L_lm
        return (loss, s_out) if return_outputs else loss


# ──────────────── 8) Training ────────────────
training_args = TrainingArguments(
    output_dir="distil_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,
    max_steps=2000,
    logging_steps=100,
    save_steps=500,
    remove_unused_columns=False,
)

trainer = DistilTrainer(
    model=student,
    args=training_args,
    train_dataset=ds,
    data_collator=collator,
)

trainer.train()
student.save_pretrained("student-1B-distilled")
tokenizer.save_pretrained("student-1B-distilled")

