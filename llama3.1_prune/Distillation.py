import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss, MSELoss
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# 1) Hyper-parameters
TEACHER = "/home/deep/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B"
DATASET = "flytech/python-codes-25k"
TEMPERATURE = 2.0
α, β, γ, δ = 1.0, 0.5, 0.1, 0.1   # logit, feat, attn, LM weights

# 2) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TEACHER, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 3) Teacher (frozen, on CPU)
teacher = LlamaForCausalLM.from_pretrained(
    TEACHER,
    device_map="cpu",
    torch_dtype=torch.float16,
    output_hidden_states=True,
    output_attentions=True,
)

teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

# 4) Student config → fresh 1 B model on GPU
student_cfg = LlamaConfig(
    vocab_size           = teacher.config.vocab_size,
    max_position_embeddings = teacher.config.max_position_embeddings,
    hidden_size          = 1024,
    intermediate_size    = 4096,
    num_hidden_layers    = 24,
    num_attention_heads  = 16,
    pad_token_id         = teacher.config.pad_token_id,
    bos_token_id         = teacher.config.bos_token_id,
    eos_token_id         = teacher.config.eos_token_id,
    output_hidden_states = True,
    output_attentions    = True,
    use_cache            = False,
)
student = LlamaForCausalLM(student_cfg).to("cuda")

# 5) Dataset & preprocessing
raw = load_dataset(DATASET, split="train")

print(raw[0])

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
    
    labels = toks["input_ids"].copy()
    
    # mask out pad tokens with -100
    labels = [
        (lab if lab != tokenizer.pad_token_id else -100)
        for lab in labels
    ]
    toks["labels"] = labels

    return toks

ds = raw.map(preprocess, remove_columns=raw.column_names)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


class DistilTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    
        # 1) pop labels
        labels = inputs.pop("labels")                     # [B, T]

        # 2) STUDENT forward (still in fp16/autocast)
        s_out     = model(**inputs)
        s_logits  = s_out.logits                          # fp16 [B, T, V]
        s_hiddens = s_out.hidden_states                   # tuple of fp16 [B, T, Hs]
        s_attns   = s_out.attentions                      # tuple of fp16 [B, Hs_heads, T, T]

        # 3) TEACHER forward on CPU (fp16), pull to GPU & cast to fp32
        with torch.no_grad():
            cpu_in = {
                "input_ids":      inputs["input_ids"].to("cpu"),
                "attention_mask": inputs["attention_mask"].to("cpu"),
            }
            t_out      = teacher(**cpu_in)
            t_logits   = t_out.logits.to(s_logits.device).float()
            t_hiddens  = [h.to(s_logits.device).float() for h in t_out.hidden_states]
            t_attns    = [a.to(s_logits.device).float() for a in t_out.attentions]

        # 4.1) Logit KL — cast student logits to fp32
        L_logits = KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits.float() / TEMPERATURE, dim=-1),
            F.softmax(   t_logits            / TEMPERATURE, dim=-1),
        ) * (TEMPERATURE ** 2)

        # 4.2) Feature MSE — cast student hidden to fp32
        feat_losses = []
        Hs = s_hiddens[0].size(-1)
        for s_h, t_h in zip(s_hiddens, t_hiddens):
            s_h_f = s_h.float()                 # [B, T, Hs]
            B,T,Ht = t_h.shape
            ratio = Ht // Hs
            t_h_ds = t_h.view(B, T, Hs, ratio).mean(-1)
            feat_losses.append(MSELoss()(s_h_f, t_h_ds))
        L_feat = torch.stack(feat_losses).mean()

        # 4.3) Attention MSE — cast student attn to fp32
        attn_losses = []
        for s_a, t_a in zip(s_attns, t_attns):
            s_a_f = s_a.float()                # [B, Hs_heads, T, T]
            B,Ht,T1,T2 = t_a.shape
            heads = s_a_f.size(1)
            ratio = Ht // heads
            t_a_ds = t_a.view(B, heads, ratio, T1, T2).mean(2)
            attn_losses.append(MSELoss()(s_a_f, t_a_ds))
        L_attn = torch.stack(attn_losses).mean()

        # 4.4) LM loss — on fp32
        L_lm = F.cross_entropy(
            s_logits.float().view(-1, s_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 5) combine
        loss = α * L_logits + β * L_feat + γ * L_attn + δ * L_lm
        
        return (loss, s_out) if return_outputs else loss
        
# 7) Train!
args = TrainingArguments(
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
    args=args,
    train_dataset=ds,
    data_collator=collator,
)

trainer.train()
trainer.save_model("student-1B-distilled")
tokenizer.save_pretrained("student-1B-distilled")

