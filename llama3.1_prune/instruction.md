# 1 cpu training

export CUDA_VISIBLE_DEVICES=""
accelerate launch --cpu QLoRA.py

