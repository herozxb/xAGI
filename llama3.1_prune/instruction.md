# 1 cpu training

export CUDA_VISIBLE_DEVICES=""
accelerate launch --cpu QLoRA.py


export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
accelerate launch --cpu --num_processes 1 QLoRA.py


# 2 GPU training
pip install transformers==4.46.0
trl 0.17.0
