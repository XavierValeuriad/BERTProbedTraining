#!/bin/bash

# Echo des commandes lancees
set -x -e

export OMP_NUM_THREADS=10

export CUDA_LAUNCH_BLOCKING=1

export SLURM_LOCALID=-1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

python3.10 -u run_training.py \
    --num_train_epochs=1 \
    --save_steps=2 \
    --logging_steps=300 \
    --model_type='bert-base-uncased' \
    --path_load_dataset="data/tokenized_train_bert_1" \
    --output_dir='model_output/' \
    --logging_dir='model_output/logs/' \
    --per_device_train_batch_size=32 \
    --do_train \
    --warmup_steps=10000 \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --report_to='tensorboard' \
    --save_strategy='steps' \
    --skip_memory_metrics='False' \
    --log_level='info' \
    --seed=42 \
    --data_seed=42 \
    --logging_first_step='True' \
    --fp16 \
    --ddp_timeout=600 \
    --ddp_find_unused_parameters='False' \
