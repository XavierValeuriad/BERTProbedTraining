#!/bin/bash
module purge
module load pytorch-gpu/py3/1.11.0

nvidia-smi

python3.11 preprocessing_dataset.py \
    --train_file='data/bookcorpus.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='/cache/' \
    --path_save_dataset='/data/tokenized_bookcorpus' \
    --output_dir='/output' \
    --overwrite_cache='False' \
    --preprocessing_num_workers=20

