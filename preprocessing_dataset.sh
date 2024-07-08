#!/bin/bash
module load pytorch-gpu/py3/1.11.0

python preprocessing_dataset.py \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='/cache/' \
    --train_file='/data/bookcorpus.hf' \
    --path_save_dataset="/data/tokenized_bookcorpus" \
    --output_dir='/output' \
    --preprocessing_num_workers=20

