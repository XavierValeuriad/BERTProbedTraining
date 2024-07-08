#!/bin/bash
module purge
module load pytorch-gpu/py3/1.11.0

python3.9 preprocessing_dataset.py \
    --train_file='data/train_bert_1.txt' \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='cache' \
    --path_save_dataset='data/tokenized_train_bert_1' \
    --output_dir='output' \
    --overwrite_cache='False' \
    --preprocessing_num_workers=20

