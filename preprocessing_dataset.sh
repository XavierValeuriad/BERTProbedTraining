#!/bin/bash
module purge
module load python/3.10.4/1.11.0

python3.10 -m pip install -requierments.txt

python3.10 preprocessing_dataset.py \
    --do_train \
    --overwrite_output_dir \
    --max_seq_length=512 \
    --log_level='info' \
    --logging_first_step='True' \
    --cache_dir='/cache/' \
    --path_save_dataset='/data/tokenized_bookcorpus' \
    --output_dir='/output' \
    --preprocessing_num_workers=20

