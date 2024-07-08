#!/bin/bash
module purge
module load python/3.11.5

python3.11 -m ensurepip
python3.11 -m ensurepip --upgrade
python3.11 -m pip install --user --no-cache-dir -r requirements.txt

python3.11 preprocessing_dataset.py \
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

