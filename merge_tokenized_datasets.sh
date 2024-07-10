module purge
module load pytorch-gpu/py3/1.11.0

python3.9 -m pip unsinstall datasets
python3.9 -m pip install python3.9 -m pip unsinstall datasets==1.8.0

python3.9 merge_tokenized_datasets.py \
  --do_train \
  --overwrite_output_dir \
  --max_seq_length=512 \
  --log_level='info' \
  --logging_first_step='True' \
  --cache_dir='cache' \
  --path_save_dataset='data/tokenized_train_bert_full' \
  --output_dir='output' \
  --overwrite_cache='False' \
  --preprocessing_num_workers=20