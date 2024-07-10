module purge
module load pytorch-gpu/py3/1.12.1

python3.9 -m pip install --user --no-cache-dir -r requirements_tokenization.txt

python3.9 merge_tokenized_datasets.py