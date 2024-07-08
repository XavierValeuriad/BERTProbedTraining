from datasets import load_from_disk
raw_dataset = load_from_disk('train_bert.hf').shuffle(seed=42)
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

from tqdm import tqdm
from math import floor

big_text = ''
min_counter, max_counter, counter = 0, int(floor(0.33*len(raw_dataset))), 0
print(f'len(raw_dataset) : {len(raw_dataset)}.')
print(f'min_counter : {min_counter}.')
print(f'max_counter : {max_counter}.')
for example in tqdm(raw_dataset.select(range(min_counter, max_counter))):
    big_text += f"{example['text']}\n"

with open('train_bert_1.txt', 'w') as file:
    file.write(big_text)

big_text = ''
min_counter, max_counter, counter = int(floor(0.33*len(raw_dataset))), int(floor(0.67*len(raw_dataset))), 0
print(f'len(raw_dataset) : {len(raw_dataset)}.')
print(f'min_counter : {min_counter}.')
print(f'max_counter : {max_counter}.')
for example in tqdm(raw_dataset.select(range(min_counter, max_counter))):
    big_text += f"{example['text']}\n"

with open('train_bert_2.txt', 'w') as file:
    file.write(big_text)

big_text = ''
min_counter, max_counter, counter = int(floor(0.67*len(raw_dataset))), len(raw_dataset), 0
print(f'len(raw_dataset) : {len(raw_dataset)}.')
print(f'min_counter : {min_counter}.')
print(f'max_counter : {max_counter}.')
for example in tqdm(raw_dataset.select(range(min_counter, max_counter))):
    big_text += f"{example['text']}\n"

with open('train_bert_3.txt', 'w') as file:
    file.write(big_text)
