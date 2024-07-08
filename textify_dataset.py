from datasets import load_from_disk
raw_dataset = load_from_disk('train_bert.hf').shuffle(seed=42)
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

from tqdm import tqdm
from math import floor

number_of_splits = 24
percent_steps = 1.0/number_of_splits

for split_number in range(number_of_splits):
    big_text = ''
    min_counter, max_counter = int(floor(split_number*percent_steps*len(raw_dataset))), int(floor((split_number+1)*percent_steps*len(raw_dataset)))
    print(f'len(raw_dataset) : {len(raw_dataset)}')
    print(f'min_counter : {min_counter}')
    print(f'max_counter : {max_counter}')
    for example in tqdm(raw_dataset.select(range(min_counter, max_counter))):
        big_text += f"{example['text']}\n"
    with open(f'train_bert_{split_number+1}.txt', 'w') as file:
        file.write(big_text)

