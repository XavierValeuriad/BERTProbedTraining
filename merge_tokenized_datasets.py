import logging, sys

from datasets import load_from_disk, concatenate_datasets
from datasets.utils.logging import set_verbosity


from transformers.utils.logging import enable_default_handler, enable_explicit_format
from transformers.utils.logging import set_verbosity as transformer_set_verbosity
from torch.distributed.elastic.multiprocessing.errors import record

logger = logging.getLogger(__name__)

@record
def main():

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_mlm", model_args, data_args)

    # training_args.local_rank = idr_torch.local_rank #COMM

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    log_level = logging.DEBUG
    logger.setLevel(log_level)
    set_verbosity(log_level)
    transformer_set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # print(training_args.device)

    # Log on each process the small summary:


    print('Starting to merge the datasets.')

    full_dataset = load_from_disk('data/tokenized_train_bert_14')
    for i in range(15, 25):
        print(f'Merging dataset #{i}.')
        loaded_dataset = load_from_disk(f'data/tokenized_train_bert_{i}')
        full_dataset['train'] = concatenate_datasets([full_dataset['train'], loaded_dataset['train']])
        full_dataset['validation'] = concatenate_datasets(
            [full_dataset['validation'], loaded_dataset['validation']]
        )
        full_dataset.save_to_disk(f'data/tokenized_train_bert_{i}_over_24_half_complete')


full_dataset = load_from_disk('data/tokenized_train_bert_14')
print(full_dataset)
for i in range(15, 25):
    print(f'Merging dataset #{i}.')
    merging_dataset = load_from_disk(f'data/tokenized_train_bert_{i}')
    full_dataset['train'] = concatenate_datasets([full_dataset['train'], merging_dataset['train']])
    full_dataset['validation'] = concatenate_datasets([full_dataset['validation'], merging_dataset['validation']])
    print(full_dataset)
    full_dataset.save_to_disk(f'data/tokenized_train_bert_half_complete_{i}')

ds['train'] = concatenate_datasets([ds['train'], ds1['train']])
ds['validation'] = concatenate_datasets([ds['validation'], ds1['validation']])
# full_dataset['train'] = concatenate_datasets([full_dataset['train'], merging_dataset['train']])
ds.save_to_disk(f'data/tokenized_train_bert_complete_{24}')
#
# full_dataset['validation'] = concatenate_datasets([full_dataset['validation'], merging_dataset['validation']])

    print("End script")