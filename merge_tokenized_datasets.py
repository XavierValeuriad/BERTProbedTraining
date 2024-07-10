import logging, os, sys

from datasets import load_from_disk, concatenate_datasets
from datasets.utils.logging import set_verbosity

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from torch.distributed.elastic.multiprocessing.errors import record

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements_training.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@record
def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']  # COMM

    # COMM
    # dist.init_process_group(backend='nccl',
    #         init_method='env://',
    #         world_size=idr_torch.size,
    #         rank=idr_torch.rank)

    parser = HfArgumentParser((TrainingArguments, ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # print(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}.")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # bookcorpus = load_dataset("bookcorpus", cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None)
    # wiki = load_dataset("wikipedia", "20220301.en", cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None)
    # wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
    #
    # assert bookcorpus.features.type == wiki.features.type
    print('Starting to merge the datasets.')

    full_dataset = load_from_disk('data/tokenized_train_bert_1')
    for i in range(2, 25):
        print(f'Merging dataset #{i}.')
        loaded_dataset = load_from_disk(f'data/tokenized_train_bert_{i}')
        full_dataset['train'] = concatenate_datasets(
            [full_dataset['train'], loaded_dataset['train']]
        )
        full_dataset['validation'] = concatenate_datasets(
            [full_dataset['validation'], loaded_dataset['validation']]
        )
        full_dataset.save_to_disk(data_args.path_save_dataset)


    print("End script")