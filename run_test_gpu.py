#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
import gzip
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import math, sys
import types

from datasets.utils import logging as dataset_logging
from datasets import load_from_disk
from dataclasses import field


# import evaluate
import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForMaskedLM,
    HfArgumentParser,
    Trainer,
    is_torch_tpu_available,
    set_seed, BertTokenizerFast, BertConfig
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from torch.distributed.elastic.multiprocessing.errors import record

import torch.distributed as dist
import idr_torch


import concurrent.futures
import itertools
import json, logging, os
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Tuple, List, Union, Dict, Mapping, Callable, Iterable

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, pad_without_fast_tokenizer_warning, _tf_collate_batch, \
    _torch_collate_batch, _numpy_collate_batch

# def create_folder_if_not_exists(folder_path: str):
#     if not os.path.exists(folder_path):
#         logging.info(f'Creating folder `{folder_path}`.')
#         os.makedirs(folder_path)


# def create_all_subfolders_if_not_exists(folder_path: str):
#     logging.info(f'Checking if `{folder_path}` exist, and creating it if not.')
#     if folder_path:
#         path = os.path.normpath(folder_path)
#         splitted_path = path.split(os.sep)
#         if len(splitted_path) == 1:
#             if '.' not in path:
#                 create_folder_if_not_exists(splitted_path[0])
#         elif len(splitted_path) > 1:
#             subpath = os.path.join(path[0], splitted_path[1])
#             if '.' not in subpath:
#                 create_folder_if_not_exists(subpath)
#             for i, _directory in enumerate(splitted_path[2:]):
#                 if '.' not in _directory:
#                     subpath = os.path.join(subpath, _directory)
#                     create_folder_if_not_exists(subpath)
#                 else:
#                     logging.warning(f"Only directories, which means with names not containing a dot '.', are created. Thus, it is assumed that {os.path.join(splitted_path[i:])} is a file.")
#                     break


import torch
from torch import Tensor, amp, nn


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements_training.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@record
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=idr_torch.size,
        rank=idr_torch.rank
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.DEBUG
    logger.setLevel(log_level)
    dataset_logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # print(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {idr_torch.rank}"
        + f" distributed training: {bool(idr_torch.rank != -1)}"
    )

    # Set seed before initializing model.
    set_seed(42)

    for i in range(torch.cuda.device_count()):
        print(f'gpu number {i} : {torch.cuda.get_device_properties(i).name}')

    # tokenized_datasets = concatenate_datasets(
    #     [
    #         load_from_disk(f'data/tokenized_train_bert_{i}')['train'] for i in range(1, 25)
    #     ]
    # )



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()