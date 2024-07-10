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

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging, math, os, sys
from datasets.utils import logging as dataset_logging
from datasets import load_from_disk
from dataclasses import dataclass, field
from typing import Optional


# import evaluate
import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForMaskedLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed, BertTokenizerFast, BertConfig,
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
from typing import Optional, Any, Tuple, List, Union, Dict, Mapping

import torch

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
from torch import Tensor

from utils import create_folder_if_not_exists

MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64


def hash_tensor(x: Tensor) -> Tensor:
    assert x.dtype == torch.int64
    while x.ndim > 0:
        x = _reduce_last_axis(x)
    return x


@torch.no_grad()
def _reduce_last_axis(x: Tensor) -> Tensor:
    assert x.dtype == torch.int64
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        acc *= MULTIPLIER
        acc += INCREMENT
        acc += x[..., i]
        # acc %= MODULUS  # Not really necessary.
    return acc


_SAVING_THREAD_POOL = concurrent.futures.ThreadPoolExecutor()
_STATISTICS_DIRECTORY_PATH = 'statistics'
# create_folder_if_not_exists(_STATISTICS_DIRECTORY_PATH)


def _save_json(subfolder_and_file: str, gradient_statistics: dict) -> None:
    with open(os.path.join(_STATISTICS_DIRECTORY_PATH, subfolder_and_file), "w") as file:
        file.write(
            json.dumps(gradient_statistics, indent=4)
        )


def _save_json(subfolder_and_file: str, gradient_statistics: dict) -> None:
    with open(os.path.join(_STATISTICS_DIRECTORY_PATH, subfolder_and_file), "w") as file:
        file.write(
            json.dumps(gradient_statistics, indent=4)
        )


class CallbackForGradientStatistics(TrainerCallback):

    def __init__(self, norm_type: float = 2.0):
        print(f'CallbackForGradientStatistics.__init__(...) : calling.')
        logging.info(f'CallbackForGradientStatistics.__init__(...) : calling.')
        self.norm_type = float(norm_type)

    CURRENT_MODEL = None
    GRADIENT_STATISTICS_DIRECTORY_NAME = 'gradient'

    _ATOMIC_COUNTER = itertools.count()
    _CURRENT_EPOCH = 0.0

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        # model = kwargs["model"]
        model = CallbackForGradientStatistics.CURRENT_MODEL
        if model is None:
            raise Exception('Please set the current model into the class variable StatisticalCallback.CURRENT_MODEL.')

        epoch = state.epoch
        logging.info(f'CallbackForGradientStatistics.on_substep_end : {epoch}.')

        CallbackForGradientStatistics._CURRENT_EPOCH = epoch

        # Computing gradient statistics (per layer)
        _index = 0
        gradient_statistics = {
            f'{parameter_name}#{++_index}':{
                'norm': parameters.grad.data.norm(self.norm_type).item(),
                'argmax': parameters.grad.data.argmax().item(),
                'max': parameters.grad.data.max().item(),
                'argmin': parameters.grad.data.argmin().item(),
                'min': parameters.grad.data.min().item(),
                'histograms': torch.histogram(parameters.grad.data, bins=24)
            }
            for parameter_name, parameters in model.named_parameters() if parameters.grad is not None
        }

        # _save_json(
        #     os.path.join(
        #         CallbackForGradientStatistics.GRADIENT_STATISTICS_DIRECTORY_NAME,
        #         f'counter_{next(CallbackForGradientStatistics._ATOMIC_COUNTER)}@collarcounter_{StatisticalDataCollatorForLanguageModeling._ATOMIC_COUNTER}@epoch_{epoch}@device_{torch.cuda.current_device()}@hostname_{socket.gethostname()}@time_{datetime.now().strftime("%I:%M%p on %B %d, %Y")}.json'
        #     ),
        #     gradient_statistics
        # )

        _SAVING_THREAD_POOL.submit(
            _save_json,
            os.path.join(CallbackForGradientStatistics.GRADIENT_STATISTICS_DIRECTORY_NAME, f'counter_{next(CallbackForGradientStatistics._ATOMIC_COUNTER)}@collarcounter_{StatisticalDataCollatorForLanguageModeling._ATOMIC_COUNTER}@epoch_{epoch}@device_{torch.cuda.current_device()}@time_{datetime.now().strftime("%I:%M%p on %B %d, %Y")}.json'),
            gradient_statistics
        )


@dataclass
class StatisticalDataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    _ATOMIC_COUNTER = itertools.count()

    def __post_init__(self):
        print(f'CallbackForGradientStatistics.__post_init__(...) : calling.')
        logging.info(f'CallbackForGradientStatistics.__post_init__(...) : calling.')
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    @staticmethod
    def tf_bernoulli(shape, probability):
        import tensorflow as tf

        prob_matrix = tf.fill(shape, probability)
        return tf.cast(prob_matrix - tf.random.uniform(shape, 0, 1) >= 0, tf.bool)

    def tf_mask_tokens(
        self, inputs: Any, vocab_size, mask_token_id, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import tensorflow as tf

        mask_token_id = tf.cast(mask_token_id, inputs.dtype)

        input_shape = tf.shape(inputs)
        # 1 for a special token, 0 for a normal token in the special tokens mask
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        masked_indices = self.tf_bernoulli(input_shape, self.mlm_probability) & ~special_tokens_mask
        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        labels = tf.where(masked_indices, inputs, -100)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, 0.8) & masked_indices

        inputs = tf.where(indices_replaced, mask_token_id, inputs)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = self.tf_bernoulli(input_shape, 0.1) & masked_indices & ~indices_replaced
        random_words = tf.random.uniform(input_shape, maxval=vocab_size, dtype=inputs.dtype)

        inputs = tf.where(indices_random, random_words, inputs)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        import tensorflow as tf

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="tf", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _tf_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in batch["input_ids"].numpy().tolist()
                ]
                # Cannot directly create as bool
                special_tokens_mask = tf.cast(tf.convert_to_tensor(special_tokens_mask, dtype=tf.int64), tf.bool)
            else:
                special_tokens_mask = tf.cast(special_tokens_mask, tf.bool)
            batch["input_ids"], batch["labels"] = self.tf_mask_tokens(
                tf.cast(batch["input_ids"], tf.int64),
                special_tokens_mask=special_tokens_mask,
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=len(self.tokenizer),
            )
        else:
            labels = batch["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                # Replace self.tokenizer.pad_token_id with -100
                labels = tf.where(labels == self.tokenizer.pad_token_id, -100, labels)
            else:
                labels = tf.identity(labels)  # Makes a copy, just in case
            batch["labels"] = labels
        return batch

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
            logging.info('torch_mask')
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    MASKING_STATISTICS_DIRECTORY_NAME = 'masking'

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        print(f'CallbackForGradientStatistics.torch_mask_tokens(...) : calling.')
        logging.info(f'CallbackForGradientStatistics.torch_mask_tokens(...) : calling.')

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices

        replaced_token_ids = inputs[indices_replaced]
        replaced_token_locations = indices_replaced.nonzero()

        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

        randomized_token_ids = inputs[indices_random]
        randomized_token_locations = indices_random.nonzero()

        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)

        random_token_ids = random_words

        _masking_data = {
            'input_hash': hash_tensor(inputs),
            'replaced_token_ids': replaced_token_ids,
            'replaced_token_locations': replaced_token_locations,
            'randomized_token_ids': randomized_token_ids,
            'randomized_token_locations': randomized_token_locations,
            'random_token_ids': random_token_ids
        }

        # _save_json(
        #     os.path.join(StatisticalDataCollatorForLanguageModeling.MASKING_STATISTICS_DIRECTORY_NAME,
        #                  f'counter_{next(StatisticalDataCollatorForLanguageModeling._ATOMIC_COUNTER)}@callbackcounter_{CallbackForGradientStatistics._ATOMIC_COUNTER}@epoch_{CallbackForGradientStatistics._CURRENT_EPOCH}@device_{torch.cuda.current_device()}@hostname_{socket.gethostname()}@time_{datetime.now().strftime("%I:%M%p on %B %d, %Y")}.json'),
        #     _masking_data
        # )

        _SAVING_THREAD_POOL.submit(
            _save_json,
            os.path.join(
                StatisticalDataCollatorForLanguageModeling.MASKING_STATISTICS_DIRECTORY_NAME,
                f'counter_{next(StatisticalDataCollatorForLanguageModeling._ATOMIC_COUNTER)}@callbackcounter_{CallbackForGradientStatistics._ATOMIC_COUNTER}@epoch_{CallbackForGradientStatistics._CURRENT_EPOCH}@device_{torch.cuda.current_device()}@time_{datetime.now().strftime("%I:%M%p on %B %d, %Y")}.json'
            ),
            _masking_data
        )

        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(bool)

        probability_matrix[special_tokens_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.23.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # dataset_name: Optional[str] = field(
    #     default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    # )
    path_load_dataset: Optional[str] = field(
        default=None, metadata={"help": "The path to load the tokenized dataset."}
    )
    # dataset_config_name: Optional[str] = field(
    #     default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    # )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.path_load_dataset is None:
            raise ValueError("Need path to load the tokenized dataset (path_load_dataset).")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # model_name_or_path: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
    #         )
    #     },
    # )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    # config_overrides: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Override some existing default config settings when a model is trained from scratch. Example: "
    #             "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
    #         )
    #     },
    # )
    # config_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    # )
    # tokenizer_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    # )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    # use_fast_tokenizer: bool = field(
    #     default=True,
    #     metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    # )
    # model_revision: str = field(
    #     default="main",
    #     metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    # )
    # use_auth_token: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
    #             "with private models)."
    #         )
    #     },
    # )
    #
    def __post_init__(self):
        if self.model_type is None:
            raise ValueError('Please provide a model_type argument.')
        # if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
        #     raise ValueError(
        #         "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
        #     )




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

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_mlm", model_args, data_args)

    training_args.local_rank = idr_torch.local_rank

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    dataset_logging.set_verbosity(log_level)
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
    logger.info(f"Training/evaluation parameters {training_args}")

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

    tokenized_datasets = load_from_disk(data_args.path_load_dataset)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    config = BertConfig.from_dict(
        {
            "_name_or_path": "bert-base-uncased",
            "architectures": [
                "BertForMaskedLM"
            ],
            "attention_probs_dropout_prob": 0.1,
            "classifier_dropout": None,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.38.2",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": 30522
        }
    )

    # if model_args.config_overrides is not None:
    #     logger.info(f"Overriding config: {model_args.config_overrides}")
    #     config.update_from_string(model_args.config_overrides)
    #     logger.info(f"New config: {config}")

    model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        # metric = evaluate.load("./accuracy.py")
        compute_metrics = None
        # def compute_metrics(eval_preds):
        #     preds, labels = eval_preds
        #     # preds have the same shape as the labels, after the argmax(-1) has been calculated
        #     # by preprocess_logits_for_metrics
        #     labels = labels.reshape(-1)
        #     preds = preds.reshape(-1)
        #     mask = labels != -100
        #     labels = labels[mask]
        #     preds = preds[mask]
        #     return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = StatisticalDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        callbacks=[CallbackForGradientStatistics()],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()