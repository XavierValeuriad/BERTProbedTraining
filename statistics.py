import concurrent.futures
import itertools
import json, logging, os
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from socket import socket
from typing import Optional, Any, Tuple, List, Union, Dict, Mapping

import torch

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, pad_without_fast_tokenizer_warning, _tf_collate_batch, \
    _torch_collate_batch, _numpy_collate_batch

from utils import create_all_subfolders_if_not_exists, hash_tensor

_SAVING_THREAD_POOL = concurrent.futures.ThreadPoolExecutor()
_STATISTICS_DIRECTORY_PATH = os.path.join(
    'statistics'
)
create_all_subfolders_if_not_exists(_STATISTICS_DIRECTORY_PATH)


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
    create_all_subfolders_if_not_exists(
        os.path.join(
            _STATISTICS_DIRECTORY_PATH,
            GRADIENT_STATISTICS_DIRECTORY_NAME
        )
    )
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
            os.path.join(CallbackForGradientStatistics.GRADIENT_STATISTICS_DIRECTORY_NAME, f'counter_{next(CallbackForGradientStatistics._ATOMIC_COUNTER)}@collarcounter_{StatisticalDataCollatorForLanguageModeling._ATOMIC_COUNTER}@epoch_{epoch}@device_{torch.cuda.current_device()}@hostname_{socket.gethostname()}@time_{datetime.now().strftime("%I:%M%p on %B %d, %Y")}.json'),
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
    create_all_subfolders_if_not_exists(
        os.path.join(
            _STATISTICS_DIRECTORY_PATH,
            MASKING_STATISTICS_DIRECTORY_NAME
        )
    )

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
                f'counter_{next(StatisticalDataCollatorForLanguageModeling._ATOMIC_COUNTER)}@callbackcounter_{CallbackForGradientStatistics._ATOMIC_COUNTER}@epoch_{CallbackForGradientStatistics._CURRENT_EPOCH}@device_{torch.cuda.current_device()}@hostname_{socket.gethostname()}@time_{datetime.now().strftime("%I:%M%p on %B %d, %Y")}.json'
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
