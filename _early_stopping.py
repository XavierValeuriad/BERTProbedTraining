# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
SEED = 42

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import os

id_model = None

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(
    id_model,
    #    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # do_sample=True,
    # temperature=0,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(id_model)

list(model.parameters())[0].data.dtype

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
end_of_turn_token = "<|end_of_turn|>"


def create_conversation(sample):
    global categories_str, label_data_column_name, end_of_turn_token
    _desired_output = sample[label_data_column_name].upper()
    if _desired_output is None:
        _desired_output = "I cannot answer"
    return {
        "prompt": f"Extract the address parts ({categories_str}) from this address : " + sample[
            "COMPLETE_ADDR"] + end_of_turn_token,
        "completion": _desired_output + end_of_turn_token
    }


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from datasets import Dataset

training_df = None

train_dataset = Dataset.from_pandas(training_df).shuffle(seed=SEED)

train_dataset = train_dataset.map(
    create_conversation,
    remove_columns=train_dataset.features,
    batched=False
).shuffle(seed=SEED)

train_dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

evaluation_df = None

evaluation_dataset = Dataset.from_pandas(evaluation_df).shuffle(seed=SEED)

evaluation_dataset = evaluation_dataset.map(
    create_conversation,
    remove_columns=evaluation_dataset.features,
    batched=False
).shuffle(seed=SEED)

evaluation_dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# from trl import setup_chat_format
from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
# peft_config = LoraConfig(
#        lora_alpha=128,
#        lora_dropout=0.05,
#        r=256,
#        bias="none",
#        target_modules="all-linear",
#        task_type="CAUSAL_LM",
# )
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from transformers import TrainingArguments
import time

GRADIENT_BATCH_SIZE = 40
BATCH_SIZE = 12
gradient_accumulation_steps = GRADIENT_BATCH_SIZE/BATCH_SIZE

def auto_batch_size(gradient_batch_size: int, training_arguments: TrainingArguments) -> int:


timestr = time.strftime("%Y%m%d-%H%M%S")

output_dir = f"training_log_{timestr}"

# NOTE: choose max_steps to fit dataset size: qty_of_example_by_step = per_device_train_batch_size * gradient_accumulation_steps
args = TrainingArguments(
    output_dir=os.path.join('_log'),  # directory to save and repository id
    num_train_epochs=100,  # number of training epochs
    max_steps=100,  # 300 on a de bon résultats 8.09 , 400 est mieux, 800 = 8.58, 900 = 8.2
    per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training (c'etait 4)
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=gradient_accumulation_steps,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    optim="adamw_torch_fused",  # use fused adamw optimizer
    logging_steps=1,  # log every 10 steps
    seed=42,
    data_seed=42,
    # save_strategy="steps",                  # save checkpoint every epoch
    # save_strategy="no",                  # save checkpoint every epoche
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    tf32=True,  # use tf32 precision
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # save_total_limit=1,
    # save_steps=30,                    #30 à 60 steps
    # push_to_hub=True,                       # push model to hub
    # report_to="tensorboard",                # report metrics to tensorboard
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from transformers import TrainerCallback


# Define custom to log both traning and evaluation loss as we
class LoggingCallback(TrainerCallback):

    def __init__(self):
        self.training_losses = []
        self.evaluation_losses = []
        self.steps = []

    def on_step_end(self, args, state, control, **kwargs):
        # print(f'args : {args}')
        # print(f'state : {state}')
        # print(f'control : {control}')
        # print(f'kwargs : {kwargs}')
        # print(f'callback : {loss}')
        if state.log_history:
            self.training_losses.append(state.log_history[-2]['loss'])
            self.evaluation_losses.append(state.log_history[-1]['eval_loss'])
            self.steps.append(state.log_history[-1]['step'])


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Define custom callback to implement early stopping
class EarlyStoppingCallback(TrainerCallback):

    def __init__(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            training_loss = state.log_history[-2]['loss']
            eval_loss = state.log_history[-1]['eval_loss']
            step = state.log_history[-1]['step']
            if eval_loss < training_loss:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    control.should_training_stop = True


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from trl import SFTTrainer

max_seq_length = None  # max sequence length for model and packing of the dataset

logger_callback = LoggingCallback()
early_stopping_callback = EarlyStoppingCallback(10)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=evaluation_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    # dataset_text_field='text',
    # nng je décomentehttps://dss-qual.dsp-noprd.aws.cld.cma-cgm.com/projects/GIA/flow/?zoneId=q9HniA0
    packing=True,  # was True
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
    },
    callbacks=[logger_callback, early_stopping_callback],
    # formatting_func=tokenizer.apply_chat_template
    # dataset_text_field="messages"
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### round 1

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
trainer.train()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.pyplot as plt

# Plot loss values against steps
plt.loglog(logger_callback.steps, logger_callback.training_losses, label='Training', color='blue')
plt.loglog(logger_callback.steps, logger_callback.evaluation_losses, label='Evaluation', color='green')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss vs. Step")
plt.legend()
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
metric_df = pd.DataFrame(
    data={
        'STEP': logger_callback.steps,
        'TRAINING_LOSS': logger_callback.training_losses,
        'EVALUATION_LOSS': logger_callback.evaluation_losses
    }
)


