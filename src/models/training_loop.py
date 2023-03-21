import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Set, Optional
from typing_extensions import Literal
import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    AutoModelForMaskedLM
)
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version

from src.data.dataloader import StateDataloader

# check_min_version("4.27.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Trainer():
    def __init__(self,
                 model_name_or_path: str,
                 output_dir: str,
                 dataloaders: Set[DataLoader],
                 dataset_name: Optional[str] = None,
                 dataset_config_name: Optional[str] = None,
                 val_max_target_length: Optional[int] = 50,
                 ignore_pad_token_for_loss: bool = True,
                 num_beams: Optional[int] = 4,
                 pad_to_max_length: bool = True,
                 config_name: Optional[str] = None,
                 tokenizer_name: Optional[str] = None,
                 use_slow_tokenizer: bool = False,
                 per_device_train_batch_size: Optional[int] = 8,
                 per_device_eval_batch_size: Optional[int] = 8,
                 learning_rate: Optional[float] = 5e-5,
                 weight_decay: Optional[float] = 0.0,
                 num_train_epochs: Optional[int] = 3,
                 max_train_steps: Optional[int] = None,
                 gradient_accumulation_steps: Optional[int] = 1,
                 lr_scheduler_type: Literal = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                               "constant_with_warmup"],
                 num_warmup_steps: Optional[int] = 0,
                 seed: Optional[int] = None,
                 model_type: Optional[str] = None,
                 checkpointing_steps: Optional[str] = None,
                 resume_from_checkpoint: Optional[str] = None,
                 with_tracking: bool = False,
                 report_to: Optional[str] = None):

        # Save the input parameters
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.dataloaders = dataloaders
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.val_max_target_length = val_max_target_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.num_beams = num_beams
        self.pad_to_max_length = pad_to_max_length
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.use_slow_tokenizer = use_slow_tokenizer
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.seed = seed
        self.model_type = model_type
        self.checkpointing_steps = checkpointing_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.with_tracking = with_tracking
        self.report_to = report_to

        self.dataloaders = self.dataloaders.__call__()

        accelerator_log_kwargs = {}

        if self.with_tracking:
            accelerator_log_kwargs["log_with"] = self.report_to
            accelerator_log_kwargs["logging_dir"] = self.output_dir

        self.accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps,
                                       **accelerator_log_kwargs)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        if self.config_name:
            config = AutoConfig.from_pretrained(self.config_name)
        elif self.model_name_or_path:
            config = AutoConfig.from_pretrained(self.model_name_or_path)
        else:
            config = CONFIG_MAPPING[self.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        if self.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=not self.use_slow_tokenizer)
        elif self.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                           use_fast=not self.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        if self.model_name_or_path:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_name_or_path),
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            self.model = AutoModelForMaskedLM.from_config(config)

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self.seed is not None:
            set_seed(self.seed)

        # Set up the optimizer
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.dataloaders['train']) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.max_train_steps * self.gradient_accumulation_steps,
        )

        self.model, self.optimizer, self.dataloaders['train'], self.dataloaders[
            'eval'], self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['eval'], self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.dataloaders['train']) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.max_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        if self.checkpointing_steps is not None and self.checkpointing_steps.isdigit():
            self.checkpointing_steps = int(checkpointing_steps)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.with_tracking:
            experiment_config = {
                "model_name_or_path": self.model_name_or_path,
                "val_max_target_length": self.val_max_target_length,
                "num_beams": self.num_beams,
                "pad_to_max_length": self.pad_to_max_length,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_train_epochs": self.num_train_epochs,
                "max_train_steps": self.max_train_steps,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "lr_scheduler_type": self.lr_scheduler_type,
                "num_warmup_steps": self.num_warmup_steps,
                "seed": self.seed,
                "model_type": self.model_type
            }
            print(experiment_config)
            # TensorBoard cannot log Enums, need the raw value
            if isinstance(experiment_config["lr_scheduler_type"], str):
                experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
            else:
                experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            self.accelerator.init_trackers("state_prediction", experiment_config)

        # Metric
        self.metric = evaluate.load("rouge")

        self.total_batch_size = self.per_device_train_batch_size * self.accelerator.num_processes * self.gradient_accumulation_steps

    def train(self):

        logger.info("***** Running training *****")
        #     logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.resume_from_checkpoint:
            if self.resume_from_checkpoint is not None or self.resume_from_checkpoint != "":
                self.accelerator.print(f"Resumed from checkpoint: {self.resume_from_checkpoint}")
                self.accelerator.load_state(self.resume_from_checkpoint)
                path = os.path.basename(self.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * self.gradient_accumulation_steps
                starting_epoch = resume_step // len(self.dataloaders['train'])
                resume_step -= starting_epoch * len(self.dataloaders['train'])

        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * self.num_update_steps_per_epoch)
        completed_steps = starting_epoch * self.num_update_steps_per_epoch
        for epoch in range(starting_epoch, self.num_train_epochs):
            self.model.train()
            if self.with_tracking:
                total_loss = 0
                total_loss_eval = 0
            for step, batch in enumerate(self.dataloaders['train']):
                # We need to skip steps until we reach the resumed step
                if self.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % self.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            completed_steps += 1
                        continue

                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if self.with_tracking:
                        total_loss += loss.detach().float()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.accelerator.log({"training_loss_batch": loss.detach().float()}, step=step)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(self.checkpointing_steps, int):
                    if completed_steps % self.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if self.output_dir is not None:
                            output_dir = os.path.join(self.output_dir, output_dir)
                        self.accelerator.save_state(output_dir)

                if completed_steps >= self.max_train_steps:
                    break

            self.model.eval()
            gen_kwargs = {
                "max_length": self.val_max_target_length,
                "num_beams": self.num_beams,
            }

            for step, batch in enumerate(self.dataloaders['eval']):
                with torch.no_grad():
                    generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

                    generated_tokens = self.accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                    )
                    labels = batch["labels"]
                    if not self.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = self.accelerator.pad_across_processes(batch["labels"], dim=1,
                                                                       pad_index=self.tokenizer.pad_token_id)

                    generated_tokens, labels = self.accelerator.gather_for_metrics((generated_tokens, labels))
                    generated_tokens = generated_tokens.cpu().numpy()
                    labels = labels.cpu().numpy()

                    if self.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                    decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
                    self.metric.add_batch(
                        predictions=decoded_preds,
                        references=decoded_labels,
                    )

                    # Compute and log the loss
                    outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                         labels=batch["labels"])
                    loss = outputs.loss
                    if self.with_tracking:
                        total_loss_eval += loss.detach().float()
            result = self.metric.compute(use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}

            logger.info(result)
            if self.with_tracking:
                result["train_loss"] = total_loss.item() / len(self.dataloaders['train'])
                result["epoch"] = epoch
                result["eval_loss"] = total_loss_eval.item() / len(self.dataloaders['eval'])
                self.accelerator.log(result, step=completed_steps)

            if self.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if self.output_dir is not None:
                    output_dir = os.path.join(self.output_dir, output_dir)
                self.accelerator.save_state(output_dir)

        if self.output_dir is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                self.output_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
            )
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.output_dir)
                all_results = {f"eval_{k}": v for k, v in result.items()}
                with open(os.path.join(self.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels





