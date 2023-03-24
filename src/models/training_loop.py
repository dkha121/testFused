
import logging
import math
import numpy as np

from typing import Set, Optional
from typing_extensions import Literal

import evaluate
import nltk
nltk.download('punkt')
import torch



from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    get_scheduler,
    AutoModelForMaskedLM
)
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from transformers.utils.versions import require_version



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Trainer():
    def __init__(self,
             model_name_or_path: str,
             output_dir: str,
             dataloaders: Set[DataLoader],

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
             mixed_precision: Literal = ['no', 'fp16', 'bf16'],

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
        self.mixed_precision = mixed_precision

        self.seed = seed
        self.model_type = model_type

        self.checkpointing_steps = checkpointing_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.with_tracking = with_tracking
        self.report_to = report_to


        self.dataloaders = self.dataloaders.__call__()



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
            'eval'], self.lr_scheduler = (
            self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['eval'], self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.dataloaders['train']) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.max_train_steps = self.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)


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


        # Metric
        self.metric = evaluate.load("rouge")

        self.total_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps

    def train(self):

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        # Only show the progress bar once on each machine.

        starting_epoch = 0
        tb_writer = SummaryWriter('experiment')
        progress_bar = tqdm(range(self.max_train_steps))


        progress_bar.update(starting_epoch * self.num_update_steps_per_epoch)
        completed_steps = starting_epoch * self.num_update_steps_per_epoch
        for epoch in range(starting_epoch, self.num_train_epochs):
            self.model.train()
            if self.with_tracking:
                total_loss = 0
                total_loss_eval = 0
            for step, batch in enumerate(self.dataloaders['train']):
                # We need to skip steps until we reach the resumed step
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.detach().float()
                tb_writer.add_scalar('training_loss_batch', loss.item(), step)


                progress_bar.update(1)
                completed_steps += 1

                if completed_steps >= self.max_train_steps:
                    break
            eval_loss, result = self.evaluate()
            total_loss_eval += eval_loss
            tb_writer.add_scalar('train_loss', total_loss.item() / len(self.dataloaders['train']), completed_steps)
            tb_writer.add_scalar('eval_loss', total_loss_eval / len(self.dataloaders['eval']), completed_steps)
            for key, value in result.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, completed_steps)
        self.tokenizer.save_pretrained(self.output_dir)
        self.model.save_pretrained(self.output_dir)
        print("Save model success")

    def evaluate(self):
        eval_loss = 0
        for step, batch in enumerate(self.dataloaders['eval']):
            with torch.no_grad():
                outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                     labels=batch["labels"])
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
                generated_tokens = batch["input_ids"].cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                if self.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
                self.metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        result = self.metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        return eval_loss, result


    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels





