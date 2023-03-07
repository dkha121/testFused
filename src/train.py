import os
import json


class Trainer:
    def __init__(self,
                 training_args,
                 data_args,
                 model_args,
                 last_checkpoint,
                 trainer,
                 train_dataset):
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.last_checkpoint = last_checkpoint
        self.trainer = trainer
        self.train_dataset = train_dataset

    def train(self):
        # Training
        if self.training_args.do_train:
            output_args_file = os.path.join(self.training_args.output_dir, f"modelargs.json")
            os.makedirs(self.training_args.output_dir, exist_ok=True)
            with open(output_args_file, "w+") as writer:
                all_argsdict = {**self.model_args.__dict__, **self.data_args.__dict__, **self.training_args.to_dict()}
                json.dump(all_argsdict, writer)

            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            self.trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples
                if self.data_args.max_train_samples is not None
                else len(self.train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
