

class Trainer:
    def __init__(self,
                 last_checkpoint,
                 trainer,
                 resume_from_checkpoint,
                 train_dataset,
                 max_train_samples
                 ):
        self.last_checkpoint = last_checkpoint
        self.trainer = trainer
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.max_train_samples = max_train_samples

    def train(self):
        # Training
        checkpoint = None
        if self.resume_from_checkpoint is not None:
            checkpoint = self.resume_from_checkpoint
        elif self.last_checkpoint is not None:
            checkpoint = self.last_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = self.max_train_samples if self.max_train_samples is not None else len(self.train_dataset)

        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
