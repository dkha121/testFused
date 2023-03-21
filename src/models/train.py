from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForMaskedLM
)
from src.data.dataloader import StateDataloader
from training_loop import Trainer


def main():

    output_dir = '/kaggle/working/'
    train_files = [r'/kaggle/input/fusedchatconverted/train_sample.json',r'/kaggle/input/woisamplejson/train_converted - Copy.json']
    val_files = [r'/kaggle/input/woisamplejson/valid_converted - Copy.json',r'/kaggle/input/fusedchatconverted/valid_sample.json']

    model_name = "lucadiliello/bart-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloader_args = {
        "model": AutoModel.from_pretrained(model_name, return_dict=False),
        "tokenizer": tokenizer,
        "dataset_name": "KETOD",
        "text_column": 'prompt',
        "target_column": 'output',
        "train_file": train_files,
        "do_train": True,
        "val_file": val_files,
        "do_eval": True,
        "test_file": None,
        "train_batch_size": 9,
        "val_batch_size": 9,
        "dynamic_batch_collate": True,
        "max_train_samples": 5000,
        "max_eval_samples": 300,
        "seed": 42
    }

    dataloaders = StateDataloader(**dataloader_args)


    trainer_args = {
        "model_name_or_path": model_name,
        "output_dir": output_dir,
        "dataloaders": dataloaders,
        "lr_scheduler_type": 'linear',
        "seed": 42,
        "with_tracking": True,
        "report_to": "wandb",
        "num_train_epochs": 25,
        "val_max_target_length": 80,
        "num_beams": 4,
        "weight_decay": 0.3
    }

    trainer = Trainer(**trainer_args)

    trainer.train()


if __name__ == "__main__":
    main()