import sys
from src.data.dataloader import StateDataloader
from training_loop import Trainer
import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, help="Whether to overwrite the model's directory.")

    parser.add_argument('--train_files', nargs='+',default=None, help="Directory to train file")

    parser.add_argument('--val_files', nargs='+', default=None, help="Directory to validation file")

    parser.add_argument('--test_files', nargs='+', default=None, help="Directory to test file")

    parser.add_argument('--model_name', type=str, default="lucadiliello/bart-small", help ="model name")

    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for the dataloader")

    parser.add_argument('--max_train_samples', type=int, default=None, help="Number of train samples")

    parser.add_argument('--max_eval_samples', type=int, default=None, help="Number of validation samples")

    parser.add_argument('--seed', type=int, default=42, help="A seed for reproducible training.")

    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],)

    parser.add_argument('--report_to', type=str, default='wandb',help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ))

    parser.add_argument('--num_train_epochs', type=int, default=10, help="number training epochs")

    parser.add_argument('--max_target_length', type=int, default=60, help="max length labels tokenize")

    parser.add_argument('--num_beams', type=int, default=4, help="number of beams")

    parser.add_argument('--weight_decay', type=float, default=0.3,  help="Weight decay to use.")

    parser.add_argument('--mixed_precision', type=str, default='fp16', help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,  help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--with_tracking', type=bool, default=True, help="Whether to enable experiment trackers for logging.")

    parser.add_argument('--do_train', type=bool, default=True, help="Whether to run training.")

    parser.add_argument('--do_eval', type=bool, default=True, help="Whether to run evaluate.")

    parser.add_argument('--do_predict', type=bool, default=False, help="Whether to run predict.")

    parser.add_argument('--text_column', type=str, default='prompt', help="The name of the column in the datasets containing the full texts (for summarization).")

    parser.add_argument('--target_column', type=str, default='output', help="The name of the column in the label containing the full texts (for summarization).")

    parser.add_argument('--checkpointing_steps', type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")


    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")


    args = parser.parse_args(args)

    return args


def main(args):

    args = parse_args(args)


    dataloader_args = {
        "model_name": args.model_name,
        "text_column": args.text_column,
        "target_column": args.target_column,
        "train_file": args.train_files,
        "do_train": args.do_train,
        "val_file": args.val_files,
        "do_eval": args.do_eval,
        "test_file": args.test_files,
        "do_predict": args.do_predict,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "seed": args.seed
    }

    dataloaders = StateDataloader(**dataloader_args)


    trainer_args = {
        "model_name_or_path": args.model_name,
        "output_dir": args.output_dir,
        "dataloaders": dataloaders,
        "lr_scheduler_type": args.lr_scheduler_type,
        "checkpointing_steps": args.checkpointing_steps,
        "seed": args.seed,
        "with_tracking": args.with_tracking,
        "report_to": args.report_to,
        "num_train_epochs": args.num_train_epochs,
        "num_beams": args.num_beams,
        "weight_decay": args.weight_decay,
        "mixed_precision": args.mixed_precision,
        "per_device_train_batch_size":dataloaders.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate
    }

    trainer = Trainer(**trainer_args)

    trainer.train()

if __name__ == "__main__":
    main(sys.argv[1:])