import numpy as np
import torch
import evaluate
import nltk
nltk.download('punkt')
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from accelerate.utils import DistributedType

class Evaluation:
    def __init__(self,
                 metric,
                 eval_dataloaders: DataLoader,
                 pad_to_max_length: bool = True,
                 ignore_pad_token_for_loss: bool = True,
                 with_tracking: bool = False,
                 num_beams: Optional[int] = 4,
                 max_target_length: Optional[int] = 40
                 ):

        self.eval_dataloaders = eval_dataloaders
        self.pad_to_max_length = pad_to_max_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.metric = metric
        self.with_tracking = with_tracking
        self.num_beams = num_beams
        self.max_target_length = max_target_length


    def eval(self,accelerator,tokenizer,model):
        model.eval()
        gen_kwargs = {
            "max_length": self.max_target_length,
            "num_beams": self.num_beams,
        }
        total_loss_eval = 0
        samples_seen = 0
        for step, batch in enumerate(self.eval_dataloaders):
            if samples_seen == 0 and accelerator.distributed_type == DistributedType.FSDP:
                model(**batch)
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not self.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1,
                                                              pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if self.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(self.eval_dataloaders) - 1:
                        decoded_preds = decoded_preds[: len(self.eval_dataloaders.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(self.eval_dataloaders.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)

                self.metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

                # Compute and log the loss
                outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                labels=batch["labels"])
                loss = outputs.loss
                if self.with_tracking:
                    total_loss_eval += loss.detach().float()
        result = self.metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        if self.with_tracking:
            return result, total_loss_eval
        return result


    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels