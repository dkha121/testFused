from typing import Optional, Dict, List, Union
import json
import datasets
import torch
from os.path import join
from datasets import DatasetDict, load_dataset, Dataset, concatenate_datasets
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorForSeq2Seq
from accelerate import Accelerator


class StateDataloader():
    def __init__(self,
                 model: AutoModel,
                 tokenizer: AutoTokenizer,

                 dataset_name: str,
                 text_column: str,
                 target_column: str,
                 train_file: Union[str, List[str]],
                 val_file: Optional[Union[str, List[str]]],
                 test_file: Optional[Union[str, List[str]]],

                 do_train: bool = False,
                 do_eval: bool = False,
                 do_predict: bool = False,

                 max_len_instruction: int = 512,
                 max_len_response: int = 80,

                 dynamic_batch_collate: bool = False,
                 train_batch_size: int = 8,
                 val_batch_size: int = 4,

                 seed: int = 42,
                 preprocessing_num_workers: int = 2,
                 ignore_pad_token_for_loss: bool = True,
                 dataset_config_name: Optional[str] = None,
                 new_special_tokens: Optional[Dict[str, str]] = None,
                 max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None,
                 max_predict_samples: Optional[int] = None
                 ) -> None:

        self.model = model
        self.tokenizer = tokenizer

        self.dataset_name = dataset_name
        self.text_column = text_column
        self.target_column = target_column
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        self.do_train = do_train
        self.do_eval = do_eval
        self.do_predict = do_predict

        self.max_len_instruction = max_len_instruction
        self.max_len_response = max_len_response

        self.padding = 'longest'

        self.dynamic_batch_collate = dynamic_batch_collate
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.seed = seed
        self.preprocessing_num_workers = preprocessing_num_workers
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.dataset_config_name = dataset_config_name
        self.new_special_tokens = new_special_tokens

        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.accelerator = Accelerator()

    def __call__(self, *args, **kwargs):
        dataloaders = {}
        if not self.do_train and not self.do_eval and not self.do_predict:
            print("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
            return

        if self.train_file is not None:
            if isinstance(self.train_file, str):
                print('\nLoading train dataset' + '.' * 10)
                self.train_dataset = self.load_data('train', self.train_file)
            else:
                print('\nLoading mutiple train datasets' + '.' * 10)
                self.train_dataset = self.load_data('train', self.train_file, multiple=True)
        if self.val_file is not None:
            if isinstance(self.val_file, str):
                print('\nLoading validation dataset' + '.' * 10)
                self.eval_dataset = self.load_data('val', self.val_file)
            else:
                print('\nLoading mutiple validation datasets' + '.' * 10)
                self.eval_dataset = self.load_data('val', self.val_file, multiple=True)
        if self.test_file is not None:
            if isinstance(self.test_file, str):
                print('\nLoading test dataset' + '.' * 10)
                self.test_dataset = self.load_data('test', self.test_file)
            else:
                print('\nLoading mutiple test datasets' + '.' * 10)
                self.test_dataset = self.load_data('test', self.test_file, multiple=True)

        if self.do_train and self.train_dataset is not None:
            if self.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(self.max_train_samples))
            if not self.dynamic_batch_collate:
                self.train_dataset = self.preprocess_data( self.train_dataset,
                                                          desc="train dataloader map pre-processing")
            dataloaders['train'] = self.get_dataloader(self.train_dataset, types_train=True,
                                                       dynamic_batch_collate=self.dynamic_batch_collate)

        if self.do_eval and self.eval_dataset is not None:
            if self.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(self.max_eval_samples))
            if not self.dynamic_batch_collate:
                self.eval_dataset = self.preprocess_data( self.eval_dataset,
                                                         desc="val dataloader map pre-processing")
            dataloaders['eval'] = self.get_dataloader(self.eval_dataset,
                                                      dynamic_batch_collate=self.dynamic_batch_collate)

        if self.do_predict and self.test_dataset is not None:
            if self.max_predict_samples is not None:
                self.test_dataset = self.test_dataset.select(range(self.max_predict_samples))
            self.test_dataset = self.preprocess_data( self.test_dataset,
                                                     desc="test dataloader map pre-processing")

        return dataloaders

    def load_data(self, key: str, data_file: Union[str, List[str]], multiple: bool = False) -> DatasetDict:
        """
        Loads a dataset from a file on disk and returns it as a dictionary of Dataset objects.

        Args:
            key (str): The key to assign to the loaded dataset in the returned dictionary of Dataset objects.
            data_file (Union[str, List[str]]): The path or paths to the data file(s) to load. If multiple is True, data_file
                                                should be a list of file paths. Otherwise, it should be a single file path.
            mutiple (bool): A flag that indicates whether the data_file argument is a list of multiple file paths.

        Returns:
            A dictionary of Dataset objects that represents the loaded dataset. If mutiple is True, the function
            concatenates the datasets from the multiple files before returning them. Otherwise, it returns a single
            dataset loaded from the data_file path.
        """

        if multiple:
            dataset_list = []
            for file in data_file:
                data_files = {key: file}
                extension = file.split(".")[-1]
                dataset_list.append(load_dataset(extension, data_files=data_files, split=key))
            dataset = concatenate_datasets(dataset_list)
            dataset.shuffle(self.seed)
            return dataset

        data_files = {key: data_file}
        extension = data_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files, split=key)
        return dataset

    def preprocess_function(self, raw_dataset_examples: DatasetDict) -> DatasetDict:
        """
        Preprocesses the raw dataset by extracting inputs and targets, tokenizing them using the tokenizer,
        and returning a dictionary containing the tokenized inputs and labels.
        :param raw_dataset_examples (DatasetDict): A dictionary containing the raw dataset.

        :return:
            DatasetDict: A dictionary containing the tokenized inputs and labels.
        """

        inputs = []
        targets = []

        for i in range(len(raw_dataset_examples[self.text_column])):
            if raw_dataset_examples[self.text_column][i] is not None and raw_dataset_examples[self.target_column][
                i] is not None:
                inputs.append(raw_dataset_examples[self.text_column][i])
                targets.append(raw_dataset_examples[self.target_column][i])  # + tokenizer.eos_token)

        model_inputs = self.tokenizer(inputs, padding=self.padding, truncation=True)
        labels = self.tokenizer(text_target=targets, padding=self.padding, truncation=True)

        if self.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in
                                   labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_data(self, raw_dataset: DatasetDict, desc) -> Dataset:
        """
        This function preprocesses the raw dataset of the whole dataset using the tokenizer.
        :param raw_dataset: Dictionary of raw datasets.
        :param desc: Description of the dataset being processed.
        :return: Preprocessed dataset as a Hugging Face Dataset object.
        """
        with self.accelerator.main_process_first():
            dataset = raw_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=raw_dataset.column_names,
                desc=desc,
            )

        return dataset

    def dynamic_collate(self, batch):
        """
        A collate function that tokenizes the inputs and targets, and applies dynamic padding and truncation
        based on the maximum length in the batch.
        """
        inputs = [example[self.text_column] for example in batch]
        targets = [example[self.target_column] for example in batch]

        # Get the maximum length in the batch
        max_len = max(len(self.tokenizer.encode(input)) + len(self.tokenizer.encode(target)) for input, target in
                      zip(inputs, targets))
        max_len = min(max_len, self.max_len_instruction) if self.max_len_instruction is not None else max_len

        # Tokenize the inputs and targets
        tokenized_inputs = self.tokenizer(inputs, max_length=max_len, padding="max_length", truncation=True)
        tokenized_targets = self.tokenizer(targets, max_length=max_len, padding="max_length", truncation=True)

        # Create the attention masks
        attention_masks = [[int(token_id != self.tokenizer.pad_token_id) for token_id in input_ids] for input_ids in
                           tokenized_inputs["input_ids"]]

        # Create the PyTorch tensors
        input_ids = torch.tensor(tokenized_inputs["input_ids"], dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        target_ids = torch.tensor(tokenized_targets["input_ids"], dtype=torch.long)
        target_ids = torch.where(target_ids == self.tokenizer.pad_token_id, -100 * torch.ones_like(target_ids),
                                 target_ids)

        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": target_ids}

    def get_dataloader(self, dataset: datasets.arrow_dataset.Dataset, types_train: bool = False,
                       dynamic_batch_collate: bool = False) -> DataLoader:
        """
        Returns a dataloader for the given dataset using the specified batch size and collator.
        The dataloader returned is for either training or validation depending on the value of types_train.
        :param dataset (datasets.arrow_dataset.Dataset): The dataset to create a dataloader for.
        :param types_train (bool): A boolean indicating whether the returned dataloader is for
                                    training or validation.
        :return:
            DataLoader: A dataloader for the given dataset using the specified batch size and collator.
        """
        dataloader: DataLoader
        label_pad_token_id = -100 if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = None
        if dynamic_batch_collate:
            data_collator = self.dynamic_collate
        else:
            data_collator = DataCollatorForSeq2Seq(tokenizer,
                                                   model=self.model,
                                                   label_pad_token_id=label_pad_token_id,
                                                   pad_to_multiple_of=None,
                                                   padding=self.padding
                                                   )
        if types_train:
            dataloader = DataLoader(dataset,
                                    collate_fn=data_collator,
                                    batch_size=self.train_batch_size,
                                    shuffle=True)
        else:
            train_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset,
                                    collate_fn=data_collator,
                                    batch_size=self.val_batch_size,
                                    sampler=train_sampler,
                                    shuffle=False)

        return dataloader
