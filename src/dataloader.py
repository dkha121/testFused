from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq


class DataLoader:
    def __init__(self,
                 config,
                 tokenizer,
                 model,
                 pad_to_max_length=False,
                 max_train_samples=200,
                 max_source_length=768,
                 max_target_length=30,
                 ignore_pad_token_for_loss=True
                 ):
        self.pad_to_max_length = pad_to_max_length
        self.max_train_samples = max_train_samples
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.padding = "max_length" if self.pad_to_max_length else 'longest'

        self.tokenizer = tokenizer
        self.config = config
        self.model = model

        self.text_column = None
        self.target_column = None

    def preprocess_function(self, examples):
        """
        This function is preprocessing each sample in the dataset
        :param examples: one input/sample
        :return: the input after tokenized
        """
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[self.text_column])):
            if examples[self.text_column][i] is not None and examples[self.target_column][i] is not None:
                inputs.append(examples[self.text_column][i])
                targets.append(examples[self.target_column][i])  # + tokenizer.eos_token)

        inputs = [inp for inp in inputs]
        selinptus = []
        for i, inp in enumerate(inputs):
            selinptus.append(inp)
        model_inputs = self.tokenizer(selinptus, max_length=self.max_source_length,
                                      padding=self.padding, truncation=True)

        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True)

        if self.padding == "max_length" or self.padding == 'longest' and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(lb if lb != self.tokenizer.pad_token_id else -100) for lb in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_data(self, data_args, model_args, training_args):
        """
        This function is to preprocess all samples in the dataset
        :param data_args: data arguments
        :param model_args: model arguments
        :param training_args: training arguments
        :return:
                - train_dataset: dataset for training
                - eval_dataset: dataset for evaluating
                - test_dataset: dataset for testing
                - data_collator: data collator for
        """
        train_dataset = None
        eval_dataset = None
        test_dataset = None
        if data_args.train_file is not None:
            print('\nLoading train dataset' + '.' * 10)
            data_files = {'train': data_args.train_file}
            extension = data_args.train_file.split(".")[-1]
            train_dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir,
                                         split='train')

        if data_args.validation_file is not None:
            print('\nLoading validation dataset' + '.' * 10)
            data_files = {'train': data_args.validation_file}
            extension = data_args.validation_file.split(".")[-1]
            eval_dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, split='train')

        if data_args.test_file is not None:
            print('\nLoading test dataset' + '.' * 10)
            data_files = {'train': data_args.test_file}
            extension = data_args.test_file.split(".")[-1]
            test_dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, split='train')

        if training_args.do_train:
            column_names = train_dataset.column_names
        elif training_args.do_eval:
            column_names = eval_dataset.column_names
        elif training_args.do_predict:
            column_names = test_dataset.column_names
        else:
            print("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
            return
        print(train_dataset.column_names, eval_dataset.column_names)
        # Get the column names for input/target.
        if data_args.text_column is None:
            self.text_column = column_names[0]
        else:
            self.text_column = data_args.text_column

        if data_args.target_column is None:
            self.target_column = column_names[1]
        else:
            self.target_column = data_args.target_column

        if self.max_train_samples is not None:
            train_dataset = train_dataset.select(range(self.max_train_samples))

        if training_args.do_train:
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))
            with training_args.main_process_first(desc="train dataloader map pre-processing"):
                train_dataset = train_dataset.map(
                    self.preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=train_dataset.column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataloader",
                )

        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
            with training_args.main_process_first(desc="validation dataloader map pre-processing"):
                eval_dataset = eval_dataset.map(
                    self.preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=eval_dataset.column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataloader",
                )

        if training_args.do_predict:
            if data_args.max_predict_samples is not None:
                test_dataset = test_dataset.select(range(data_args.max_predict_samples))
            with training_args.main_process_first(desc="prediction dataloader map pre-processing"):
                test_dataset = test_dataset.map(
                    self.preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=test_dataset.column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataloader",
                )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        return train_dataset, eval_dataset, test_dataset, data_collator
