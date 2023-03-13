from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq


class DataLoader:
    def __init__(self,
                 training_args,
                 train_file,
                 validation_file,
                 test_file,
                 cache_dir,
                 tokenizer,
                 model,
                 do_train,
                 do_eval,
                 do_predict,
                 pad_to_max_length,
                 max_train_samples,
                 max_eval_samples,
                 max_predict_samples,
                 max_source_length,
                 max_target_length,
                 ignore_pad_token_for_loss,
                 text_column,
                 target_column,
                 preprocessing_num_workers,
                 overwrite_cache
                 ):

        self.train_args = training_args

        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file

        self.cache_dir = cache_dir

        self.do_train = do_train
        self.do_eval = do_eval
        self.do_predict = do_predict

        self.pad_to_max_length = pad_to_max_length
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.padding = "max_length" if self.pad_to_max_length else 'longest'
        self.tokenizer = tokenizer

        self.model = model

        self.text_column = text_column
        self.target_column = target_column

        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def __call__(self, *args, **kwargs):
        if self.train_file is not None:
            print('\nLoading train dataset' + '.' * 10)
            self.train_dataset = self.load_data(self.train_file, self.cache_dir)

        if self.validation_file is not None:
            print('\nLoading validation dataset' + '.' * 10)
            self.eval_dataset = self.load_data(self.validation_file, self.cache_dir)

        if self.test_file is not None:
            print('\nLoading test dataset' + '.' * 10)
            self.test_dataset = self.load_data(self.test_file, self.cache_dir)

        if self.do_train:
            column_names = self.train_dataset.column_names
        elif self.do_eval:
            column_names = self.eval_dataset.column_names
        elif self.do_predict:
            column_names = self.test_dataset.column_names
        else:
            print("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
            return

        # Get the column names for input/target.
        self.text_column = self.text_column
        self.target_column = self.target_column

        if self.do_train:
            if self.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(self.max_train_samples))
            self.train_dataset = self.preprocess_data(self.train_args, self.train_dataset,
                                                      desc="train dataloader map pre-processing")

        if self.do_eval:
            if self.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(self.max_eval_samples))
            self.eval_dataset = self.preprocess_data(self.train_args, self.eval_dataset,
                                                      desc="val dataloader map pre-processing")

        if self.do_predict:
            if self.max_predict_samples is not None:
                self.test_dataset = self.test_dataset.select(range(self.max_predict_samples))
            self.test_dataset = self.preprocess_data(self.train_args, self.test_dataset,
                                                     desc="test dataloader map pre-processing")

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        return data_collator

    def load_data(self, data_file, cache_dir):
        data_files = {'train': data_file}
        extension = data_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files, cache_dir=cache_dir, split='train')
        return dataset

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

        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length,
                                      padding=self.padding, truncation=True)

        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length,
                                    padding=self.padding, truncation=True)

        if self.padding == "max_length" or self.padding == 'longest' and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(lb if lb != self.tokenizer.pad_token_id else -100) for lb in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_data(self, training_args, dataset, desc):
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

        with training_args.main_process_first(desc=desc):
            dataset = dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=dataset.column_names,
                load_from_cache_file=not self.overwrite_cache,
                desc="Running tokenizer on train dataloader",
            )

        return dataset
