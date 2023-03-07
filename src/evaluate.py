import os


class Evaluation:
    def __init__(self, data_args, training_args, trainer, eval_dataset, test_dataset, tokenizer, logger):
        self.data_args = data_args
        self.training_args = training_args
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.logger = logger

        self.max_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )
        self.num_beams = self.data_args.num_beams \
            if self.data_args.num_beams is not None \
            else self.training_args.generation_num_beams

    def eval(self):
        """
        This function is to evalidate in the val dataset
        :return:
        """
        if self.training_args.do_eval:
            self.logger.info("*** Evaluate ***")
            metrics = self.trainer.evaluate(max_length=self.max_length, num_beams=self.num_beams,
                                            metric_key_prefix="eval")
            max_eval_samples = self.data_args.max_eval_samples \
                if self.data_args.max_eval_samples is not None \
                else len(self.eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)

    def predict(self):
        """
        This function is to predict in the test dataset
        :return:
        """
        if self.training_args.do_predict:
            self.logger.info("*** Predict ***")

            predict_results = self.trainer.predict(
                self.test_dataset, metric_key_prefix="predict", max_length=self.max_length, num_beams=self.num_beams
            )
            metrics = predict_results.metrics
            max_predict_samples = (
                self.data_args.max_predict_samples
                if self.data_args.max_predict_samples is not None
                else len(self.test_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(self.test_dataset))

            self.trainer.log_metrics("predict", metrics)
            self.trainer.save_metrics("predict", metrics)

            if self.trainer.is_world_process_zero():
                if self.training_args.predict_with_generate:
                    predictions = self.tokenizer.batch_decode(
                        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]
                    output_prediction_file = os.path.join(self.training_args.output_dir, "generated_predictions.txt")
                    with open(output_prediction_file, "w") as writer:
                        writer.write("\n".join(predictions))
