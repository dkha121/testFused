import os


class Evaluation:
    def __init__(self,
                 trainer,
                 eval_dataset,
                 test_dataset,
                 tokenizer,
                 max_length,
                 val_max_target_length,
                 num_beams,
                 max_eval_samples,
                 max_predict_samples,
                 predict_with_generate,
                 output_dir):

        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.val_max_target_length = val_max_target_length
        self.num_beams = num_beams
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples
        self.predict_with_generate = predict_with_generate
        self.output_dir = output_dir

    def eval(self):
        """
        This function is to evalidate in the val dataset
        :return:
        """

        metrics = self.trainer.evaluate(max_length=self.max_length, num_beams=self.num_beams,
                                        metric_key_prefix="eval")
        max_eval_samples = self.max_eval_samples \
            if self.max_eval_samples is not None \
            else len(self.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def predict(self):
        """
        This function is to predict in the test dataset
        :return:
        """
        predict_results = self.trainer.predict(
            self.test_dataset, metric_key_prefix="predict", max_length=self.max_length, num_beams=self.num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = self.max_predict_samples if self.max_predict_samples is not None else len(self.test_dataset)

        metrics["predict_samples"] = min(max_predict_samples, len(self.test_dataset))

        self.trainer.log_metrics("predict", metrics)
        self.trainer.save_metrics("predict", metrics)

        if self.trainer.is_world_process_zero():
            if self.predict_with_generate:
                predictions = self.tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(self.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
