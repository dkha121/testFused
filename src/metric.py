from datasets import load_metric
import nltk
import numpy as np
import json
import os


# Metric
metrics = {
    'rouge': load_metric("rouge"),
    'accuracy': load_metric("accuracy"),
    'f1': load_metric("f1"),
    'recall': load_metric("recall"),
    'precision': load_metric("precision"),
}


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, ignore_pad_token_for_loss=True, input_ids=None, global_step=0):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # input_ids = input_ids.cpu()
    input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
    decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    # print(decoded_preds, decoded_labels)
    class_decoded_labels, class_decoded_preds = [], []
    for i in range(len(decoded_labels)):
        if len(decoded_labels[i]) < 30 and decoded_labels[i] != 'not present':
            class_decoded_labels.append(decoded_labels[i])
            class_decoded_preds.append(decoded_preds[i])
    results = {}
    for metric_name, metric in metrics.items():
        if 'rouge' in metric_name:
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        elif metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if len(class_decoded_labels) > 0:
                classes_names = set(class_decoded_preds + class_decoded_labels)
                classes_vocab = {k: v for v, k in enumerate(classes_names)}
                class_decoded_preds = [classes_vocab[x] for x in class_decoded_preds]
                class_decoded_labels = [classes_vocab[x] for x in class_decoded_labels]
                if metric_name == 'accuracy':
                    result = metric.compute(predictions=class_decoded_preds, references=class_decoded_labels)
                else:
                    result = metric.compute(predictions=class_decoded_preds, references=class_decoded_labels,
                                            average="micro")

                # Extract a few results from ROUGE
                result = {key: value * 100 for key, value in result.items()}
            # print(metric_name, result)
        else:
            result = {metric_name: metric}
        results.update(result)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    results["gen_len"] = np.mean(prediction_lens)
    results = {k: round(v, 4) for k, v in results.items()}

    # # write result
    # output_prediction_file = os.path.join(output_dir, f"intermediate_result_{global_step}.jsonl")
    # with open(output_prediction_file, "a") as writer:
    #     for inputs, preds, labels in zip(decoded_inputs, decoded_preds, decoded_labels):
    #         output_str = json.dumps({'input_ids': inputs, 'preds': preds, 'labels': labels})
    #         writer.write(output_str + '\n')
    #
    # output_prediction_file = os.path.join(output_dir, f"intermediate_metrics_{global_step}.json")
    # with open(output_prediction_file, "a") as writer:
    #     json.dump(results, writer)

    return results
