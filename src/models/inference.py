from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

path_to_tokenizerjson = r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\BaseLineV1\state_prediction\dev\output\tokenizer.json'
path_to_save_dir = r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\BaseLineV1\state_prediction\dev\output'


# Load the tokenizer from the tokenizer.json file
tokenizer = PreTrainedTokenizerFast(tokenizer_file=path_to_tokenizerjson)


# Load the model from the model.bin file
model = BartForConditionalGeneration.from_pretrained(path_to_save_dir)

while True:
    sentence = input()
    if sentence == 'exit':
        break
    input_tokens = tokenizer(sentence, return_tensors="pt")
    output = model.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"])

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
