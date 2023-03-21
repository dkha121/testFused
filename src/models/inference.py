from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
path_to_tokenizerjson = ''
path_to_save_dir = ''


# Load the tokenizer from the tokenizer.json file
tokenizer = PreTrainedTokenizerFast(tokenizer_file=path_to_tokenizerjson)


# Load the model from the model.bin file
model = BartForConditionalGeneration.from_pretrained(path_to_save_dir)

sentence = "Instruction: You must be given the type of belief state between specified people or speakers base on this dialogue [CTX]SYSTEM: Okay! What cuisine would you like to try?[EOT]. USER: I've got a couple of Jamaican buddies and I want to make them feel at home.[EOT]. [EOD]. [OPT] user_action: affirm_intent, bye, general, inform, negate_intent, select, inform_intent, request, thank, request_alts, negate, greet, affirm; 1. Seek, 2. Chitchat, 3. Database: Restaurant; Slots: time,has_vegetarian_options,Postcode,has_live_music,count,address,price_range,ref,number_of_seats,serves_alcohol,choice,restaurant_name,area,cuisine,has_seating_outdoors,city,intent,phone_number,rating,date [Q] What is the belief state? "
input_tokens  = tokenizer(sentence, return_tensors="pt")
output = model.generate(input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"])

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)