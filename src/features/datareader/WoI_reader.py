import re
import json
import random
# import sys
# sys.path.append("./")
from src.features.datareader.data_reader import DataReader
from src.config import data_config


class WOIReader(DataReader):
    def __call__(self, *args, **kwargs):
        self.load_data()
        self.get_utterance()
        self.define_input()

    def load_data(self):
        """
        This function is to read the data file (.json, .csv, ...)
        :return: the list of dictionaries or the dictionary of samples in the dataset
        """
        with open(self.data_path, encoding='utf-8') as json_file:
            self.data = [json.loads(json_str) for json_str in json_file]

    def get_utterance(self):
        """
        This function is to get utterance (<=5 utterance)
        :return: the list of list, each list contain utterances for each Input
                EX: [[utterance1, utterance2, utterance3, ...], [utterance4, utterance5, utterance6, ...]]
        """
        for dialogues in self.data:
            for key in dialogues:
                list_utter_full = dialogues[key]['dialog_history']
                list_turns = []
                query_key = None
                for turn in list_utter_full:
                    if 'SearchAgent' not in turn['action']:
                        if query_key is not None:
                            turn.__setitem__("query_key", query_key)
                            query_key = None
                        list_turns.append(turn)
                    elif turn['text'] != '':
                        query_key = turn['text']
                    len_turns = len(list_turns)

            # Set the starting index for each sub-list based on the first turn of the conversation
            if list_turns[0]['action'] == 'Apprentice => Wizard':
                # If the first turn is from the Apprentice, set the starting index options to 3, 5, and 7
                idx_turns = [3, 5, 7]
            else:
                # If the first turn is from the Wizard, set the starting index options to 2, 4, and 6
                idx_turns = [2, 4, 6]

            # Iterate over each possible starting index for a sub-list
            for idx_turn in idx_turns:
                # Check if the resulting sub-list is within the bounds of the input list
                if idx_turn <= len_turns:
                    # Append the sub-list to the list of utterances, ensuring it includes the max_elements turns preceding the starting index
                    start_idx = max(0, idx_turn - self.context_window)
                    end_idx = idx_turn + 1
                    self.list_utter.append(list_turns[start_idx:end_idx])


    def define_instruction(self,child_dialogue):
        """
        This function is to define the input and label for module state prediction
        :param child_dialogue: dialogue history for module 1
        :return: dictionary of input include two keys:
                - prompt: instruction
                - output: label
        """
        # Define instruction
        list_instruction = [data_config.INSTRUCTION1, data_config.INSTRUCTION2, data_config.INSTRUCTION3,
                            data_config.INSTRUCTION4, data_config.INSTRUCTION5, data_config.INSTRUCTION6,
                            data_config.INSTRUCTION7, data_config.INSTRUCTION8, data_config.INSTRUCTION9,
                            data_config.INSTRUCTION10, data_config.INSTRUCTION11, data_config.INSTRUCTION12,
                            data_config.INSTRUCTION13, data_config.INSTRUCTION14, data_config.INSTRUCTION15,
                            data_config.INSTRUCTION16]
        instruction = random.choice(list_instruction)
        # Define input
        dict_input = dict()
        list_turn = []
        regex_pattern = r"\s{2,}"
        for utter in child_dialogue:
            utter['text'] = re.sub(regex_pattern, " ", utter['text'])
            if utter['action'] == "Apprentice => Wizard":
                list_turn.append(data_config.USER_SEP + utter['text'] + data_config.EOT_SEP)
            elif utter['action'] == "Wizard => Apprentice":
                list_turn.append(data_config.SYSTEM_SEP + utter['text'] + data_config.EOT_SEP)

        last_system_utter = child_dialogue[-1]
        domain = ''
        domain_slots = ''
        dict_input['prompt'] = instruction.replace("<DIALOGUE_CONTEXT>",
                                                   ''.join([turn for turn in list_turn[:len(list_turn) - 1]])).replace('<DOMAIN>', domain).replace('<SLOT>',domain_slots)
        nowhite = ' '.join(dict_input['prompt'].split())
        dict_input['prompt'] = nowhite
        # Define label
        dict_label = dict()

        if 'query_key' not in last_system_utter:
            dict_input['output'] = "Chitchat: None; general"
        else:
            dict_label['Seek'] = last_system_utter['query_key']
            dict_input['output'] = "Seek: " + last_system_utter['query_key'] + '; general'

        return dict_input

    def define_input(self):
        """
        This function is to define the input for the model State Prediction
        :return: list of dictionaries with two keys:
                - 'prompt': the input
                - 'output': the label
                EX: [{'output': ******, 'prompt': *******}, {'output': ******, 'prompt': *******}, ...]
        """
        lines_wrote = 0
        with open(self.sample_path, 'w', encoding='utf-8') as f:
            for child_dialogue in self.list_utter:
                if len(child_dialogue) <= 2 or child_dialogue[-1]['action'] != 'Wizard => Apprentice':
                    continue
                lines_wrote+=1
                dict_input = self.define_instruction(child_dialogue)
                json.dump(dict_input, f)
                f.write("\n")
            print('Number of lines wrote: ',lines_wrote)
