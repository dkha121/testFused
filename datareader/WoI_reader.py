
import json
import random
import sys
import pandas as pd
from data_reader import DataReader

sys.path.append("./")


import data_config


def define_domain(service):
    """
    This function is to define group domain base on service of the utterance
    :param service: service of the utterance
    :return: domain of the utterance
    """
    domain = None
    return domain


def define_instruction(child_dialogue):
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
                        data_config.INSTRUCTION10, data_config.INSTRUCTION11]
    instruction = random.choice(list_instruction)
    # Define input
    dict_input = dict()
    list_turn = []
    for utter in child_dialogue:
        if utter['action'] == "Apprentice => Wizard":
            list_turn.append(data_config.USER_SEP + utter['text'] + data_config.EOT_SEP)
        elif utter['action'] == "Wizard => Apprentice":
            list_turn.append(data_config.SYSTEM_SEP + utter['text'] + data_config.EOT_SEP)

    # frame = child_dialogue[-1]['frames'][0]
    # service = frame['service']
    # domain = define_domain(service)

    last_system_utter = child_dialogue[-1]
    domain = data_config.DOMAIN_BUS
    dict_input['prompt'] = instruction.replace("<DIALOGUE_CONTEXT>",
                                               ''.join([turn for turn in list_turn[:len(list_turn)-1]])).replace('<DOMAIN>', domain)

    # Define label
    list_action = []
    dict_label = dict()

    if 'query_key' not in last_system_utter:
        dict_input['output'] = "Chitchat: None"
    else:
        dict_label['Seek'] = last_system_utter['query_key']
        dict_input['output'] = "Seek: " + last_system_utter['query_key'] +'; general'


    return dict_input


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
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            self.data.append(result)

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

            idx_turn = 0
            if list_turns[0]['action'] == 'Apprentice => Wizard':
                while idx_turn <= len_turns:
                    if idx_turn % 2 == 0:
                        self.list_utter.append(list_turns[idx_turn:idx_turn + 2])
                        if idx_turn + 4 <= len_turns:
                            self.list_utter.append(list_turns[idx_turn:idx_turn + 4])
                        if idx_turn + 6 <= len_turns:
                            self.list_utter.append(list_turns[idx_turn:idx_turn + 6])
                    else:
                        if idx_turn + 3 <= len_turns:
                            self.list_utter.append(list_turns[idx_turn:idx_turn + 3])
                        if idx_turn + 5 <= len_turns:
                            self.list_utter.append(list_turns[idx_turn:idx_turn + 5])
                    idx_turn += 1
            else:
                while idx_turn <= len_turns:
                    if idx_turn % 2 == 0:
                        self.list_utter.append(list_turns[idx_turn:idx_turn + 1])
                        if idx_turn + 3 <= len_turns:
                            self.list_utter.append(list_turns[idx_turn:idx_turn + 3])
                        if idx_turn + 5 <= len_turns:
                            self.list_utter.append(list_turns[idx_turn:idx_turn + 5])
                    else:
                        if idx_turn + 2 <= len_turns:
                            self.list_utter.append(list_turns[idx_turn:idx_turn + 2])
                        if idx_turn + 4 <= len_turns:
                            self.list_utter.append(list_turns[idx_turn:idx_turn + 4])
                    idx_turn += 1

    def define_input(self):
        """
        This function is to define the input for the model State Prediction
        :return: list of dictionaries with two keys:
                - 'prompt': the input
                - 'output': the label
                EX: [{'output': ******, 'prompt': *******}, {'output': ******, 'prompt': *******}, ...]
        """
        with open(self.sample_path, 'w', encoding='utf-8') as f:
            for child_dialogue in self.list_utter:
                if len(child_dialogue) == 0:
                    continue
                dict_input = define_instruction(child_dialogue)
                json.dump(dict_input, f)
                f.write("\n")