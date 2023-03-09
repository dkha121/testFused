import json
import random
import sys
import pandas as pd

sys.path.append("./")

from datareader.data_reader import DataReader
from config import data_config


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
                        data_config.INSTRUCTION10]
    instruction = random.choice(list_instruction)
    # Define input
    dict_input = dict()
    list_turn = []
    for utter in child_dialogue:
        if len(utter["metadata"]) == 0:
            list_turn.append(data_config.USER_SEP + utter['text'] + data_config.EOT_SEP)
        else:
            list_turn.append(data_config.SYSTEM_SEP + utter['text'] + data_config.EOT_SEP)

    frame = child_dialogue[-1]['dialog_action']["dialog_act"]
    domain = list(child_dialogue[-1]['dialog_action']["dialog_act"].keys())[0].split("-")[0]
    # domain = define_domain(service)
    dict_input['prompt'] = instruction.replace("<DIALOGUE_CONTEXT>",
                                               ''.join([turn for turn in list_turn])).replace('<DOMAIN>', domain)

    # Define label
    list_action = []
    dict_label = dict()
    if 'dialog_act' in  child_dialogue[-1].keys():
        dict_input['output'] = "General"
    else:
        dict_label['Database'] = frame['service']
        for action in frame['actions']:
            dict_action = dict()
            act = action['act']
            if len(action['slot']) > 0:
                dict_action[act] = [(action['slot'] + ' ~ ' + value) for value in action['values']]
            else:
                dict_action[act] = ['none']
            list_action.append(dict_action)
        dict_input['output'] = "Database: " + frame['service'] + '; ' \
                               + str(list_action).replace('{', '').replace(']}', ')').replace(': [', ': (')
    return dict_input


class FUSEDCHATReader(DataReader):
    def __call__(self, *args, **kwargs):
        self.load_data()
        self.get_utterance()
        self.define_input()

    def load_data(self):
        """
        This function is to read the data file (.json, .csv, ...)
        :return: the list of dictionaries or the dictionary of samples in the dataset
        """
        with open(self.data_path, encoding='utf-8') as f:
            self.data = json.load(f)

    def get_utterance(self):
        """
        This function is to get utterance (<=5 utterance)
        :return: the list of list, each list contain utterances for each Input
                EX: [[utterance1, utterance2, utterance3, ...], [utterance4, utterance5, utterance6, ...]]
        """

        for id, dialogue in self.data.items():
            list_turns = []
            for key in dialogue["log"]:

                turn = key
                for k, v in dialogue["dialog_action"].items():
                    if int(k) == int(dialogue["log"].index(key)):
                        turn.__setitem__("dialog_action", v)

                list_turns.append(turn)
                len_turns = len(list_turns)

                idx_turn = 0
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
                if len(child_dialogue) <= 1:
                    continue
                dict_input = define_instruction(child_dialogue)
                json.dump(dict_input, f)
                f.write("\n")
