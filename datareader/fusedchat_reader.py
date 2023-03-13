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
    domain = []
    for i in service:
        domain.append(list(i.split("-"))[0])
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
    service = []
    for i in frame:
        service.append(i)
    domain = define_domain(service)
    dict_input['prompt'] = instruction.replace("<DIALOGUE_CONTEXT>",
                                               ''.join([turn for turn in list_turn])).replace('<DOMAIN>', domain)

    # Define label
    if 'dialog_act' in child_dialogue[-1].keys():
        dict_input['output'] = "Chitchat: None"
    else:

        domain_actions = []
        for key, action in child_dialogue[-1].items():

            domain_actions.append(action['dialog_act'])

            for domain_action in domain_actions:

                list_domain = dict()
                for domain_key, domain_val in domain_action.items():

                    domain1 = domain_key.split("-")[0]
                    action1 = domain_key.split("-")[1]
                    a = dict()
                    a[action1] = domain_val
                    if domain1 not in list_domain.keys():
                        list_domain[domain1] = a
                    else:
                        list_domain[domain1].__setitem__(action1, domain_val)
        output = dict()
        for k, v in list_domain.items():
            f = ""
            list_ac = []
            for ke, va in v.items():
                list_slot = []
                for val in va:
                    d = " ~ ".join(val)
                    list_slot.append(d)

                c = ke + ": " + "(" + "; ".join(list_slot) + ")"
                list_ac.append(c)
            f = "[" + ", ".join(list_ac) + "]"
            output[k] = f
        if len(list(output.keys())) == 1:
            for k, v in output.items():
                dict_input['output'] = "Database: " + k + "; " + v
        else:
            m = []
            for k, v in output.items():
                z = '{' + "Database: " + k + "; " + v + '}'
                m.append(z)
            dict_input['output'] = " AND ".join(m)
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
