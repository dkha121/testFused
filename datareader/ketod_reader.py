import json
import random
import sys
sys.path.append("./")

from datareader.data_reader import DataReader
from config import data_config


def define_instruction(child_dialogue):
    """
    :return:
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
        if utter['speaker'] == "USER":
            list_turn.append(data_config.USER_SEP + utter['utterance'] + data_config.EOT_SEP)
        elif utter['speaker'] == "SYSTEM":
            list_turn.append(data_config.SYSTEM_SEP + utter['utterance'] + data_config.EOT_SEP)

    dict_input['prompt'] = instruction + data_config.CTX_SEP \
                           + ''.join([turn for turn in list_turn]) \
                           + data_config.EOD_SEP + data_config.OPT_SEP + data_config.LIST_ACT \
                           + data_config.LIST_RULE + data_config.LIST_DOMAIN

    # Define label
    list_action = []
    dict_label = dict()
    frame = child_dialogue[-1]['frames'][0]
    if len(frame['slots']) == 0:
        dict_input['output'] = "Chitchat: None"
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


class KETODReader(DataReader):
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
        for dialogue in self.data:
            list_turns = dialogue['turns']
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
                if len(child_dialogue) == 0:
                    continue
                dict_input = define_instruction(child_dialogue)
                json.dump(dict_input, f)
                f.write("\n")
