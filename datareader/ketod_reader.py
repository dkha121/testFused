import json
import random
import sys
sys.path.append("./")

from datareader.data_reader import DataReader
from config import data_config
from config.ontology import dict_schema, dict_user_action


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

    def define_instruction(self, child_dialogue):
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
            if utter['speaker'] == "USER":
                list_turn.append(data_config.USER_SEP + utter['utterance'] + data_config.EOT_SEP)
            elif utter['speaker'] == "SYSTEM":
                list_turn.append(data_config.SYSTEM_SEP + utter['utterance'] + data_config.EOT_SEP)

        frame = child_dialogue[-1]['frames'][0]
        if len(frame['slots']) == 0:
            dict_input['prompt'] = instruction.replace("<DIALOGUE_CONTEXT>",
                                                        ''.join([turn for turn in list_turn]))\
                                                        .replace('<DOMAIN>', '')
            dict_input['output'] = "Chitchat: None"
            return dict_input

        # Get old domain
        service = frame['service'].lower()
        dict_domain, dict_slot = self.read_schema()
        assert service in dict_domain.keys(), f"Domain {service} is not exists!"
        # Map to new domain
        domain = dict_domain[service]
        # Define instructions
        dict_input['prompt'] = instruction.replace("<DIALOGUE_CONTEXT>",
                                                   ''.join([turn for turn in list_turn]))\
                                                        .replace('<DOMAIN>', dict_schema[domain])

        # Define label
        list_action = []
        dict_label = dict()
        # Get domain
        dict_label['Database'] = frame['service']
        for action in frame['actions']:
            dict_action = dict()
            act = action['act']
            new_act = dict_user_action[act]
            slot_name = action['slot']
            if len(slot_name) > 0 and slot_name != 'intent':
                dict_action[new_act] = [(dict_slot[domain][slot_name] + ' ~ ' + value) for value in action['values']]
            else:
                dict_action[new_act] = ['none']
            list_action.append(dict_action)
        dict_input['output'] = "Database: " + frame['service'] + '; ' \
                               + str(list_action).replace('{', '').replace(']}', ')').replace(': [', ': (')

        return dict_input

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
                dict_input = self.define_instruction(child_dialogue)
                if dict_input:
                    json.dump(dict_input, f)
                    f.write("\n")
