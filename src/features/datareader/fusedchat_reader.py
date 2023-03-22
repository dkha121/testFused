import json
import random
# import sys
# sys.path.append("./")

from src.config import data_config
from src.features.datareader.data_reader import DataReader


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
            if self.context_window is None:
                self.context_window = 5
            cw = self.context_window - 1
            idx_turn = 0
            if len_turns %2 ==0:
                while idx_turn <= len_turns-2:
                    self.list_utter.append(list_turns[max(0, idx_turn - cw):idx_turn + 1])
                    idx_turn += 2
            else:
                while idx_turn <= len_turns-1:
                    self.list_utter.append(list_turns[max(0, idx_turn - cw):idx_turn + 1])
                    idx_turn += 2

    def define_instruction(self,child_dialogue, dict_slot):
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
        prefix = data_config.SYSTEM_SEP if len(child_dialogue) % 2 == 0 else data_config.USER_SEP

        for index, utter in enumerate(child_dialogue):
            list_turn.append(prefix + utter['text'] + data_config.EOT_SEP)
            prefix = data_config.USER_SEP if prefix == data_config.SYSTEM_SEP else data_config.SYSTEM_SEP

        list_domain_slots = []

        # There are two case, current utterances is TOD and ODD
        if 'dialog_action' not in child_dialogue[-1].keys():
            dict_input['prompt'] = instruction.replace("<DIALOGUE_CONTEXT>",
                                                       ''.join([turn for turn in list_turn])).replace('<DOMAIN>; Slots: <SLOT>',
                                                                                                     "")
        else:
            frame = child_dialogue[-1]['dialog_action']["dialog_act"]
            service = [i for i in frame]
            domain_set = set(i.split("-")[0] for i in service)
            for k, v in dict_slot.items():
                if k in domain_set:
                    slot_set = {y for i, y in v.items()}
                    database_and_slots = f"{k}; Slots: {', '.join(slot_set)}"
                    list_domain_slots.append(database_and_slots)

            dict_input['prompt'] = instruction.replace("<DIALOGUE_CONTEXT>",
                                                       ''.join([turn for turn in list_turn])).replace('<DOMAIN>; Slots: <SLOT>', " AND ".join(list_domain_slots))
        nowhite = ' '.join(dict_input['prompt'].split())
        dict_input['prompt'] = nowhite
        # Define label
        # there are three case in label. If current utterance is ODD : label = Chitchat: general
        list_domain = dict()
        if 'dialog_action' not in child_dialogue[-1].keys():
            dict_input['output'] = "Chitchat: general"
        # if current utterance has domain general : label = Chitchat: action
        elif 'general' in domain_set:
            dict_input['output'] = "Chitchat: " + list(service[0].split("-"))[1]
        # if current utterance is TOD
        else:
            domain_actions = []
            domain_actions.append(child_dialogue[-1]['dialog_action']['dialog_act'])
            for domain_action in domain_actions:
                for domain_key, domain_val in domain_action.items():
                    domain1 = domain_key.split("-")[0]
                    action1 = domain_key.split("-")[1].lower()
                    action_slot = dict()
                    action_slot[action1] = domain_val
                    for f in domain_val:
                        if f[0] == 'none':
                            domain_val.remove(f)
                    if domain1 not in list_domain.keys():
                        list_domain[domain1] = action_slot
                    else:
                        list_domain[domain1].__setitem__(action1, domain_val)

            output = dict()
            for k, v in list_domain.items():
                list_ac = []
                for ke, va in v.items():
                    list_slot = [" ~ ".join(val) for val in va if val]
                    if list_slot:
                        c = ke + ": " + "(" + "; ".join(list_slot) + ")"
                        list_ac.append(c)
                list_ac = [x for x in list_ac if "()" not in x]
                if list_ac:
                    f = "[" + ", ".join(list_ac) + "]"
                    output[k] = f

            none_list = [k for k, v in output.items() if v == '[]']
            for zeros in none_list:
                del output[zeros]
            if len(output) == 1:
                k, v = next(iter(output.items()))
                dict_input['output'] = f"Database: {k}; {v}"
            elif len(output) == 0:
                dict_input['output'] = ''
            else:
                complete_output = [f"Database: {k}; {v}" for k, v in output.items()]
                dict_input['output'] = " AND ".join(complete_output)
        return dict_input

    def define_input(self):
        """
        This function is to define the input for the model State Prediction
        :return: list of dictionaries with two keys:
                - 'prompt': the input
                - 'output': the label
                EX: [{'output': ******, 'prompt': *******}, {'output': ******, 'prompt': *******}, ...]
        """
        _, dict_slot = self.read_schema()
        with open(self.sample_path, 'w', encoding='utf-8') as f:
            for child_dialogue in self.list_utter:
                if len(child_dialogue) <= 0:
                    continue
                dict_input = self.define_instruction(child_dialogue, dict_slot)
                if dict_input['output'] == '':
                    continue
                json.dump(dict_input, f, indent=4)
                f.write("\n")
