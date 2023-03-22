# import sys
# sys.path.append("./")
import json
import random
from tqdm import tqdm

from src.features.datareader.data_reader import DataReader
from src.config import data_config
from src.config.ontology import dict_user_action, LIST_CHITCHAT_ACT


class KETODReader(DataReader):
    def __call__(self, *args, **kwargs):
        self.dict_domain, self.dict_slot = self.read_schema()
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
                EX: [[utterance1, utterance2, utterance3, ...],
                    [utterance4, utterance5, utterance6, ...]]
        """
        cw = self.context_window - 1  # slide window
        for dialogue in self.data:
            list_turns = dialogue['turns']
            for idx in range(len(list_turns)):
                if idx % 2 == 0:  # current_utter is user
                    if list_turns[idx + 1]['enrich']:  # next utter enrich = True
                        child_dialogue = list_turns[max(0, idx - cw):max(0, idx + 2)]
                    else:  # next utter enrich = False
                        child_dialogue = list_turns[max(0, idx - cw):max(0, idx + 1)]
                    self.list_utter.append(child_dialogue)

    def define_sample_tod(self, frame, instruction, list_turn, dict_input):
        """
        Define sample for TOD only
        """
        # Get old domain
        service = frame['service'].lower()
        assert service in self.dict_domain.keys(), f"Domain {service} is not exists!"
        # Map to new domain
        domain = self.dict_domain[service]

        # Define instructions
        dict_input['prompt'] = instruction.replace(
            "<DIALOGUE_CONTEXT>", ''.join([turn for turn in list_turn])) \
            .replace('<DOMAIN>', domain) \
            .replace('<SLOT>', str(set(self.dict_slot[domain].values())) \
                     .replace('[', '').replace(']', '').replace("'", "")) \
            .replace(' {', ' ').replace('} ', '. ')

        # Define label
        list_action = []
        for action in frame['actions']:
            dict_action = dict()
            act = action['act']
            new_act = dict_user_action[act]
            slot_name = action['slot']

            if len(slot_name) > 0:
                if slot_name != 'intent':
                    slot_name = self.dict_slot[domain][slot_name]
                dict_action[new_act] = [
                    (slot_name + ' ~ ' + value)
                    for value in action['values']]
            elif len(slot_name) == 0:
                dict_action[new_act] = ['general']
            list_action.append(dict_action)

        dict_input['output'] = "Database: " + domain + '; ' \
                               + str(list_action).replace('{', '') \
                                   .replace(']}', ')') \
                                   .replace(': [', ': (') \
                                   .replace('()', "('general')") \
                                   .replace('[}]', '') \
                                   .replace("'", "")
        nowhite = ' '.join(dict_input['prompt'].split())
        dict_input['prompt'] = nowhite
        return dict_input

    def define_instruction(self, child_dialogue):
        """
        Define the input and label for module state prediction
        :param child_dialogue: dialogue history for module 1
        :return: dictionary of input include two keys:
                - prompt: instruction
                - output: label
        """
        # Define instruction
        list_instruction = [data_config.INSTRUCTION1, data_config.INSTRUCTION2,
                            data_config.INSTRUCTION3, data_config.INSTRUCTION4,
                            data_config.INSTRUCTION5, data_config.INSTRUCTION6,
                            data_config.INSTRUCTION7, data_config.INSTRUCTION8,
                            data_config.INSTRUCTION9, data_config.INSTRUCTION10]
        instruction = random.choice(list_instruction)

        # Define input
        dict_input = dict()
        list_turn = []

        for utter in child_dialogue:
            if utter['speaker'] == "USER":
                speaker = data_config.USER_SEP
            else:
                speaker = data_config.SYSTEM_SEP
            if 'enriched_utter' in utter:  # For TOD -> ODD or ODD
                list_turn.append(speaker + utter['enriched_utter']
                                 + data_config.EOT_SEP)
            else:  # only TOD
                list_turn.append(speaker + utter['utterance']
                                 + data_config.EOT_SEP)

        if len(child_dialogue) == self.context_window + 1:  # TOD -> ODD
            entity_query = child_dialogue[-1]['entity_query']  # system label
            frame = child_dialogue[-2]['frames'][0]  # user label
            list_turn = list_turn[:-1]  # remove SYSTEM utterance

            if len(entity_query) < 1:  # []
                actions = frame['actions']
                if len([action['slot'] for action in actions if len(action['slot']) > 0]) == 0: # "slot" is empty
                    # input
                    dict_input['prompt'] = instruction.replace(
                        "<DIALOGUE_CONTEXT>", ''.join([turn for turn in list_turn])) \
                        .replace('<DOMAIN>', '')
                    # label
                    chitchat_actions = set([dict_user_action[action['act']] for action in actions if action['act'] in LIST_CHITCHAT_ACT])
                    if len(chitchat_actions) == 0:
                        chitchat_actions = 'general'
                    else:
                        chitchat_actions = str(chitchat_actions).replace("{", "").replace("}", "").replace("'", "")
                    dict_input['output'] = f"{chitchat_actions}. Chitchat: None"

                    nowhite = ' '.join(dict_input['prompt'].split())
                    dict_input['prompt'] = nowhite
                    return dict_input
                else:
                    return self.define_sample_tod(frame, instruction, list_turn, dict_input)

            else:  # Ex: [['rentalcars : pickup location : LGB Airport']]
                # Get old domain
                service = frame['service'].lower()
                assert service in self.dict_domain.keys(), f"Domain" \
                                                           " {service} is not exists!"
                # Map to new domain
                domain = self.dict_domain[service]
                dict_input['prompt'] = instruction.replace(
                    "<DIALOGUE_CONTEXT>", ''.join([turn for turn in list_turn])) \
                    .replace('<DOMAIN>', domain)
                dict_input['output'] = "general. Seek: {}".format(
                    ' and '.join([turn[0] for turn in entity_query]))

                nowhite = ' '.join(dict_input['prompt'].split())
                dict_input['prompt'] = nowhite
                return dict_input

        else:  # TOD or ODD
            frame = child_dialogue[-1]['frames'][0]
            return self.define_sample_tod(frame, instruction, list_turn, dict_input)

    def define_input(self):
        """
        This function is to define the input for the model State Prediction
        :return: list of dictionaries with two keys:
                - 'prompt': the input
                - 'output': the label
                EX: [{'output': ******, 'prompt': *******},
                     {'output': ******, 'prompt': *******}, ...]
        """
        with open(self.sample_path, 'w', encoding='utf-8') as f:
            for child_dialogue in tqdm(self.list_utter):
                if len(child_dialogue) != (self.context_window + 1) and child_dialogue[-1]['speaker'] != 'USER':
                    continue
                dict_input = self.define_instruction(child_dialogue)
                if dict_input:
                    json.dump(dict_input, f)
                    f.write("\n")
