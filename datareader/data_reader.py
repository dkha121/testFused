from abc import abstractmethod, ABC
from typing import Dict, Union, Tuple
import pandas as pd


class DataReader(ABC):
    def __init__(self, data_path: str, sample_path: str, schema_path=None):
        """
        Preprocessing dataset format
        """
        self.data_path = data_path
        self.sample_path = sample_path
        self.schema_path = schema_path

        self.data = []
        self.list_utter = []
        self.list_input = []

    def start(self):
        print(f"\nStart reading {self.data_path}")

    def end(self):
        print(f"\nInput sample file is save to {self.sample_path}")

    def read_schema(self) -> Union[Tuple[Dict, Dict], None]:
        """
        Read schema guided to map old domain to new domain
        :return: two dictionaries to map old_domain to new_domain, old_slot to new_slot
        """
        if self.schema_path:
            # Read schema guided file
            schema_guided = pd.read_excel(self.schema_path, None)
            # Create dataframe of schema_guided
            df_schema = pd.DataFrame(columns=['domain', 'old slots', 'original dataset', 'new slots'])
            for schema in schema_guided.values():
                schema.columns = schema.columns.str.lower()
                schema.columns = schema.columns.str.replace('original domain', 'original dataset')
                df_schema = pd.concat([df_schema, schema], axis=0, ignore_index=True)
                df_schema['original dataset'] = df_schema['original dataset'].str.lower()
                df_schema['original dataset'] = df_schema['original dataset'].str.strip()
                df_schema['old slots'] = df_schema['old slots'].str.strip()
                df_schema['new slots'] = df_schema['new slots'].str.strip()

            df_schema = df_schema.dropna(how='all').reset_index(drop=True)
            df_schema = df_schema.fillna(method='ffill')
            # Create dict to map old domain to new domain
            dict_domain = dict(zip(df_schema['original dataset'], df_schema['domain']))

            dict_slot = dict()
            for new_domain in set(dict_domain.values()):
                child_df = df_schema[df_schema['domain'] == new_domain]
                dict_slot[new_domain] = dict(zip(child_df['old slots'], child_df['new slots']))

            return dict_domain, dict_slot

        return None

    @abstractmethod
    def load_data(self) -> None:
        """
        Read the data file (.json, .csv, ...)
        :return: the list of dictionaries or the dictionary of samples in the dataset
        """
        pass

    @abstractmethod
    def get_utterance(self) -> None:
        """
        Implement your convert logics to get utterances (<=5 utterance)
        :return: the list of list, each list contain utterances for each Input
                EX: [[utterance1, utterance2, utterance3, ...], [utterance4, utterance5, utterance6, ...]]
        """
        pass

    @abstractmethod
    def define_instruction(self, child_dialogue) -> Dict:
        """
        This function is to define the input and label for module state prediction
        :param child_dialogue: dialogue history for module 1
        :return: dictionary of input include two keys:
                - prompt: instruction
                - output: label
        """
        pass

    @abstractmethod
    def define_input(self) -> None:
        """
        Define the training sample.
        :return: list of dictionaries with two keys:
            - 'prompt': the sample
            - 'output': the label
            EX: [{'output': ******, 'prompt': *******}, {'output': ******, 'prompt': *******}, ...]
        """
        pass
