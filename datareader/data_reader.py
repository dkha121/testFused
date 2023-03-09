from abc import abstractmethod, ABC
from typing import List, Dict, Union


class DataReader(ABC):
    def __init__(self, data_path: str, sample_path: str):
        """
        Preprocessing dataset format
        """
        self.data_path = data_path
        self.sample_path = sample_path

        # self.instruction = data_config.INSTRUCTION
        # self.ctx_sep = data_config.CTX_SEP
        # self.eod_sep = data_config.EOD_SEP
        # self.eot_sep = data_config.EOT_SEP
        # self.ques_sep = data_config.QUES_SEP
        # self.user_sep = data_config.USER_SEP
        # self.sys_sep = data_config.SYSTEM_SEP
        # self.list_act = data_config.LIST_ACT
        # self.list_domain = data_config.LIST_DOMAIN

        self.data = []
        self.list_utter = []
        self.list_input = []

    def start(self):
        print(f"\nStart reading {self.data_path}")

    def end(self):
        print(f"\nInput sample file is save to {self.sample_path}")

    @abstractmethod
    def load_data(self) -> None:
        """
        Read the data file (.json, .csv, ...)
        :return: the list of dictionaries or the dictionary of samples in the dataset
        """
        pass

    @abstractmethod
    def get_utterance(self) -> List[List[str]]:
        """
        Implement your convert logics to get utterances (<=5 utterance)
        :return: the list of list, each list contain utterances for each Input
                EX: [[utterance1, utterance2, utterance3, ...], [utterance4, utterance5, utterance6, ...]]
        """
        pass

    @abstractmethod
    def define_input(self) -> List[Dict[str, str]]:
        """
        Define the training sample.
        :return: list of dictionaries with two keys:
            - 'prompt': the sample
            - 'output': the label
            EX: [{'output': ******, 'prompt': *******}, {'output': ******, 'prompt': *******}, ...]
        """
        pass



