from abc import abstractmethod, ABC


class DataReader(ABC):
    def __init__(self, data_path: str, sample_path: str):
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
    def load_data(self):
        pass

    @abstractmethod
    def get_utterance(self):
        pass

    @abstractmethod
    def define_input(self):
        pass



