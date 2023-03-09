import ontology


path_schema_guided = ""
EOT_SEP = "[EOT]. "
USER_SEP = "USER: "
DIALOGUE_CONTEXT = ""
SYSTEM_SEP = "SYSTEM: "
LIST_RULE = "1. Seek, 2. General, 3. Database: "

INSTRUCTION1 = f"Instruction: In this task given a dialogue as context you must be given the type of belief state " \
               f"between specified people or speakers. [CTX]{DIALOGUE_CONTEXT}[EOD]. " \
               f"[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? "
INSTRUCTION2 = f'Instruction: Please use this dialogue [CTX]{DIALOGUE_CONTEXT}[EOD] to predict the type of ' \
               f'belief state between two people. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state?'
INSTRUCTION3 = f'Instruction: This dialogue [CTX]{DIALOGUE_CONTEXT}[EOD] is used to generate ' \
               f'the belief state between two speakers. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION4 = f'Instruction: You must be given the type of belief state between specified people or speakers ' \
               f'base on this dialogue [CTX]{DIALOGUE_CONTEXT}[EOD]. '\
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION5 = f'Instruction: Please predict the type of belief state between two persons or two speakers base on ' \
               f'this dialogue [CTX]{DIALOGUE_CONTEXT}[EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION6 = f'Instruction: Let give the belief state between two persons or two speakers base on ' \
               f'this dialogue [CTX]{DIALOGUE_CONTEXT}[EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION7 = 'INSTRUCTION7 '
INSTRUCTION8 = 'INSTRUCTION8 '
INSTRUCTION9 = 'INSTRUCTION9 '
INSTRUCTION10 = 'INSTRUCTION10 '
INSTRUCTION11 = 'INSTRUCTION11'
INSTRUCTION12 = f'Instruction: By examining the given conversation, you must be able to recognize the belief state that exists  ,' \
               f'between the two individuals. [CTX]{DIALOGUE_CONTEXT}[EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION13 = f'Instruction: You will be required to analyze the conversation provided and determine the type of belief state between  ,' \
               f'the specified individuals or speakers. [CTX]{DIALOGUE_CONTEXT}[EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION14 = f'Instruction: The goal of this assignment is to determine the belief state between the specified individuals by analyzing ,' \
               f'the dialogue provided. [CTX]{DIALOGUE_CONTEXT}[EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION15 = f'Instruction: Your objective is to identify the belief state between two people or speakers by analyzing ,' \
               f'the given dialogue. [CTX]{DIALOGUE_CONTEXT}[EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION16 = f'Instruction: You need to determine the type of belief state between the specified speakers based on,' \
               f'this provided dialogue. [CTX]{DIALOGUE_CONTEXT}[EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
