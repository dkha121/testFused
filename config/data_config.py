path_schema_guided = ""
EOT_SEP = "[EOT]. "
USER_SEP = "USER: "
SYSTEM_SEP = "SYSTEM: "

LIST_ACT = "user_action: inform, request, greeting, bye, general; "
LIST_RULE = "1. Chitchat, 2. General, 3. Database: "
DOMAIN_BUS = "Bus; Slot: day, departure, destination, leaveat. "

INSTRUCTION1 = f"Instruction: In this task given a dialogue as context you must be given the type of belief state " \
               f"between specified people or speakers. [CTX]<DIALOGUE_CONTEXT>[EOD]. " \
               f"[OPT] {LIST_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? "
INSTRUCTION2 = f'Instruction: Please use this dialogue [CTX]<DIALOGUE_CONTEXT>[EOD] to predict the type of ' \
               f'belief state between two people. ' \
               f'[OPT] {LIST_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state?'
INSTRUCTION3 = f'Instruction: This dialogue [CTX]<DIALOGUE_CONTEXT>[EOD] is used to generate ' \
               f'the belief state between two speakers. ' \
               f'[OPT] {LIST_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION4 = f'Instruction: You must be given the type of belief state between specified people or speakers ' \
               f'base on this dialogue [CTX]<DIALOGUE_CONTEXT>[EOD]. '\
               f'[OPT] {LIST_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION5 = f'Instruction: Please predict the type of belief state between two persons or two speakers base on ' \
               f'this dialogue [CTX]<DIALOGUE_CONTEXT>[EOD]. ' \
               f'[OPT] {LIST_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION6 = f'Instruction: Let give the belief state between two persons or two speakers base on ' \
               f'this dialogue [CTX]<DIALOGUE_CONTEXT>[EOD]. ' \
               f'[OPT] {LIST_ACT}; {LIST_RULE} <DOMAIN> [Q] What is the belief state? '
INSTRUCTION7 = 'INSTRUCTION7 '
INSTRUCTION8 = 'INSTRUCTION8 '
INSTRUCTION9 = 'INSTRUCTION9 '
INSTRUCTION10 = 'INSTRUCTION10 '
INSTRUCTION11 = 'INSTRUCTION11'
INSTRUCTION12 = 'INSTRUCTION12'
INSTRUCTION13 = 'INSTRUCTION13'
INSTRUCTION14 = 'INSTRUCTION14'
INSTRUCTION15 = 'INSTRUCTION15'
INSTRUCTION16 = 'INSTRUCTION16'

a = f"{INSTRUCTION1}"
