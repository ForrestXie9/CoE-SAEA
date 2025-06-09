
import re
import time
from llm.interface_LLM import InterfaceLLM
import numpy as np

class Generate_Functions():

    def __init__(self, q_value_m, Num,  current_iter, total_iter, operator, parents, rank, code_size, current_code_size, offspring_c_decription, debug, paras):

        self.llm_api_endpoint  =  paras.llm_api_endpoint
        self.llm_api_key = paras.llm_api_key
        self.llm_model = paras.llm_model
        self.exp_debug_mode= paras.exp_debug_mode

        self.debug_mode = debug
        self.num_arm = len(q_value_m)
        self.q_value_m = q_value_m
        total_num = np.sum(Num)
        self.current_iter = current_iter
        self.total_iter = total_iter

        self.parents_code = parents
        self.rank = rank
        self.operator = operator
        self.pop_code_size = code_size
        self.current_code_size = current_code_size


        self.offspring_c_decription = offspring_c_decription


        self.interface_llm = InterfaceLLM(self.llm_api_endpoint, self.llm_api_key, self.llm_model, self.debug_mode)

        self.Role_description = (
            "You are tasked with developing a decision-making strategy to select the most suitable action from a given action set at each time slot. "
            "Each selected action receives a score, and your objective is to maximize the cumulative scores over all time slots. "
        )

        self.Code_Instructions = (
            "Implement your selection strategy in Python as a function named `selection`. "
            "This function should accept the following four inputs: `score_set`, `total_selection_count`, `current_time_slot`, and `total_time_slots`. "
            "The function must return a single output: `action_index`, which represents the index of the selected action."
        )

        self.Input_Output_Interpretations = (
            "The following describes the inputs and output of function:\n"
            "- `score_set` (dictionary): A dictionary where:\n"
            "  - Keys are integers (0 to 7) representing action indices, numbered 0 to 7.\n"
            "  - Values are lists of floats in the range [0, 1], where:\n"
            "    * Each float corresponds to a historical score for the respective action.\n"
            "    * The length of the list represents the number of times the action has been selected.\n"
            "- `total_selection_count` (integer): The total number of times all actions have been selected.\n"
            "- `current_time_slot` (integer): The current time slot. \n"
            "- `total_time_slots` (integer): The total number of time slots. \n"
            "\nOutput:\n"
            "- action_index (integer, between 0 and 7): The index of the selected action in action_set.\n")

        self.Guidance = "\nUseful tips: \n" + self.offspring_c_decription['algorithm'][0]

        self.Helpful_hints = (
            "\nPlease note: \n1) Scores and selection counts may be zero, so design the code to handle these cases without errors. "
            "\n2) Your designed selection strategy can use any mathematical operations."
            "\n3) You only import NumPy library."
            "\n3) Avoid adding extra explanations.")

    def get_prompt_initilisation(self):


        prompt_content = self.Role_description + self.Code_Instructions + self.Input_Output_Interpretations  + self.Helpful_hints +  self.Guidance

        return prompt_content

    def get_prompt_mutation(self):

        prompt_indiv = "\n" + self.parents_code['code'] + "\n"

        Code_Instructions = ("The current selection strategy is implemented as follows:\n" + prompt_indiv +
                             "\nThis strategy is ranked " + str(self.rank + 1) + " out of " + str(self.current_code_size) +
                             " strategies you generated. Please assist me in creating a completely new strategy that improves upon this one, focusing on enhancing its performance, efficiency, readability, and robustness.")

        prompt_content = self.Role_description + Code_Instructions + self.Code_Instructions + self.Input_Output_Interpretations+ self.Helpful_hints  + self.Guidance

        return prompt_content



    def Read_code(self):

        if self.operator == "i1":
            prompt_content = self.get_prompt_initilisation()
        else:
            prompt_content = self.get_prompt_mutation()

        response = self.interface_llm.get_response(prompt_content)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if code == None:
            code = re.findall(r"def.*return", response, re.DOTALL)

        while (len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")
            time.sleep(1)
            response = self.interface_llm.get_response(prompt_content)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if code == None:
                code = re.findall(r"def.*return", response, re.DOTALL)

        code = code[0]
        prompt_func_outputs = ["action_index"]
        code_all = code + " " + ", ".join(s for s in prompt_func_outputs)


        return code_all


    def code2file(self,code):
        with open("./select_fun.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return



