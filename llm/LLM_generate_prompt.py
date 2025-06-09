
import re
import time
from llm.interface_LLM import InterfaceLLM
import numpy as np

class Generate_Knowledge():

    def __init__(self, q_value_m, Num,  current_iter, total_iter, operator, parents, rank, code_size, current_code_size, debug, paras):
        # LLM_selection(q_value_m, v_vaule_m, Succ, Num)
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

        self.interface_llm = InterfaceLLM(self.llm_api_endpoint, self.llm_api_key, self.llm_model, self.debug_mode)




    def get_prompt_initilisation(self):


        self.Role_description = (
            "Your task is to create a clear and concise prompt to guide the design of the action selection function. "
            "The function should select the most appropriate action from a given set of options at each time slot, "
            "balancing exploration and exploitation. "
        )


        self.Input_Output_Interpretations = (
            "The following describes the inputs and output of the function:\n"
            "\nInputs:\n"
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

        self.Role_description_OUT = "Note that the prompt should follow a Chain-of-Thought framework, providing relevant context and key insights, to effectively guide the design process. It must be brief and enclosed in braces {}."


        prompt_content = self.Role_description +  self.Input_Output_Interpretations + self.Role_description_OUT


        return prompt_content


    def get_prompt_mutation(self):
        self.Role_description = (
            "Your task is to create a clear and concise prompt to guide the design of the action selection function. "
            "The function should select the most appropriate action from a given set of options at each time slot, "
            "balancing exploration and exploitation. "
        )


        self.Input_Output_Interpretations = (
            "The following describes the inputs and output of the function:\n"
            "\nInputs:\n"
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

        prompt_indiv = "\n" + self.parents_code['algorithm'][0] + "\n"

        Code_Instructions = (
                "The following is an existing prompt for the design of action selection function: " + prompt_indiv + "\n" 
                "It is ranked " + str(self.rank + 1) + " out of " + str(self.current_code_size) +
                " generated prompts.  Please assist me in creating a completely new prompt that improves upon this one. "
                "The new prompt must be enclosed within braces {}."
        )

        prompt_content =  self.Role_description +  self.Input_Output_Interpretations  + Code_Instructions

        return prompt_content


    def Read_code(self):

        if self.operator ==  "i1":
            prompt_content = self.get_prompt_initilisation()
        else:
            prompt_content = self.get_prompt_mutation()

        response = self.interface_llm.get_response(prompt_content)

        prompt = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if prompt == None:
            prompt = re.findall(r"\{(.*)\}", response, re.DOTALL)

        while (len(prompt) == 0):
            if self.debug_mode:
                print("Error: prompt not identified, wait 1 seconds and retrying ... ")
            time.sleep(1)
            response = self.interface_llm.get_response(prompt_content)

            prompt = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if prompt == None:
                prompt = re.findall(r"\{(.*)\}", response, re.DOTALL)

        return prompt
