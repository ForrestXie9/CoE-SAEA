import numpy as np
from llm.Orga_Code import Orga_Code
from llm.Orga_Code_prompt import Orga_Code_prompt

class LLM_Configurators_Setting(object):

    def __init__(self, num_arm, hf, ghf, maxFEs, popsize, pop_code_size, debug, paras):
        self.num_arm = num_arm
        self.ghf = ghf
        self.hf = hf
        self.maxFEs = maxFEs
        self.popsize = popsize
        self.debug = debug
        self.paras = paras
        self.pop_code_size = pop_code_size


    def Prompt_description(self, q_value_m, Num, id, pop_code, operator, candidate_fit = None):

        self.O_Code_prompt = Orga_Code_prompt(self.num_arm, self.ghf, q_value_m, Num, id, self.maxFEs, self.popsize,
                                pop_code, self.pop_code_size, self.debug, self.paras)


        offspring_c = self.O_Code_prompt.generate_prompt(operator)
        print(operator, 'prompt')


        while offspring_c.get('algorithm') == None:
            offspring_c = self.O_Code_prompt.generate_prompt(operator)


        return offspring_c



    def Selection_code(self, q_value_m, Num, id, pop_code, code_database, offspring_c_decription, scores_dic, total_time_solots, operator, candidate_fit = None):


        self.O_Code = Orga_Code(self.hf, self.num_arm, self.ghf, q_value_m, Num, id, self.maxFEs, self.popsize,
                                pop_code, code_database, self.pop_code_size, self.debug, offspring_c_decription, scores_dic, total_time_solots, self.paras)

        index1, offspring_c, rank = self.O_Code.generate_arm_index(operator)
        print(operator, index1)
        creat_num = 0
        while index1 < 1 or index1 > self.num_arm:
            creat_num += 1
            if self.debug:
                print("Warning! Generated Algorithm Error, retrying ... ")
            index1, offspring_c, rank = self.O_Code.generate_arm_index(operator)
            if creat_num > 30:
                break


        return  index1, offspring_c, rank