import numpy as np
from pyDOE import lhs
from Configurations.DE_operator import DEoperator
from Configurations.RBF_pre_arm import RBF_pre_arm
from Configurations.GP_lcb_arm import GP_lcb_arm
from Configurations.RBF_ls_arm import RBF_ls_arm
from Configurations.GP_EI_arm import GP_EI_arm
from Configurations.PRS_pre_arm import PRS_pre_arm
from Configurations.PRS_ls_arm import PRS_ls_arm
from Configurations.GBC_eoi_arm import KNN_eoi_arm
from Configurations.GBC_eor_arm import KNN_eor_arm
import json
import os
from llm.LLM_Configurators_Setting import LLM_Configurators_Setting

class CoE_SAEA(object):
    def __init__(self, runs, fun_name, maxFEs, FUN, dim, LB, UB, debug, paras):
        self.paras = paras
        self.maxFEs = maxFEs
        self.fun_name = fun_name
        self.NFEs = 0
        self.popsize = 100
        self.initial_sample_size = self.popsize

        self.total_time_solots = self.maxFEs - self.popsize

        self.F = 0.5
        self.CR = 0.9
        self.level = 5

        self.dim = dim
        self.cxmin = np.array(LB)
        self.cxmax = np.array(UB)
        self.bound = self.cxmax - self.cxmin
        self.FUN = FUN

        self.database = None
        self.hx = None
        self.hf = None

        self.CE = np.zeros((maxFEs, 2))
        self.gfs = np.zeros(maxFEs)
        self.VRmin = np.tile(LB, (self.popsize, 1))
        self.VRmax = np.tile(UB, (self.popsize, 1))

        self.id = 1
        self.rbf_model = []
        self.gp_model = []
        self.knn_model = []
        self.prs_model = []

        self.Save_rp = []
        self.Save_rl = []
        self.Save_gl = []
        self.Save_ge = []
        self.Save_pp = []
        self.Save_pl = []
        self.Save_Gi = []
        self.Save_Go = []

        self.num_arm = 8

        self.q_value_m = np.zeros(self.num_arm).tolist()
        self.Num = np.zeros(self.num_arm).tolist()

        # Decision-expert
        self.pop_code = []
        self.pop_code_size = 10
        self.debug = debug
        self.output_path = "./"
        self.code_database = []

        # Scoring-expert
        self.pop_code_se = []
        self.pop_code_size_se = 10
        self.code_database_se = []
        self.pop_code_prompt = []
        self.code_database_prompt = []

        self.runs = runs

        self.scores_dic = {i: [0.0] for i in range(0, 8)}
        self.selection_count = {i: 0 for i in range(0, 8)}


    def initPop(self):
        sam = np.tile(self.cxmin, (self.initial_sample_size, 1)) + (np.tile(self.cxmax, (self.initial_sample_size, 1)) - np.tile(self.cxmin, (self.initial_sample_size, 1))) * lhs(self.dim, samples=self.initial_sample_size, criterion='center')
        fitness = np.zeros((self.initial_sample_size))
        for i in range(self.initial_sample_size):
            fitness[i] = self.FUN(sam[i, :])
            self.CE[self.NFEs, :] = [self.NFEs , self.FUN(sam[i, :])]
            self.NFEs += 1
            self.gfs[i] = np.min(self.CE[0:self.NFEs, 1])

        self.database = [sam,  fitness]
        self.hx = sam
        self.hf = fitness
        sort_index = np.argsort(self.hf)
        self.ghf = self.hf[sort_index[:self.initial_sample_size]]
        self.ghx = self.hx[sort_index[:self.initial_sample_size]]

    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def add2pop(self,population,offspring):

        population.append(offspring)
        return True

    def add2pop2(self,population,offspring):

        population.append(offspring)
        return True

    def deduplicate_database(self):
        # Create a dictionary to store the best (min objective) entry for each unique 'code'
        unique_codes = {}
        unique_codes_prompt = {}
        for entry in self.code_database:
            code = entry['code']
            objective = entry['objective']

            # If the code is not in the dictionary, or this entry has a smaller objective, update it
            if code not in unique_codes or objective < unique_codes[code]['objective']:
                unique_codes[code] = entry

        # Convert the dictionary values back to a list and update self.code_database
        self.code_database = list(unique_codes.values())

        for entry2 in self.code_database_prompt:
            algorithm = entry2['algorithm'][0]
            objective = entry2['objective']

            # Update the dictionary if this entry has a smaller objective
            if algorithm not in unique_codes_prompt or objective < unique_codes_prompt[algorithm]['objective']:
                unique_codes_prompt[algorithm] = entry2
            # Update self.code_database_prompt with deduplicated entries
        self.code_database_prompt = list(unique_codes_prompt.values())

    def update_or_replace_in_database(self, offspring_c):

        for i, ind in enumerate(self.code_database):
            if ind['code'] == offspring_c['code']:
                if offspring_c['objective'] < ind['objective']:
                    self.code_database[i] = offspring_c
                return

        self.code_database.append(offspring_c)

    def update_code_database_prompt(self, offspring_c_description):

        for i, entry2 in enumerate(self.code_database_prompt):
            if entry2['algorithm'][0] == offspring_c_description['algorithm'][0]:
                if offspring_c_description['objective'] < entry2['objective']:
                    self.code_database_prompt[i] = offspring_c_description
                return

        self.code_database_prompt.append(offspring_c_description)


    def update(self, index, reward, offspring_c, offspring_c_decription, candidate_fit, operator, rank):
        if candidate_fit == None:
           candidate_fit = np.max(self.hf) + 1 # Output the worst tag

        index2 = int(index - 1)
        average_re = self.q_value_m[index2]
        new_re = (average_re * self.Num[index2] + reward)/(self.Num[index2] + 1)
        self.q_value_m[index2] = new_re
        self.Num[index2] = self.Num[index2] + 1
        self.selection_count[index2] = self.selection_count[index2] + 1
        self.scores_dic[index2].append(reward)

        if operator == "s1":
            offspring_c = self.pop_code[rank]
            offspring_c['objective'] = candidate_fit

        if operator == "i1" or operator == "e1":
            offspring_c['objective'] = candidate_fit
            offspring_c_decription['objective'] = candidate_fit
            self.update_code_database_prompt(offspring_c_decription)

        self.update_or_replace_in_database(offspring_c)
        self.code_database = sorted(self.code_database, key=lambda x: x['objective'])
        self.pop_code = self.code_database[:self.pop_code_size]
        self.code_database_prompt = sorted(self.code_database_prompt, key=lambda x: x['objective'])
        self.pop_code_prompt = self.code_database_prompt[:self.pop_code_size]


    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
            # Write the code to the file
            file.write(code)
        return

    def add2pop(self,population, offspring):
        population.append(offspring)
        return True


    def run_action(self, index1, offspring_c, offspring_c_decription, operator, rank):

        candidate_fit =  []
        reward = None
        if 1 <= index1 <= self.num_arm:
            if self.NFEs < self.maxFEs:

                if index1 == 1:
                    offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward, self.NFEs, self.CE, self.gfs, candidate_position, candidate_fit = RBF_pre_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm, self.paras)


                elif index1 == 2:
                    offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward, self.NFEs, self.CE, self.gfs, candidate_position, candidate_fit = GP_lcb_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm, self.paras)


                elif index1 == 3:
                    self.hx, self.hf, reward, self.NFEs, self.CE, self.gfs, candidate_position, candidate_fit = RBF_ls_arm(self.ghx, self.ghf, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.cxmin, self.cxmax, self.num_arm, self.paras)


                elif index1 == 4:
                    offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward, self.NFEs, self.CE, self.gfs, candidate_position, candidate_fit = GP_EI_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm, self.paras)


                elif index1 == 5:
                    offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward, self.NFEs, self.CE, self.gfs, candidate_position, candidate_fit = PRS_pre_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.num_arm, self.paras)


                elif index1 == 6:
                    self.hx, self.hf, reward, self.NFEs, self.CE, self.gfs, candidate_position, candidate_fit = PRS_ls_arm(self.ghx, self.ghf, self.hx, self.hf, self.FUN, self.NFEs, self.CE, self.gfs, self.cxmin, self.cxmax, self.num_arm, self.paras)


                elif index1 == 7:
                    offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward, self.NFEs, self.CE, self.gfs, candidate_position, candidate_fit = KNN_eoi_arm(self.ghx, self.ghf, offspring, self.hx, self.hf,self.FUN, self.NFEs, self.level, self.CE, self.gfs, self.num_arm, self.paras)


                elif index1 == 8:
                    offspring = DEoperator(self.ghx, self.initial_sample_size, self.dim, self.ghx, self.F, self.CR, self.VRmax, self.VRmin)
                    self.hx, self.hf, reward, self.NFEs, self.CE, self.gfs, candidate_position, candidate_fit = KNN_eor_arm(self.ghx, self.ghf, offspring, self.hx, self.hf, self.FUN, self.NFEs, self.level, self.CE, self.gfs, self.num_arm, self.paras)

                # if candidate_fit != []:

                self.update(index1, reward, offspring_c, offspring_c_decription, candidate_fit, operator, rank)


        return self.NFEs, candidate_fit


    def operator_choice(self, candidate_fit = None):

        if len(self.pop_code) < self.pop_code_size:
            operator = "i1"
        else:
            N = len(self.ghf)

            if candidate_fit == []:
                p1 = 0
            else:
                ghf_sum = np.concatenate((self.ghf.reshape(1, -1), candidate_fit.reshape(1, -1)), axis=1)
                index = np.argsort(ghf_sum)
                in_ = (np.where(index == N)[1][0]) + 1
                p1 = (N+1-in_) / N
            if np.random.rand() < p1:
                operator = "s1"
            else:
                operator = "e1"

        return operator

    def run(self):
        if self.database is None:
            self.initPop()

        candidate_fit = 0
        operator = "i1"
        while self.NFEs < self.maxFEs:
            sort_index = np.argsort(self.hf)
            self.ghf = self.hf[sort_index[:self.initial_sample_size]]
            self.ghx = self.hx[sort_index[:self.initial_sample_size]]

            if self.NFEs >= 1000:
                break

            if (len(self.pop_code) < self.pop_code_size) or (len(self.pop_code_prompt) < self.pop_code_size):
                operator == "i1"  # initialize piror knowledge and configuration function
            else:
                operator = self.operator_choice(candidate_fit) # exploration or exploitation

            C1 = LLM_Configurators_Setting(self.num_arm, self.ghf, self.hf, self.maxFEs, self.popsize, self.pop_code_size, self.debug,
                              self.paras)

            if operator == "i1" or operator == "e1":

                offspring_c_decription = C1.Prompt_description(self.q_value_m, self.Num, self.id, self.pop_code_prompt, operator, candidate_fit)

            elif operator == "s1":
                offspring_c_decription = {}

            index1, offspring_c, rank = C1.Selection_code(self.q_value_m, self.Num, self.id, self.pop_code, self.code_database, offspring_c_decription, self.scores_dic, self.total_time_solots, operator, candidate_fit)

            self.NFEs, candidate_fit = self.run_action(index1, offspring_c, offspring_c_decription, operator, rank)
            self.id += 1

            if self.NFEs >= 1000:
                break

        # Save population to a file
        main_folder = "./CoE_SAEA_LLM_Outputs/pops"
        subfolder = os.path.join(main_folder, f"Test_Problems_{self.fun_name}_dim_{self.dim}")
        filename = os.path.join(subfolder, f"Configuration_strategy_population_Fun_{self.fun_name}_dim_{self.dim}_run{str(self.runs)}.json")
        filename2 = os.path.join(subfolder, f"Prior_knowledge_population_Fun_{self.fun_name}_dim_{self.dim}_run{str(self.runs)}.json")
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'w') as f:
            json.dump(self.code_database, f, indent=5)
        directory2 = os.path.dirname(filename2)

        if not os.path.exists(directory2):
            os.makedirs(directory2)

        with open(filename2, 'w') as f:
            json.dump(self.code_database_prompt, f, indent=5)


        return self.hf, self.maxFEs, self.gfs

