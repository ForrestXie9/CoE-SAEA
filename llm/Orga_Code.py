import numpy as np
import time
import importlib
from llm.LLM_Configuration_fun import Generate_Functions


class Orga_Code(object):

    def __init__(self,  hf, num_arm, ghf, q_value_m, Num, id, maxFEs, popsize, pop_code, code_database, pop_code_size, debug, offspring_c_decription, scores_dic, total_time_solots, paras):

        self.num_arm = num_arm
        self.ghf = ghf
        self.q_value_m = q_value_m
        self.Num = Num
        self.id = id
        self.maxFEs = maxFEs
        self.popsize = popsize
        self.debug = debug
        self.paras = paras
        self.total_time_solots = self.maxFEs - self.popsize
        self.pop_code = pop_code
        self.pop_code_size = pop_code_size
        self.offspring_c_decription = offspring_c_decription
        self.scores_dic = scores_dic
        self.code_database = code_database
        self.hf = hf
        # self.total_time_solots = total_time_solots

    def operator_choice(self, candidate_fit = None):
        if candidate_fit == None:
            candidate_fit = np.max(self.hf)

        if len(self.pop_code) < self.pop_code_size:
            operator = "i1"
        else:
            N = len(self.ghf)
            ghf = np.reshape(self.ghf, (1, N))
            ghf_sum = np.concatenate((self.ghf.reshape(1, -1), candidate_fit.reshape(1, -1)), axis=1)
            index = np.argsort(ghf_sum)
            in_ = (np.where(index == N)[1][0]) + 1
            p1 = (N+1-in_) / N
            if np.random.rand() < p1:
                operator = "s1"
            else:
                operator = "e1"

        return operator

    def generate_arm_index(self, operator):


        if operator == "e1" or operator == "i1":
            try:
                index, offspring_c, rank = self.get_code(operator)
                index = index + 1

            except Exception as e:
                print(f"Generated Algorithm Error: {e}. Retrying...")
                index = 100 # Output an invalid index
                offspring_c = {}
                rank = []
        elif operator == "s1":
            offspring_c = {}
            index, rank = self.parent_selection2(operator)
            print(index)

            index = index + 1

        return index, offspring_c, rank

    def get_code(self, operator):

        index = 100

        if operator == "e1" or operator == "i1":
            while not isinstance(index, int) or index < 0 or index >= self.num_arm:
                # print(index)
                offspring_c, rank = self._get_code(operator)
                if operator == "e1" or operator == "i1":
                    while self.check_duplicate(self.pop_code, offspring_c['code']):
                        if self.debug:
                            print("duplicated code, wait 1 second and retrying ... ")
                        offspring_c, rank = self._get_code(operator)
                time.sleep(1)
                self.code2file(offspring_c['code'])
                # try:
                score_set = np.array(self.q_value_m)
                self.Num = np.array(self.Num)
                total_num = np.sum(self.Num)
                action_set = np.arange(0, self.num_arm)

                heuristic_module = importlib.import_module("select_fun")
                eva = importlib.reload(heuristic_module)

                index = eva.selection(self.scores_dic, total_num, self.id, self.total_time_solots)
                index = int(index)


        elif operator == "s1":
            try:
                offspring_c = {}
                index, rank = self.parent_selection2(operator)
            except Exception as e:
                print(f"Generated Algorithm Error: {e}. Retrying...")
                index = 100
                offspring_c = {}
                rank = []

        return index, offspring_c, rank


    def parent_selection2(self, operator):
        self.population_management()
        index_list = []
        reward_list = []
        pop_code_copy = self.pop_code[:]

        for i in range(len(pop_code_copy)):
            offspring_c = pop_code_copy[i]

            self.code2file(offspring_c['code'])

            self.Num = np.array(self.Num)

            total_num = np.sum(self.Num)

            # Dynamically import and reload the heuristic module for each iteration
            heuristic_module = importlib.import_module("select_fun")
            eva = importlib.reload(heuristic_module)

            try:

                index = eva.selection(self.scores_dic, total_num, total_num, self.total_time_solots)
                index = int(index)
                index_list.append(index)  # If the selection is successful, append the index
            except Exception as e:
                # Handle the error, print a message, and remove the problematic code from pop_code
                print(f"Error in eva.selection at index {i}. Error: {e}")

                self.pop_code[i]['objective'] = 100 * np.max(self.hf)

        # print(index_list)
        selected_index = int(np.random.randint(0, len(index_list) - 1))

        selected_element = int(index_list[selected_index])

        return selected_element, selected_index

    def _get_code(self, operator):
        rank = []
        offspring_c = {}
        if operator == "i1":
            se = Generate_Functions(self.q_value_m, self.Num, self.id, self.maxFEs - self.popsize, operator, None, None, self.pop_code_size, len(self.pop_code), self.offspring_c_decription, self.debug, self.paras)
            offspring_c = {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None}
            offspring_c['code'] = se.Read_code()

        elif operator == "e1":
            parents, rank = self.parent_selection(operator)
            se = Generate_Functions(self.q_value_m, self.Num, self.id, self.maxFEs - self.popsize, operator, parents, rank, self.pop_code_size, len(self.pop_code), self.offspring_c_decription, self.debug, self.paras)
            offspring_c = {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None}
            offspring_c['code'] = se.Read_code()

        return offspring_c, rank



    def code2file(self,code):
        with open("./select_fun.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return

    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def parent_selection(self, operator):
        self.population_management()
        ranks = [i for i in range(len(self.pop_code))]
        probs = [1 / (rank + 1) for rank in ranks]
        total_sum = sum(probs)

        normalized_probs = [prob / total_sum for prob in probs]
        cumulative_probs = [sum(normalized_probs[:i + 1]) for i in range(len(normalized_probs))]
        rand_value = np.random.rand()
        selected_index = next(idx for idx, cum_prob in enumerate(cumulative_probs) if rand_value < cum_prob)
        parents = self.pop_code[selected_index]


        return parents, selected_index

    def population_management(self):
        # Delete the worst individual

        self.pop_code = sorted(self.pop_code, key=lambda x: x['objective'])

        if len(self.pop_code) > self.pop_code_size:
            self.pop_code.pop()


        return self.pop_code


