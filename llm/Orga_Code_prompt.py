import numpy as np
import time
import importlib

from llm.LLM_generate_prompt import Generate_Knowledge



class Orga_Code_prompt(object):

    def __init__(self,  num_arm, ghf, q_value_m, Num, id, maxFEs, popsize, pop_code, pop_code_size, debug, paras):

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


    def operator_choice(self, candidate_fit = None):

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

    def generate_prompt(self, operator):

        try:
            offspring_c = self.get_code(operator)
        except Exception as e:

            print(f"Generated Prompt Error: {e}. Retrying...")

            offspring_c = {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None}

        return offspring_c


    def get_code(self, operator):
        offspring_c = self._get_code(operator)
        if operator == "e1" or operator == "i1":
            while self.check_duplicate(self.pop_code, offspring_c['algorithm']):
                  if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                  offspring_c = self._get_code(operator)

        time.sleep(1)

        return offspring_c


    def _get_code(self, operator):
        offspring_c = {'algorithm': None, 'code': None, 'objective': None, 'other_inf': None}
        if operator == "i1":
            se = Generate_Knowledge(self.q_value_m, self.Num, self.id, self.maxFEs - self.popsize, operator, None, None, self.pop_code_size, len(self.pop_code), self.debug, self.paras)
            offspring_c['algorithm'] = se.Read_code()

        elif operator == "e1":
            parents, rank = self.parent_selection(operator)
            se = Generate_Knowledge(self.q_value_m, self.Num, self.id, self.maxFEs - self.popsize, operator, parents, rank, self.pop_code_size, len(self.pop_code), self.debug, self.paras)
            offspring_c['algorithm'] = se.Read_code()

        return offspring_c


    def add2pop(self,population,offspring):

        population.append(offspring)
        return True

    def add2pop2(self, population, offspring):
        population.append(offspring)
        return True

    def update(self, index, reward, offspring_c, candidate_fit):
        index2 = index - 1
        average_re = self.q_value_m[index2]
        reward = reward
        new_re = (average_re * self.Num[index2] + reward) / (self.Num[index2] + 1)
        self.q_value_m[index2] = new_re
        self.Num[index2] = self.Num[index2] + 1
        fitness = reward
        offspring_c['objective'] = candidate_fit

        self.add2pop(self.pop_code, offspring_c)
        self.add2pop2(self.code_database, offspring_c)


    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['algorithm']:
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

