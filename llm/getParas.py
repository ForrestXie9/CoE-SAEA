
class Paras():
    def __init__(self):
        #####################
        ### LLM settings  ###
        #####################
        self.llm_use_local = False  # if use local model
        self.llm_local_url = None  # your local server 'http://127.0.0.1:11012/completions'
        self.llm_api_endpoint = "oa.api2d.site"
        self.llm_api_key = "your key"  # use your key
        self.llm_model = "gpt-3.5-turbo-1106"

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 3  # number of populations, default = 10
        self.ec_operators = ['e1','e2','m1','m2']  # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = [1, 1, 1, 1]  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./ael_seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./ael_results/pops/population_generation_0.json"

    def set_paras(self, *args, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)



if __name__ == "__main__":

    # Create an instance of the Paras class
    paras_instance = Paras()

    # Setting parameters using the set_paras method
    paras_instance.set_paras(llm_use_local=True, llm_local_url='http://example.com', ec_pop_size=8)

    # Accessing the updated parameters
    print(paras_instance.llm_use_local)  # Output: True
    print(paras_instance.llm_local_url)  # Output: http://example.com
    print(paras_instance.ec_pop_size)    # Output: 8
            
            
            
