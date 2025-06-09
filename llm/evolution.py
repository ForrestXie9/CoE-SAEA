import re
import time
from llm.ael_prompts import GetPrompts
from llm.interface_LLM import InterfaceLLM
import numpy as np
class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, **kwargs):
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # # -------------------- RZ: use local LLM --------------------
        # assert 'use_local_llm' in kwargs
        # assert 'url' in kwargs
        # self._use_local_llm = kwargs.get('use_local_llm')
        # self._url = kwargs.get('url')
        # # -----------------------------------------------------------

        # set prompt interface
        getprompts = GetPrompts()
        self.prompt_task = getprompts.get_task()
        self.prompt_func_name = getprompts.get_func_name()
        self.prompt_func_inputs = getprompts.get_func_inputs()
        self.prompt_func_outputs = getprompts.get_func_outputs()
        self.prompt_inout_inf = getprompts.get_inout_inf()
        self.prompt_other_inf = getprompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = False # close prompt checking

        # -------------------- RZ: use local LLM --------------------

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def get_prompt_g1(self, x, obj_p, lb, ub):
        # Convert given individuals to the desired string format
        pop_content = " "
        # x = x.tolist()
        a, b = np.shape(x)
        for i in range(a):
            # pop_content+="point: <start>"+",".join(str(idx) for idx in x[i].tolist())+"<end> \n value: "+str(y[i])+" objective 1: "+str(obj_p[i][0])+" objective 2: "+str(obj_p[i][1])+"\n\n"
            pop_content += "point: <start>" + ",".join(
                str(idx) for idx in x[i,:].tolist()) + "<end> \n function value: " + str(
                round(obj_p[i], 4))  + "\n\n"

        # prompt_content = "Now you will help me minimize a function with " + str(b) + " variables. The search space for each variable ranges from " + str(lb) + " to "+ str(ub) + \
        #                                                                                                                                                                 ". I have some points with their function values. The points start with <start> and end with <end>.\n\n" \
        #                  + pop_content \
        #                  + "Give me " + str(a) + " new points that are different from all points above. The function value of each new point should be lower than any of the existing points." \
        #                                          " Do not write code. Do not give any explanation." \
        #                                          " Each output new point must start with <start> and end with <end>"

        prompt_content = "Now you will help me minimize an objective with " + str(b) + " variables. The search space for each variable ranges from " + str(lb) + " to "+ str(ub) + ". " \
                         "I have some points with their function values. The points start with <start> and end with <end>.\n\n" \
                         + pop_content + "Please give me " + str(20) + " new points that are different from all points above. " \
                                                                      " Do not write code. Do not give any explanation." \
                                                                      " Your response must only contain new points." \
                                                                      " Each output new point must start with <start> and end with <end>"
        # prompt_content = ""
        # prompt_content = "Now you will help me minimize " + str(len(obj_p[0])) + " objectives with " + str(len(x[
        #                                                                                                            0])) + " variables. I have some points with their objective values. The points start with <start> and end with <end>.\n\n" \
        #                  + pop_content \
        #                  + "Give me two new points that are different from all points above, and not dominated by any of the above. Do not write code. Do not give any explanation. Each output new point must start with <start> and end with <end>"
        # return prompt_content

        return prompt_content

    def _get_alg(self,prompt_content):

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if code == None:
            code = re.findall(r"def.*return", response, re.DOTALL)

        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")
            time.sleep(1)

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if code == None:
                code = re.findall(r"def.*return", response, re.DOTALL)

        algorithm = algorithm[0]
        code = code[0] 

        code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 


        return [code_all, algorithm]

    def _get_alg2(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)
        # algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        off = re.findall(r"<start>(.*?)<end>", response)
        off1 = np.array([np.fromstring(s, sep=',') for s in off])
        return off1



    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e1(self,parents):
      
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def g1(self, x, obj_p, lb, ub):
        prompt_content = self.get_prompt_g1(x, obj_p, lb, ub)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        off1 = self._get_alg2(prompt_content)

        if self.debug_mode:
            # print("\n >>> check designed algorithm: \n", algorithm)
            # print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return off1