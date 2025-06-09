import http.client
import json

class InterfaceAPI2D():

    def __init__(self,key,model_LLM,debug_mode):
        self.key = key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
   
    def get_response(self,prompt_content):


        payload_explanation = json.dumps({
        "model": self.model_LLM,
        #"model": "gpt-4-0613",
        "messages": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "safe_mode": False
        })
        headers = {
        'Authorization': 'Bearer '+self.key,
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        'x-api2d-no-cache': 1
        }
      
        while True:
            try:
                conn = http.client.HTTPSConnection("oa.ai01.org")
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data['choices'][0]['message']['content']
                break
            except :
                print("Error in API. Restarting the process...")
                continue

        # code = re.findall(r"import.*return", response, re.DOTALL)
        # algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        # if code == None:
        #     code = re.findall(r"def.*return", response, re.DOTALL)

        # while(len(code)<1 or len(algorithm)<1):    # aviod outline response that doest not contain any solution (i.e., responses without <trace>(.*?)</trace>)
        
        #     print(" warning ! retrying ...")

        #     while True:
        #         try:
        #             conn = http.client.HTTPSConnection("oa.api2d.net")
        #             conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
        #             res = conn.getresponse()
        #             data = res.read()
        #             json_data = json.loads(data)
        #             response = json_data['choices'][0]['message']['content']
        #             break
        #         except:
        #             print("Error in API. Restarting the process...")
        #             continue

        #     code = re.findall(r"import.*return", response, re.DOTALL)
        #     algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        #     if code == None:
        #         code = re.findall(r"def.*return", response, re.DOTALL)


        # if self.prompt_func_name not in code[0]:
        #     code = re.findall(r"def.*return", response, re.DOTALL)
        # code = code[0]
        # algorithm = algorithm[0]

        # code_all = code+" "+", ".join(s for s in self.prompt_func_outputs) 

        # if self.debug_mode:
        #     print("check response: ",response)
        #     print("check created algorithm: ",algorithm)
        #     print("check created code: ",code_all)
        #     input()

        return response
    
