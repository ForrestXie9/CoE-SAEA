import numpy as np
import time
from llm.getParas  import Paras
from Benchmarkss.variable_domain import variable_domain
import os
import opfunu_v3
from CoE_SAEA import CoE_SAEA

def RUN_CoE_SAEA(MaxFEs, runs, D, FUN, fun_name, LB, UB, f_bias, debug, paras):
    time_begin = time.time()
    np.seterr(all='ignore')

    gsamp1 = np.zeros((runs, MaxFEs))

    for r in range(runs):
        # main loop
        print('\n')
        print('FUNCTION:', fun_name, 'RUN:', r + 1)
        print('\n')
        opt = CoE_SAEA(r, fun_name, MaxFEs, FUN, D, LB, UB, debug, paras)
        hisf, mf, gfs = opt.run()
        print('Best fitness:', min(hisf))
        gsamp1[r, :] = gfs[:mf]

    # obtain output results
    samp_mean = np.mean(gsamp1[:, -1])
    samp_mean_error = samp_mean - f_bias
    std_samp = np.std(gsamp1[:, -1])

    os.makedirs('./result', exist_ok=True)
    time_cost = time.time() - time_begin
    last_value = [samp_mean, samp_mean_error, std_samp, time_cost]
    np.savetxt(f"result/NFE{mf}_{fun_name}_runs={runs}_Dim={D}.txt", gsamp1)
    np.savetxt(f"./result/{fun_name}_runs={runs}_Dim={D}.txt", last_value)


if __name__ == "__main__":

    TestFuns = ['F12005', 'F22005', 'F32005', 'F42005', 'F52005', 'F62005', 'F72005',
                'F82005', 'F92005', 'F102005']

    TestFuns2 = ['F12015', 'F22015', 'F32015', 'F42015', 'F52015', 'F62015', 'F72015',
                 'F82015', 'F92015', 'F102015']
    ####llm
    paras = Paras()
    # Set parameters #
    paras.set_paras(llm_api_endpoint="**********",  #add your api_endpoint
                    llm_api_key="**********", #add your key
                    llm_model="**********",  #choose llm model
                    exp_debug_mode=False)

    #####
    dims = [10, 30]  # Dimensions
    Runs = 20  # Number of runs
    MaxFEs = 1000
    debug = True

    d = len(dims)
    o = len(TestFuns) + len(TestFuns2)  # Two benchmark sets

    f_bias_set = [-450, -450, -450, -450, -310, 390, -180, -140, -330, -330]
    f_bias_set2 = [i * 100 for i in range(1, len(TestFuns2)+1)]

    for i in range(1, 2):
        for j in range(0, 10):
            if j < len(TestFuns):  # first test problems
                f_bias = f_bias_set[j]
                fun_name = TestFuns[j]
                Xmin, Xmax = variable_domain(fun_name)
            else:
                problem_index = j - len(TestFuns)   # second test problems
                f_bias = f_bias_set2[problem_index]
                fun_name = TestFuns2[problem_index]
                Xmin = -100
                Xmax = 100
            LB = [Xmin] * dims[i]
            UB = [Xmax] * dims[i]

            funcs = opfunu_v3.get_functions_by_classname(fun_name)
            FUN = funcs[0](ndim=dims[i])


            RUN_CoE_SAEA(MaxFEs, Runs, dims[i], FUN.evaluate, fun_name, LB, UB, f_bias, debug, paras)


