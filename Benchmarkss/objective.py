import numpy as np
import opfunu

from opfunu.cec_based.cec2015 import F12015,F22015,F32015,F42015,F52015,F62015,F72015,F82015,F92015,F102015,F112015,F122015,F132015,F142015,F152015

# CEC 15的 15个function分别是
# F12015, F22015, F32015, F42015, F52015,
# F62015, F72015, F82015, F92015, F102015,
# F112015, F122015, F132015, F142015, F152015
class CEC_2015_functions():
    def __init__(self, dim=10, func_name = ''):
        self.dim = dim
        self.bounds = [(-100.0, 100.0)] * dim
        self.name = func_name + '_' + str(dim)
        # CEC 15的 15个function分别是
        # F12015, F22015, F32015, F42015, F52015, F62015, F72015, F82015, F92015, F102015, F112015, F122015, F132015, F142015, F152015
        funcs = opfunu.get_functions_by_classname(func_name)
        self.func = funcs[0](ndim=dim)


    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        X = X.ravel() # shape: [dim]
        if X.shape[0] !=  self.dim:
            assert False
        else:
            fval = self.func.evaluate(X)
        return fval

class CEC_2005_functions():
    def __init__(self, dim=10, func_name = ''):
        #不错修改
        self.dim = dim
        self.bounds = [(-100.0, 100.0)] * dim
        self.name = func_name + '_' + str(dim)

        funcs = opfunu.get_functions_by_classname(func_name)
        self.func = funcs[0](ndim=dim)


    def get_bound(self):
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.bounds[i][0]
            ub[i] = self.bounds[i][1]
        return self.dim, lb, ub

    def evaluate(self, X):
        X = X.ravel() # shape: [dim]
        if X.shape[0] !=  self.dim:
            assert False
        else:
            fval = self.func.evaluate(X)
        return fval



