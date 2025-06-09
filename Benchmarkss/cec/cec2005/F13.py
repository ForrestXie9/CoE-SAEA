#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:18, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import cos
import numpy as np

class Model(Root):
    def __init__(self, f_name="Shifted Expanded Griewank's plus Rosenbrock's Function (F8F2)", f_shift_data_file="data_EF8F2",
                 f_ext='.txt', f_bias=-130):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def __f8__(self, x=None):
        return x ** 2 / 4000 - cos(x) + 1

    def __f2__(self, x=None):
        a, b = np.shape(x)
        x = np.reshape(x, (a, b))
        f2 = 100 * (x[:, 0] ** 2 - x[:, 1]) ** 2 + (x[:, 0] - 1) ** 2
        f = 1 + f2 ** 2 / 4000 - cos(f2)
        return f


    def run(self, solution=None):
        problem_size, dim = np.shape(solution)
        if dim > 100:
            shift_data = np.array(-1 + 1 * np.random.random((1, dim)))
        else:
            shift_data = self.load_shift_data()[:dim]
        z = solution - shift_data + 1
        result = 0
        for i in range(0, dim):
            if i == dim - 1:
                a = np.array(z[:, i]).reshape(problem_size,1)
                b = np.array(z[:, 0]).reshape(problem_size,1)
                c = np.append(a, b, axis=1)
                result += self.__f2__(c)
            else:
                result += self.__f2__(z[:, i:i + 2])

        return result + self.f_bias


