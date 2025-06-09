#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:52, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import dot, sin, sqrt
import numpy as np
from numpy import linalg as LA

class Model(Root):
    def __init__(self, f_name="Shifted Rotated Expanded Scaffer's F6 Function", f_shift_data_file="data_E_ScafferF6",
                 f_ext='.txt', f_bias=-300, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def __fxy__(self, x=None, y=None):
        return 0.5 + (sin(sqrt(x**2+y**2))**2 - 0.5) / (1 + 0.001* (x**2 + y**2))**2

    def run(self, solution=None):
        def rot_matrix(D, c):
            A = np.random.normal(0, 1, (D, D))
            P, _ = LA.qr(A)
            A = np.random.normal(0, 1, (D, D))
            Q, _ = LA.qr(A)
            u = np.random.random((1, D))
            D = np.power(c, (u - np.min(u)) / (np.max(u) - np.min(u)))
            D = np.squeeze(D)
            D = np.diag(D)
            M = np.dot(np.dot(P, D), Q)
            return M

        problem_size, dim = np.shape(solution)
        if dim > 100:
            shift_data = np.array(-100 + 200 * np.random.random((1, dim)))
        else:
            shift_data = self.load_shift_data()[:dim]
        if dim == 10 or dim == 30 or dim == 50:
            self.f_matrix = "E_ScafferF6_M_D" + str(dim)
            matrix = self.load_matrix_data(self.f_matrix)
        else:
            c = 3
            matrix = rot_matrix(dim, c)
        z = dot((solution - shift_data), matrix)
        result = 0
        for i in range(0, dim):
            if i == dim - 1:
                a = np.array(z[:, i]).reshape(problem_size, 1)
                b = np.array(z[:, 0]).reshape(problem_size, 1)
                result += self.__fxy__(a, b)
            else:
                a = np.array(z[:, i]).reshape(problem_size, 1)
                b = np.array(z[:, i+1]).reshape(problem_size, 1)
                result += self.__fxy__(a, b)
        # return np.reshape(result + self.f_bias, problem_size)

        return result + self.f_bias


