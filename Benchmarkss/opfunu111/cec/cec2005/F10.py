#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:51, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import dot, cos, pi
import numpy as np
from numpy import linalg as LA
class Model(Root):
    def __init__(self, f_name="Shifted Rotated Rastrigin's Function", f_shift_data_file="data_rastrigin",
                 f_ext='.txt', f_bias=-330, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

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
            shift_data = np.array(-5 + 10 * np.random.random((1, dim)))
        else:
            shift_data = self.load_shift_data()[:dim]
        if dim == 10 or dim == 30 or dim == 50:
            self.f_matrix = "rastrigin_M_D" + str(dim)
            matrix = self.load_matrix_data(self.f_matrix)
        else:
            c = 2
            matrix = rot_matrix(dim, c)
        z = dot((solution - shift_data), matrix)
        result = 0
        # for i in range(0, problem_size - 1):
        #     result += z[:,i] ** 2 - 10 * cos(2 * pi * z[:, i]) + 10
        result = np.sum((np.square(z) - 10 * np.cos(2 * np.pi * solution) + 10), axis=1) - 330
        return result


