#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import sum, dot, cos, sqrt
import numpy as np
from numpy import linalg as LA
from numpy.random import normal

class Model(Root):
    def __init__(self, f_name="Shifted Rotated Griewank's Function without Bounds", f_shift_data_file="data_griewank",
                 f_ext='.txt', f_bias=-180, f_matrix=None):
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
            shift_data = np.array(-600 + 0 * np.random.random((1, dim)))
        else:
            shift_data = self.load_shift_data()[:dim]
        if dim == 10 or dim == 30 or dim == 50:
            self.f_matrix = "griewank_M_D" + str(dim)
            matrix = self.load_matrix_data(self.f_matrix)
        else:
            c = 3
            M = rot_matrix(dim, c)
            matrix = M * (1 + 0.3* normal(0, 1, (dim, dim)))
        z = dot((solution - shift_data), matrix)
        vt1 = sum(z**2) / 4000 + 1
        result = 1.0
        for i in range(0, dim):
            result *= cos(z[:, i] / sqrt(i+1))
        return vt1 - result + self.f_bias

