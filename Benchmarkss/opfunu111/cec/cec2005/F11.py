#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:54, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import sum, dot, cos, pi
import numpy as np
from numpy import linalg as LA

class Model(Root):
    def __init__(self, f_name="Shifted Rotated Weierstrass Function", f_shift_data_file="data_weierstrass",
                 f_ext='.txt', f_bias=90, f_matrix=None, a=0.5, b=3, k_max=20):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix
        self.a = a
        self.b = b
        self.k_max = k_max

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
            shift_data = np.array(-0.5 + 0.5 * np.random.random((1, dim)))
        else:
            shift_data = self.load_shift_data()[:dim]
        if dim == 10 or dim == 30 or dim == 50:
            self.f_matrix = "weierstrass_M_D" + str(dim)
            matrix = self.load_matrix_data(self.f_matrix)
        else:
            c = 5
            matrix = rot_matrix(dim, c)
        z = dot((solution - shift_data), matrix)
        result = 0.0
        for i in range(0, dim):
            result += np.sum([self.a ** k * cos(2 * pi * self.b ** k * (z[:,i] + 0.5)) for k in range(0, self.k_max)],axis=0)
        result -= dim * np.sum([self.a ** k * cos(2 * pi * self.b ** k * 0.5) for k in range(0, self.k_max)],axis=0)
        return result + self.f_bias

