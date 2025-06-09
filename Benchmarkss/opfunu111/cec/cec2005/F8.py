#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:39, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import sum, dot, cos, exp, pi, e, sqrt
import numpy as np
from numpy import linalg as LA

class Model(Root):
    def __init__(self, f_name="Shifted Rotated Ackley's Function with Global Optimum on Bounds", f_shift_data_file="data_ackley",
                 f_ext='.txt', f_bias=-140, f_matrix=None):
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
            shift_data = np.array(-30 + 60 * np.random.random((1, dim)))
            shift_data = shift_data.reshape(-1)[:dim]
        else:
            shift_data = self.load_shift_data()[:dim]
        if dim == 10 or dim == 30 or dim == 50:
            self.f_matrix = "ackley_M_D" + str(dim)
            matrix = self.load_matrix_data(self.f_matrix)
        else:
            c = 100
            matrix = rot_matrix(dim, c)
        t1 = int(dim/2)
        for j in range(0, t1-1):
            shift_data[2*(j+1)-1] = -32 * shift_data[2*(j+1)]
        z = dot((solution - shift_data), matrix)
        # f = sum(x. ^ 2, 2);
        # f = 20 - 20. * exp(-0.2. * sqrt(f. / D)) - exp(sum(cos(2. * pi. * x), 2). / D) + exp(1);
        result = np.sum(z ** 2, axis=1)
        result = -20 * exp(-0.2 * sqrt(result / dim)) - exp(np.sum(cos(2 * pi * z), axis=1)/dim) + 20 + e
        return result + self.f_bias





