#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:48, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import cos, pi
import numpy as np
from numpy import linalg as LA
class Model(Root):
    def __init__(self, f_name="Shifted Rastrigin's Function", f_shift_data_file="data_rastrigin",
                 f_ext='.txt', f_bias=-330):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

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
        c = 2
        M = rot_matrix(dim, c)
        z = np.dot(solution - shift_data, M)

        f = np.sum(z ** 2 - 10 * cos(2 * pi * z) + 10, axis=1)
        return f + self.f_bias

