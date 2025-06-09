#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:17, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%
import numpy as np
from numpy import linalg
from numpy.random import normal
from opfunu111.cec.cec2005.root import Root
from numpy import dot


class Model(Root):
    def __init__(self, f_name="Shifted Rotated High Conditioned Elliptic Function", f_shift_data_file="data_high_cond_elliptic_rot",
                 f_ext='.txt', f_bias=-450, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)
        self.f_matrix = f_matrix

    def run(self, solution):

        problem_size, dim = np.shape(solution)
        if dim > 100:
            shift_data = np.array(-100 + 200 * np.random.random((1, dim)))
        else:
            shift_data = self.load_shift_data()[:dim]
        if dim == 2 or dim == 10 or dim == 30 or dim == 50:
            self.f_matrix = "elliptic_M_D" + str(dim)
            matrix = self.load_matrix_data(self.f_matrix)
        else:
            A = normal(0, 1, (dim, dim))
            [matrix, r] = cgs(A)

        z = (dot((solution - shift_data), matrix))**2
        result = 0
        for i in range(0, dim):
            result += result + (10**6) ** (i / (dim - 1)) * z[:, i]**2
        return result + self.f_bias

def cgs(A):
    """Classical Gram-Schmidt (CGS) algorithm"""
    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.empty((m, n))
    R[0, 0] = linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0, 0]
    for k in range(1, n):
        R[:k-1, k] = np.dot(Q[:m, :k-1].T, A[:m, k])
        z = A[:m, k] - np.dot(Q[:m, :k-1], R[:k-1, k])
        R[k, k] = linalg.norm(z) ** 2
        Q[:m, k] = z / R[k, k]
    return Q, R

