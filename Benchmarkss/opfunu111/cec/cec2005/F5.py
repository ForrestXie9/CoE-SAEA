#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:50, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import dot, max, abs, array,round
from pandas import read_csv
from numpy.linalg import *
import numpy as np

class Model(Root):
    def __init__(self, f_name="Schwefel's Problem 2.6 with Global Optimum on Bounds", f_shift_data_file="data_schwefel_206",
                 f_ext='.txt', f_bias=-310):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def load_shift_data(self):
        data = read_csv(self.support_path_data + self.f_shift_data_file + self.f_ext, delimiter='\s+', index_col=False, header=None)
        data = data.values
        shift_data = data[:1, :]
        matrix_data = data[1:, :]
        return shift_data, matrix_data

    def run(self, solution=None):
        problem_size, dim = np.shape(solution)
        if dim > 100:
            shift_data = np.array(-100 + 200 * np.random.random((1, dim)))
            matrix_data = round(-100 + 2 * 100. * np.random.random((dim, dim)))
            while np.linalg.det(matrix_data) == 0:
                  matrix_data = round(-100 + 2 * 100. * np.random.random((dim, dim)))
        else:
            shift_data, matrix_data = self.load_shift_data()
        shift_data = shift_data.reshape(-1)[:dim]
        matrix_data = matrix_data[:dim, :dim]
        t1 = int(0.25 * dim) + 1
        t2 = int(0.75 * dim)
        shift_data[:t1] = -100
        shift_data[t2:] = 100
        # B = A * o'
        B = dot(matrix_data, shift_data.reshape(dim,1))
        result = np.zeros((problem_size, ))
        for i in range(0, problem_size):
            solution1 = np.transpose(solution)
            solution2 = solution1[:,i]
            z = dot(matrix_data, solution2)
            result[i,] = np.max(abs(z-B))
        return result + self.f_bias



