#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:07, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import dot, sin, cos,round
from pandas import read_csv
import numpy as np

class Model(Root):
    def __init__(self, f_name="Schwefel's Problem 2.13", f_shift_data_file="data_schwefel_213",
                 f_ext='.txt', f_bias=-460):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def load_shift_data(self):
        data = read_csv(self.support_path_data + self.f_shift_data_file + self.f_ext, delimiter='\s+', index_col=False, header=None)
        data = data.values
        a_matrix = data[:100, :]
        b_matrix = data[100:200, :]
        shift_data = data[200:, :]
        return shift_data, a_matrix, b_matrix

    def run(self, solution=None):
        problem_size, dim = np.shape(solution)
        if dim > 100:
            shift_data = np.array(-3 + 6 * np.random.random((1, dim)))
            a_matrix = round(-100 + 200 * np.random.random((dim, dim)))
            b_matrix = round(-100 + 200 * np.random.random((dim, dim)))

        else:
            shift_data, a_matrix, b_matrix = self.load_shift_data()
            shift_data = shift_data.reshape(-1)[:dim]
            a_matrix = a_matrix[:dim, :dim]
            b_matrix = b_matrix[:dim, :dim]
        result = np.zeros((problem_size,))
        shift_data = np.tile(shift_data,(dim, 1))
        A = np.sum(a_matrix * sin(shift_data) + b_matrix * cos(shift_data), axis=1)
        for i in range(0, problem_size):
            xx = np.tile(solution[i,:],(dim, 1))
            B = np.sum(a_matrix*sin(xx) + b_matrix * cos(xx), axis=1)
            C = (A-B)**2
            C = np.reshape(C, (1, dim))
            result[i,] = np.sum(C, axis=1)
        return result + self.f_bias

#     alpha = repmat(alpha, D, 1);
#     A = sum(a. * sin(alpha) + b. * cos(alpha), 2);
# for i=1:ps
#     xx=repmat(x(i,:),D,1);
#     B=sum(a.*sin(xx)+b.*cos(xx),2);
#     f(i,1)=sum((A-B).^2,1);
# end
