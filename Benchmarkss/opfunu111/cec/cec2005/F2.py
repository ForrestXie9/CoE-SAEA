#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:07, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import sum
import numpy as np

class Model(Root):
    def __init__(self, f_name="Shifted Schwefel's Problem 1.2", f_shift_data_file="data_schwefel_102", f_ext='.txt', f_bias=-450):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def run(self, solution=None):
        problem_size, dim = np.shape(solution)
        if dim > 100:
            shift_data = np.array(-100 + 200 * np.random.random((1, dim)))
            shift_data = shift_data[:dim]
        else:
            shift_data = self.load_shift_data()[:dim]
        result = 0
        z = solution - shift_data
        # shift_data = np.tile(shift_data,(problem_size,1))
        for i in range(0, dim):
            result += (np.sum(z[:, 0:i], axis=1)) ** 2
            # result = result + (np.sum(solution[:, 0:i] - (shift_data[:i]), axis=1))**2
            # y += (np.sum(z[:, 0:i], axis=1)) ** 2
        return result + self.f_bias


