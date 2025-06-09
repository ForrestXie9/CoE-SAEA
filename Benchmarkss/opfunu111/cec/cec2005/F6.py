#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:21, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
import numpy as np

class Model(Root):
    def __init__(self, f_name="Shifted Rosenbrock's Function", f_shift_data_file="data_rosenbrock",
                 f_ext='.txt', f_bias=390):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def run(self, solution=None):
        problem_size, dim = np.shape(solution)
        if dim > 100:
            shift_data = np.array(-90 + 180 * np.random.random((1, dim)))
        else:
            shift_data = self.load_shift_data()[:dim]
        z = solution - shift_data + 1
        result = 0
        for i in range(0, dim-1):
             result += (100 * (z[:,i] ** 2 - z[:, i + 1]) ** 2 + (z[:, i] - 1) ** 2)
            # f = np.sum(100. * (z[:, 1:dim - 1] ** 2 - z[:, 2: dim]) ** 2 + (z[:, 1:dim - 1] - 1) ** 2, axis=1)
        return result + self.f_bias

