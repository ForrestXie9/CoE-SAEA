#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import sum
import numpy as np


class Model(Root):
    def __init__(self, f_name="Shifted Sphere Function", f_shift_data_file="data_sphere", f_ext='.txt', f_bias=-450):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def run(self, solution=None):
        problem_size,dim = np.shape(solution)
        if dim > 100:
            shift_data = -100 + 200 * np.random.random((1, dim))
        else:
            shift_data = self.load_shift_data()[:dim]
        result = np.sum((solution - shift_data)**2, axis=1) + self.f_bias
        # y = np.sum(np.square(z), axis=1) - 450
        return result











