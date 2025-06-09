#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:02, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu111.cec.cec2005.root import Root
from numpy import sum, dot, sqrt, array, cos, pi, exp, e, ones, identity, max
import numpy as np

class Model(Root):
    def __init__(self, f_name="Hybrid Composition Function 1", f_shift_data_file="data_hybrid_func1",
                 f_ext='.txt', f_bias=120):
        Root.__init__(self, f_name, f_shift_data_file, f_ext, f_bias)

    def __f12__(self, solution=None):
        return sum(solution**2 - 10*cos(2*pi*solution) + 10)

    def __f34__(self, solution=None, a=0.5, b=3, k_max=20):
        result = 0.0
        c = np.shape(solution)[1]
        for i in range(c):
            result += sum([a ** k * cos(2 * pi * b ** k * (solution + 0.5)) for k in range(0, k_max)])
        return result - np.shape(solution)[1] * sum([ a**k * cos(2*pi*b**k * 0.5) for k in range(0, k_max) ])

    def __f56__(self, solution=None):
        result = sum(solution**2) / 4000
        temp = 1.0
        for i in range(np.shape(solution)[1]):
            temp *= cos(solution[:, i] / sqrt(i+1))
        return result - temp + 1

    def __f78__(self, solution=None):
        return -20*exp(-0.2*sqrt(sum(solution**2)/np.shape(solution)[1])) - exp(sum(cos(2*pi*solution))/np.shape(solution)[1]) + 20 + e

    def __f910__(self, solution=None):
        return sum(solution ** 2)

    def __fi__(self, solution=None, idx=None):
        if idx == 0 or idx == 1:
            return self.__f12__(solution)
        elif idx == 2 or idx == 3:
            return self.__f34__(solution)
        elif idx == 4 or idx == 5:
            return self.__f56__(solution)
        elif idx == 6 or idx == 7:
            return self.__f78__(solution)
        else:
            return self.__f910__(solution)

    def run(self, solution=None):
        problem_size, dim = np.shape(solution)
        if dim > 100:
            shift_data = np.array(-90 + 180 * np.random.random((1, dim)))
        else:
            shift_data = self.load_matrix_data(self.f_shift_data_file)
            shift_data = shift_data[:problem_size, :dim]
        num_funcs = 10
        C = 2000
        xichma = ones(dim)
        lamda = array([1, 1, 10, 10, 5.0 / 60, 5.0 / 60, 5.0 / 32, 5.0 / 32, 5.0 / 100, 5.0 / 100])
        bias = array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        y = 5 * ones(dim)
        matrix = identity(dim)
        weights = ones(num_funcs)
        fits = np.array(ones(num_funcs))
        for i in range(3, num_funcs):
            shift_data1 = shift_data[:, i].reshape(problem_size, 1)
            w_i = exp(-sum((solution - shift_data1) ** 2)/(2*dim * xichma[i]**2))
            z = dot((solution - shift_data1) / lamda[i], matrix)
            # ddd = dot((y/lamda[i]), matrix)
            fit_i = self.__fi__(z, i)
            f_maxi = self.__fi__(dot((y/lamda[i]), matrix).reshape(1, dim), i)
            fit_i = C * fit_i / f_maxi
            weights[i] = w_i
            # fits[i] = np.array(fits[i])
            fits[i] = np.array(fit_i)
        sw = sum(weights)
        maxw = max(weights)

        for i in range(0, num_funcs):
            if weights[i] != maxw:
                weights[i] = weights[i] * (1 - maxw ** 10)
            weights[i] = weights[i] / sw

        fx = sum(dot(weights, (fits + bias)))
        return fx + self.f_bias

        for i in range(10):
            oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
            cur_subfunc = subfunctions[i]
            f = cur_subfunc(
                np.matmul((p - oo) / (np.repeat(lambd[i, :].reshape(1, D), p.shape[0], axis=0)), MD[i * D:(i + 1) * D, :]))
            fmax = cur_subfunc(np.matmul(tmp_y / lambd[i, :].reshape(1, D), MD[i * D:(i + 1) * D, :]))
            f1 = 2000 * f / fmax
            final_y = final_y + weight[:, i] * (f1 + bias[i])
        return final_y

