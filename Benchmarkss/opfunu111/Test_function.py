import numpy as np
from numpy import linalg
from numpy.random import normal
from numpy import sum, dot, cos, sqrt, e, pi, exp, sin
import math
from numpy import linalg as LA
import numpy as identity
# class test(Root):

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

def F1(solution):
    problem_size,dim = np.shape(solution)
    if dim > 100:
        shift_data = -100 + 200 * np.random.random((1, dim))
    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_sphere.txt")
    shift_data = shift_data[:dim]
    result = np.sum((solution - shift_data)**2, axis=1) - 450
    # y = np.sum(np.square(z), axis=1) - 450
    return result

def F2(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-100 + 200 * np.random.random((1, dim)))

    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_schwefel_102.txt")
    shift_data = shift_data[:dim]
    result = 0
    z = solution - shift_data
    for i in range(0, dim):
        result += (np.sum(z[:, 0:i], axis=1)) ** 2
    return result - 450

def F3(solution):

    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-100 + 200 * np.random.random((1, dim)))

    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_high_cond_elliptic_rot.txt")
    shift_data = shift_data[:dim]
    if dim == 10:
        matrix = np.loadtxt(
            fname="E:/support_data/elliptic_M_D10.txt")
    elif dim == 30:
        matrix = np.loadtxt(
            fname="E:/support_data/elliptic_M_D30.txt")
    elif dim == 50:
        matrix = np.loadtxt(
            fname="E:/support_data/elliptic_M_D50.txt")
    else:
        A = normal(0, 1, (dim, dim))
        [matrix, r] = cgs(A)
    z = (dot((solution - shift_data), matrix))**2
    result = 0
    for i in range(0, dim):
        result += result + (10**6) ** (i / (dim - 1)) * z[:, i]**2
    return result - 450

def F4(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-100 + 200 * np.random.random((1, dim)))
    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_schwefel_102.txt")
    result = 0
    z = solution - shift_data
    for i in range(0, dim):
        result += (np.sum(z[:, 0:i], axis=1)) ** 2
    result = result * (1 + 0.4 * abs(normal(0, 1))) - 450
    return result

def F5(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-100 + 200 * np.random.random((1, dim)))
        matrix_data = round(-100 + 2 * 100. * np.random.random((dim, dim)))
        while np.linalg.det(matrix_data) == 0:
            matrix_data = round(-100 + 2 * 100. * np.random.random((dim, dim)))
    else:

        data = np.loadtxt(
            fname="E:/support_data/data_schwefel_206.txt")
        shift_data = data[:1, :]
        matrix_data = data[1:, :]
    shift_data = shift_data.reshape(-1)[:dim]
    matrix_data = matrix_data[:dim, :dim]
    t1 = int(0.25 * dim) + 1
    t2 = int(0.75 * dim)
    shift_data[:t1] = -100
    shift_data[t2:] = 100
    B = dot(matrix_data, shift_data.reshape(dim, 1))
    result = np.zeros((problem_size,))
    for i in range(0, problem_size):
        solution1 = np.transpose(solution)
        solution2 = solution1[:, i]
        z = dot(matrix_data, solution2)
        result[i,] = np.max(abs(z - B))
    return result - 310

def F6(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-90 + 180 * np.random.random((1, dim)))
    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_rosenbrock.txt")
    shift_data = shift_data[:dim]
    z = solution - shift_data + 1
    result = 0
    for i in range(0, dim - 1):
        result += (100 * (z[:, i] ** 2 - z[:, i + 1]) ** 2 + (z[:, i] - 1) ** 2)
    return result + 390

def F8(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-30 + 60 * np.random.random((1, dim)))
        shift_data = shift_data.reshape(-1)[:dim]
    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_ackley.txt")
        shift_data = shift_data[:dim]
    if dim == 10:
        matrix = np.loadtxt(
            fname="E:/support_data/ackley_M_D10.txt")
    elif dim == 30:
        matrix = np.loadtxt(
            fname="E:/support_data/ackley_M_D30.txt")
    elif dim == 50:
        matrix = np.loadtxt(
            fname="E:/support_data/ackley_M_D50.txt")
    else:
        c = 100
        matrix = rot_matrix(dim, c)
    t1 = int(dim / 2)
    for j in range(0, t1 - 1):
        shift_data[2 * (j + 1) - 1] = -32 * shift_data[2 * (j + 1)]
    z = dot((solution - shift_data), matrix)
    result = np.sum(z ** 2, axis=1)
    result = -20 * exp(-0.2 * sqrt(result / dim)) - exp(np.sum(cos(2 * pi * z), axis=1) / dim) + 20 + e
    return result - 140

def F9(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-5 + 10 * np.random.random((1, dim)))
    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_rastrigin.txt")
    shift_data = shift_data[:dim]
    c = 2
    M = rot_matrix(dim, c)
    z = np.dot(solution - shift_data, M)
    f = np.sum(z ** 2 - 10 * cos(2 * pi * z) + 10, axis=1)
    return f - 330

def F10(p):  # 5 f13 D30ShiftedRotatedRastrigin
    problem_size, D = np.shape(p)
    if D <= 100:
        O_ori = np.loadtxt(
            fname="E:/support_data/data_rastrigin.txt")
    else:
        O_ori = -5 + 10 * np.random.random((1, D))
    if D == 10:
        MD30 = np.loadtxt(
            fname="E:/support_data/rastrigin_M_D10.txt")
    elif D == 30:
        MD30 = np.loadtxt(
            fname="E:/support_data/rastrigin_M_D30.txt")
    elif D == 50:
        MD30 = np.loadtxt(
            fname="E:/support_data/rastrigin_M_D50.txt")
    else:
        c = 2
        MD30 = rot_matrix(D, c)
    O_or = O_ori[0:D]
    O = O_or.reshape(1, D)
    z = np.dot((p - O), MD30)
    y = np.sum((np.square(z) - 10 * np.cos(2 * np.pi * p) + 10), axis=1) - 330
    return y

def F11(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-0.5 + 0.5 * np.random.random((1, dim)))
    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_weierstrass.txt")
    shift_data = shift_data[:dim]
    if dim == 10:
        matrix = np.loadtxt(
            fname="E:/support_data/weierstrass_M_D10.txt")
    elif dim == 30:
        matrix = np.loadtxt(
            fname="E:/support_data/weierstrass_M_D30.txt")
    elif dim == 50:
        matrix = np.loadtxt(
            fname="E:/support_data/weierstrass_M_D50.txt")
    else:
        c = 5
        matrix = rot_matrix(dim, c)
    z = dot((solution - shift_data), matrix)
    result = 0.0
    for i in range(0, dim):
        result += np.sum([0.5 ** k * cos(2 * pi * 3 ** k * (z[:, i] + 0.5)) for k in range(0, 20)],
                         axis=0)
    result -= dim * np.sum([0.5 ** k * cos(2 * pi * 3 ** k * 0.5) for k in range(0, 20)], axis=0)
    return result + 90

def F12(solution):

    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-3 + 6 * np.random.random((1, dim)))
        a_matrix = round(-100 + 200 * np.random.random((dim, dim)))
        b_matrix = round(-100 + 200 * np.random.random((dim, dim)))

    else:
        shift_data=  np.loadtxt(
            fname="E:/support_data/data_schwefel_213.txt")
        # shift_data = shift_data.reshape(-1)[:dim]
        a_matrix = shift_data[0:100, :]
        b_matrix = shift_data[100:200, :]
        a_matrix = a_matrix[:dim, :dim]
        b_matrix = b_matrix[:dim, :dim]
        shift_data = shift_data.reshape(-1)[:dim]
    result = np.zeros((problem_size,))
    shift_data = np.tile(shift_data, (dim, 1))
    A = np.sum(a_matrix * sin(shift_data) + b_matrix * cos(shift_data), axis=1)
    for i in range(0, problem_size):
        xx = np.tile(solution[i, :], (dim, 1))
        B = np.sum(a_matrix * sin(xx) + b_matrix * cos(xx), axis=1)
        C = (A - B) ** 2
        C = np.reshape(C, (1, dim))
        result[i,] = np.sum(C, axis=1)
    return result - 460

def f2(x):
    a, b = np.shape(x)
    x = np.reshape(x, (a, b))
    f2 = 100 * (x[:, 0] ** 2 - x[:, 1]) ** 2 + (x[:, 0] - 1) ** 2
    f = 1 + f2 ** 2 / 4000 - cos(f2)
    return f

def F13(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-1 + 1 * np.random.random((1, dim)))
    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_EF8F2.txt")
        shift_data = shift_data.reshape(-1)[:dim]
    z = solution - shift_data + 1
    result = 0
    for i in range(0, dim):
        if i == dim - 1:
            a = np.array(z[:, i]).reshape(problem_size, 1)
            b = np.array(z[:, 0]).reshape(problem_size, 1)
            c = np.append(a, b, axis=1)
            result += f2(c)
        else:
            result += f2(z[:, i:i + 2])
    return result - 130
def fxy(x, y):
    return 0.5 + (sin(sqrt(x**2+y**2))**2 - 0.5) / (1 + 0.001* (x**2 + y**2))**2
def F14(solution):
    problem_size, dim = np.shape(solution)
    if dim > 100:
        shift_data = np.array(-100 + 200 * np.random.random((1, dim)))
    else:
        shift_data = np.loadtxt(
            fname="E:/support_data/data_E_ScafferF6.txt")
    shift_data = shift_data.reshape(-1)[:dim]
    if dim == 10:
        matrix = np.loadtxt(
            fname="E:/support_data/E_ScafferF6_M_D10.txt")
    elif dim == 30:
        matrix = np.loadtxt(
            fname="E:/support_data/E_ScafferF6_M_D30.txt")
    elif dim == 50:
        matrix = np.loadtxt(
            fname="E:/support_data/E_ScafferF6_M_D50.txt")
    else:
        c = 3
        matrix = rot_matrix(dim, c)
    z = dot((solution - shift_data), matrix)
    result = 0
    for i in range(0, dim):
        if i == dim - 1:
            a = np.array(z[:, i]).reshape(problem_size, 1)
            b = np.array(z[:, 0]).reshape(problem_size, 1)
            result += fxy(a, b)
        else:
            a = np.array(z[:, i]).reshape(problem_size, 1)
            b = np.array(z[:, i + 1]).reshape(problem_size, 1)
            result += fxy(a, b)
    return result

def LAckley(p):  # 32.768 f12
    problem_size, D = np.shape(p)
    tmp1 = np.sqrt(np.sum(np.square(p), axis=1) / D)
    tmp2 = np.sum(np.cos(2 * np.pi * p), axis=1)
    y = -20 * np.exp(-0.2 * tmp1) - np.exp(tmp2 / D) + 20 + np.exp(1)
    return y


def LRastrigin(p):  # 5 f34
    y = np.sum((np.square(p) - 10 * np.cos(2 * np.pi * p) + 10), axis=1)
    return y


def LSphere(p):
    # f56
    y = np.sum(np.square(p), axis=1)
    return y

def LWeierstrass(p):  # f78
    ps, D = p.shape
    kmax = 20
    a = 0.5
    b = 3
    y = []
    for i in range(p.shape[0]):
        acc = 0
        for j in range(D):
            for k in range(kmax):
                acc += a ** k + math.cos(2 * math.pi * b ** k * (p[i, j] + 0.5))
                acc -= a ** k * math.cos(2 * math.pi * b ** k * 0.5)
        y.append(acc)
    return np.array(y)

def LGriewank(p):  # 600 f910
    problem_size, D = np.shape(p)
    tmp1 = np.sum(np.square(p), axis=1) / 4000
    tmp2 = np.cos(p / np.sqrt(np.arange(0, D) + 1))
    y = 1 + tmp1 - np.prod(tmp2, axis=1)
    return y

def F15(p):  # [-5, 5] f_bias = 120
    problem_size, D = np.shape(p)
    if D <= 100:
        O_ori = np.loadtxt(
            fname="E:/support_data/data_hybrid_func1.txt")
    else:
        O_ori = -5 + 10 * np.random.random((10, D))
    if D == 10:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func1_M_D10.txt")
    elif D == 30:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func1_M_D30.txt")
    elif D == 50:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func1_M_D50.txt")
    else:
        # MD = identity(D)
        MD = np.empty((10 * D, D))
        for i in range(10):
            MD[i * D:(i + 1) * D, :] = np.identity(D)
    final_y = 0
    subfunctions = [LRastrigin, LRastrigin, LWeierstrass, LWeierstrass, LGriewank, LGriewank,  LAckley, LAckley, LSphere, LSphere]
    bias = 120
    sigma = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    lambd = np.array([1, 1, 10, 10, 5/60, 5 / 60, 5 / 32, 5 / 32, 5 / 100, 5 / 100])
    #lambd = np.array([0.1 * 5 / 32, 5 / 32, 2 * 1, 1, 2 * 5 / 100, 5 / 100, 2 * 10, 10, 2 * 5 / 60, 5 / 60])
    lambd = np.repeat(lambd.reshape(10, 1), D, axis=1)
    O = O_ori[:, 0:D]
    bias = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    O[9, :] = 0
    weight = np.empty((p.shape[0], 10))
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        weight[:, i] = np.exp(-np.sum(np.square(p - oo), axis=1) / (2 * D * sigma[i] * sigma[i]))
    maxweight = np.amax(weight, axis=1)
    for i in range(p.shape[0]):
        for j in range(10):
            if weight[i, j] != maxweight[i]:
                weight[i, j] = weight[i, j] * (1 - np.power(maxweight[i], 10))
    sumweight = np.sum(weight, axis=1)
    weight = weight / np.repeat(sumweight.reshape(-1, 1), 10, axis=1)
    tmp_y = np.ones((1, D)) * 5
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        cur_subfunc = subfunctions[i]
        f = cur_subfunc(
            np.matmul((p - oo) / (np.repeat(lambd[i, :].reshape(1, D), p.shape[0], axis=0)), MD[i * D:(i + 1) * D, :]))
        fmax = cur_subfunc(np.matmul(tmp_y / lambd[i, :].reshape(1, D), MD[i * D:(i + 1) * D, :]))
        f1 = 2000 * f / fmax
        final_y = final_y + weight[:, i] * (f1 + bias[i])
    return final_y

def F16(p):  # [-5, 5] f_bias = 120
    problem_size, D = np.shape(p)
    if D <= 100:
        O_ori = np.loadtxt(
            fname="E:/support_data/data_hybrid_func1.txt")
    else:
        O_ori = -5 + 10 * np.random.random((10, D))
    if D == 10:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func1_M_D10.txt")
    elif D == 30:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func1_M_D30.txt")
    elif D == 50:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func1_M_D50.txt")
    else:
        c = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        # MD = rot_matrix(D, c)
        MD = np.empty((10 * D, D))
        for i in range(10):
            MD[i * D:(i + 1) * D, :] = rot_matrix(D, c[i])
    final_y = 0
    subfunctions = [LRastrigin, LRastrigin, LWeierstrass, LWeierstrass, LGriewank, LGriewank,  LAckley, LAckley, LSphere, LSphere]
    bias = 120
    sigma = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    lambd = np.array([1, 1, 10, 10, 5/60, 5 / 60, 5 / 32, 5 / 32, 5 / 100, 5 / 100])
    #lambd = np.array([0.1 * 5 / 32, 5 / 32, 2 * 1, 1, 2 * 5 / 100, 5 / 100, 2 * 10, 10, 2 * 5 / 60, 5 / 60])
    lambd = np.repeat(lambd.reshape(10, 1), D, axis=1)
    O = O_ori[:, 0:D]
    bias = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    O[9, :] = 0
    weight = np.empty((p.shape[0], 10))
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        weight[:, i] = np.exp(-np.sum(np.square(p - oo), axis=1) / (2 * D * sigma[i] * sigma[i]))
    maxweight = np.amax(weight, axis=1)
    for i in range(p.shape[0]):
        for j in range(10):
            if weight[i, j] != maxweight[i]:
                weight[i, j] = weight[i, j] * (1 - np.power(maxweight[i], 10))
    sumweight = np.sum(weight, axis=1)
    weight = weight / np.repeat(sumweight.reshape(-1, 1), 10, axis=1)
    tmp_y = np.ones((1, D)) * 5
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        cur_subfunc = subfunctions[i]
        f = cur_subfunc(
            np.matmul((p - oo) / (np.repeat(lambd[i, :].reshape(1, D), p.shape[0], axis=0)), MD[i * D:(i + 1) * D, :]))
        fmax = cur_subfunc(np.matmul(tmp_y / lambd[i, :].reshape(1, D), MD[i * D:(i + 1) * D, :]))
        f1 = 2000 * f / fmax
        final_y = final_y + weight[:, i] * (f1 + bias[i])
    return final_y

def F18(p):  # [-5, 5] f_bias = 10 cec2005 F18
    problem_size, D = np.shape(p)
    if D <= 100:
        O_ori = np.loadtxt(
            fname="E:/support_data/data_hybrid_func2.txt")
    else:
        O_ori = -5 + 10 * np.random.random((10, D))
    if D == 10:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D10.txt")
    elif D == 30:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D30.txt")
    elif D == 50:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D50.txt")
    else:
        c = [2, 3, 2, 3, 2, 3, 20, 30, 200, 300]
        # MD = rot_matrix(D, c)
        MD = np.empty((10 * D, D))
        for i in range(10):
            MD[i * D:(i + 1) * D, :] = rot_matrix(D, c[i])
    final_y = 0
    subfunctions = [LAckley, LAckley, LRastrigin, LRastrigin, LSphere, LSphere, LWeierstrass, LWeierstrass, LGriewank, LGriewank]
    bias = 10
    sigma = np.array([1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2])
    lambd = np.array([2*5/32, 5/32, 2*1, 1, 2*5/100, 5 / 100, 2 * 10, 10, 2*5 / 60, 5 / 60])
    #lambd = np.array([0.1 * 5 / 32, 5 / 32, 2 * 1, 1, 2 * 5 / 100, 5 / 100, 2 * 10, 10, 2 * 5 / 60, 5 / 60])
    lambd = np.repeat(lambd.reshape(10, 1), D, axis=1)
    O = O_ori[:, 0:D]
    bias = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    O[9, :] = 0
    weight = np.empty((p.shape[0], 10))
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        weight[:, i] = np.exp(-np.sum(np.square(p - oo), axis=1) / (2 * D * sigma[i] * sigma[i]))
    maxweight = np.amax(weight, axis=1)
    for i in range(p.shape[0]):
        for j in range(10):
            if weight[i, j] != maxweight[i]:
                weight[i, j] = weight[i, j] * (1 - np.power(maxweight[i], 10))
    sumweight = np.sum(weight, axis=1)
    weight = weight / np.repeat(sumweight.reshape(-1, 1), 10, axis=1)
    tmp_y = np.ones((1, D)) * 5
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        cur_subfunc = subfunctions[i]
        f = cur_subfunc(
            np.matmul((p - oo) / (np.repeat(lambd[i, :].reshape(1, D), p.shape[0], axis=0)), MD[i * D:(i + 1) * D, :]))
        fmax = cur_subfunc(np.matmul(tmp_y / lambd[i, :].reshape(1, D), MD[i * D:(i + 1) * D, :]))
        f1 = 2000 * f / fmax
        final_y = final_y + weight[:, i] * (f1 + bias[i])
    return final_y

def F19(p):  # [-5, 5] f_bias = 10
    problem_size, D = np.shape(p)
    if D <= 100:
        O_ori = np.loadtxt(
            fname="E:/support_data/data_hybrid_func2.txt")
    else:
        O_ori = -5 + 10 * np.random.random((10, D))
    if D == 10:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D10.txt")
    elif D == 30:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D30.txt")
    elif D == 50:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D50.txt")
    else:
        c = [2, 3, 2, 3, 2, 3, 20, 30, 200, 300]
        # MD = rot_matrix(D, c)
        MD = np.empty((10 * D, D))
        for i in range(10):
            MD[i * D:(i + 1) * D, :] = rot_matrix(D, c[i])
    final_y = 0
    subfunctions = [LAckley, LAckley, LRastrigin, LRastrigin, LSphere, LSphere, LWeierstrass, LWeierstrass, LGriewank,
                    LGriewank]
    bias = 10
    sigma = np.array([0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2])
    lambd = np.array([0.1 * 5 / 32, 5 / 32, 2 * 1, 1, 2 * 5 / 100, 5 / 100, 2 * 10, 10, 2 * 5 / 60, 5 / 60])
    lambd = np.repeat(lambd.reshape(10, 1), D, axis=1)
    O = O_ori[:, 0:D]
    bias = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    O[9, :] = 0
    weight = np.empty((p.shape[0], 10))
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        weight[:, i] = np.exp(-np.sum(np.square(p - oo), axis=1) / (2 * D * sigma[i] * sigma[i]))
    maxweight = np.amax(weight, axis=1)
    for i in range(p.shape[0]):
        for j in range(10):
            if weight[i, j] != maxweight[i]:
                weight[i, j] = weight[i, j] * (1 - np.power(maxweight[i], 10))
    sumweight = np.sum(weight, axis=1)
    weight = weight / np.repeat(sumweight.reshape(-1, 1), 10, axis=1)

    tmp_y = np.ones((1, D)) * 5
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        cur_subfunc = subfunctions[i]
        f = cur_subfunc(
            np.matmul((p - oo) / (np.repeat(lambd[i, :].reshape(1, D), p.shape[0], axis=0)), MD[i * D:(i + 1) * D, :]))
        fmax = cur_subfunc(np.matmul(tmp_y / lambd[i, :].reshape(1, D), MD[i * D:(i + 1) * D, :]))
        f1 = 2000 * f / fmax
        final_y = final_y + weight[:, i] * (f1 + bias[i])
    return final_y

def F20(p):  # [-5, 5] f_bias = 10 cec05F20
    problem_size, D = np.shape(p)
    if D <= 100:
        O_ori = np.loadtxt(
            fname="E:/support_data/data_hybrid_func2.txt")
    else:
        O_ori = -5 + 10 * np.random.random((10, D))
    if D == 10:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D10.txt")
    elif D == 30:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D30.txt")
    elif D == 50:
        MD = np.loadtxt(
            fname="E:/support_data/hybrid_func2_M_D50.txt")
    else:
        c = [2, 3, 2, 3, 2, 3, 20, 30, 200, 300]
        # MD = rot_matrix(D, c)
        MD = np.empty((10 * D, D))
        for i in range(10):
            MD[i * D:(i + 1) * D, :] = rot_matrix(D, c[i])
    final_y = 0
    subfunctions = [LAckley, LAckley, LRastrigin, LRastrigin, LSphere, LSphere, LWeierstrass, LWeierstrass, LGriewank,
                    LGriewank]
    bias = 10
    sigma = np.array([0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2])
    lambd = np.array([0.1 * 5 / 32, 5 / 32, 2 * 1, 1, 2 * 5 / 100, 5 / 100, 2 * 10, 10, 2 * 5 / 60, 5 / 60])
    lambd = np.repeat(lambd.reshape(10, 1), D, axis=1)
    O = O_ori[:, 0:D]
    O[:1, 1::2] = 5
    bias = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    O[9, :] = 0
    weight = np.empty((p.shape[0], 10))
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        weight[:, i] = np.exp(-np.sum(np.square(p - oo), axis=1) / (2 * D * sigma[i] * sigma[i]))
    maxweight = np.amax(weight, axis=1)
    for i in range(p.shape[0]):
        for j in range(10):
            if weight[i, j] != maxweight[i]:
                weight[i, j] = weight[i, j] * (1 - np.power(maxweight[i], 10))
    sumweight = np.sum(weight, axis=1)
    weight = weight / np.repeat(sumweight.reshape(-1, 1), 10, axis=1)
    tmp_y = np.ones((1, D)) * 5
    for i in range(10):
        oo = np.repeat(O[i, :].reshape(1, D), p.shape[0], axis=0)
        cur_subfunc = subfunctions[i]
        f = cur_subfunc(
            np.matmul((p - oo) / (np.repeat(lambd[i, :].reshape(1, D), p.shape[0], axis=0)), MD[i * D:(i + 1) * D, :]))
        fmax = cur_subfunc(np.matmul(tmp_y / lambd[i, :].reshape(1, D), MD[i * D:(i + 1) * D, :]))
        f1 = 2000 * f / fmax
        final_y = final_y + weight[:, i] * (f1 + bias[i])
    return final_y




















