import numpy as np
import time
from pyDOE import *  # 含有拉丁超立方采样方法
import matplotlib.pyplot as plt
from GAN_model3 import GAN
from De_operator_2 import DE_2 as DE
from numpy import sum, dot, cos, sqrt, e, pi, exp
#from De_operator_2 import full_cross_mutation as fc
from numpy.random import normal
from numpy import linalg as LA
import math
from numpy import linalg
from my_rbf import RBF
from De_operator_2 import full_cross_mutation
# import test_Function
# 测试函数定义

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

def Ellipsoid(p):  # 5.12 f123 D203050
    d = np.arange(0, D) + 1
    y = np.square(p)

    return np.dot(y, d).transpose()


def Rosenbrock(p):  # 2.048 f456
    y = np.sum(100 * np.square(p[:, 1:] - np.square(p[:, :-1])) + np.square(p[:, :-1] - 1), axis=1)
    return y


def Ackley(p):  # 32.768 f789  f12
    tmp1 = np.sqrt(np.sum(np.square(p), axis=1) / D)
    tmp2 = np.sum(np.cos(2 * np.pi * p), axis=1)
    y = -20 * np.exp(-0.2 * tmp1) - np.exp(tmp2 / D) + 20 + np.exp(1)
    return y


def Griewank(p):  # 600 f101112
    tmp1 = np.sum(np.square(p), axis=1) / 4000
    tmp2 = np.cos(p / np.sqrt(np.arange(0, D) + 1))
    y = 1 + tmp1 - np.prod(tmp2, axis=1)
    return y

def LAckley(p):  # 32.768 f12
    tmp1 = np.sqrt(np.sum(np.square(p), axis=1) / D)
    tmp2 = np.sum(np.cos(2 * np.pi * p), axis=1)
    y = -20 * np.exp(-0.2 * tmp1) - np.exp(tmp2 / D) + 20 + np.exp(1)
    return y


def LRastrigin(p):  # 5 f34
    y = np.sum((np.square(p) - 10 * np.cos(2 * np.pi * p) + 10), axis=1)
    return y


def LSphere(p):  # f56
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
    tmp1 = np.sum(np.square(p), axis=1) / 4000
    tmp2 = np.cos(p / np.sqrt(np.arange(0, D) + 1))
    y = 1 + tmp1 - np.prod(tmp2, axis=1)
    return y

def ShiftedSphereFunction(p): #CEC2005F1 bias = -450
    O_or = O_ori[0:D]
    O = O_or.reshape(1, D)
    z = p - O
    y = np.sum(np.square(z), axis=1) - 450
    return y

def schwefel_102(p): #CEC2005F2 bias= -450
    O_or = O_ori[0:D]
    O = O_or.reshape(1, D)
    z = p - O
    y = 0
    for i in range(D):
        y += (np.sum(z[:, 0:i], axis=1)) ** 2
    return y - 450

def ShiftedRotatedRastrigin(p):  # 5 f13 D30
    O_or = O_ori[0:D]
    O = O_or.reshape(1, D)
    z = np.dot((p - O), MD30)
    y = np.sum((np.square(z) - 10 * np.cos(2 * np.pi * p) + 10), axis=1) - 330
    return y

def rastrigin_func(p):#cecF9 bias -330
    O_or = O_ori[0:D]
    O = O_or.reshape(1, D)
    z = p - O
    f = np.sum(z ** 2 - 10 * cos(2 * pi * z) + 10, axis=1)
    #f = sum(x. ^ 2 - 10. * cos(2. * pi. * x) + 10, 2);
    return f - 330

def schwefel_102_noise_func(p): #CEC2005F4
    O_or = O_ori[0:D]
    O = O_or.reshape(1, D)
    z = p - O
    result = 0
    for i in range(D):
        result += (np.sum(z[:, 0:i], axis=1)) ** 2
    #f=f.*(1+0.4.*abs(normrnd(0,1,ps,1)));
    result*(1 + 0.4 * abs(normal(0, 1))) - 450
    return result

def RHCF(p):  # [-5, 5] f_bias = 10
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

def RHCF1(p):  # [-5, 5] f_bias = 120 cec F16
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



# 参数设置
functions1 = [Ellipsoid, Rosenbrock, Ackley, Griewank, ShiftedRotatedRastrigin, RHCF, schwefel_102_noise_func,
              RHCF1]
np.set_printoptions(threshold=np.inf)  # 设置显示精度，threshold=np.inf为完全显示
Dim = [30, 50, 100, 200]  # test dimensionality
# max_gen = 250  # the maxinal evolution generation
max_FEs = 1000  # the maximal FEs
max_times = 20  # the total independent run times
max_FEs1 = 800
max_FEs2 = max_FEs-max_FEs1
NL = 4
CR = 0.8
F = 0.5

if __name__ == "__main__":
    start_time = time.time()
    for i in range(0, 4):
        D = Dim[i]
        print(D)
        if D >= 50:
            NP = 200  # the population size 可调整
        elif D < 50:
            NP = 100

        for fi in range(7, 8):
            O_ori = None
            MD = np.empty((NP * 10, D))
            c = None
            if fi == 0:
                bound = 5.12  # the boundary of the test problem
            elif fi == 1:
                bound = 2.048
            elif fi == 2:
                bound = 32.768
            elif fi == 3:
                bound = 600
            elif fi == 4 or fi == 5:
                 bound = 5
            elif fi == 6:
                bound = 100
            elif fi == 7:
                bound = 5
            if fi == 4:
                if D <= 100:
                    O_ori = np.loadtxt(
                        fname="E:/Expensive optimization/Code/18/CA-LLSO_Code-main (1)/CA-LLSO_Code-main/ReadDataFiles/rastrigin_func_data.txt")
                else:
                    O_ori = -5 + 10 * np.random.random((1, D))
                if D == 10:
                    MD30 = np.loadtxt(
                        fname="E:/Expensive optimization/Code/18/CA-LLSO_Code-main (1)/CA-LLSO_Code-main/ReadDataFiles/rastrigin_M_D10.txt")
                elif D == 30:
                    MD30 = np.loadtxt(
                        fname="E:/Expensive optimization/Code/18/CA-LLSO_Code-main (1)/CA-LLSO_Code-main/ReadDataFiles/rastrigin_M_D30.txt")
                elif D == 50:
                    MD30 = np.loadtxt(
                        fname="E:/Expensive optimization/Code/18/CA-LLSO_Code-main (1)/CA-LLSO_Code-main/ReadDataFiles/rastrigin_M_D50.txt")
                else:
                    c = 2
                    MD30 = rot_matrix(D, c)
            if fi == 5:
                if D <= 100:
                    O_ori = np.loadtxt(
                        fname="E:/Expensive optimization/Code/18/CA-LLSO_Code-main (1)/CA-LLSO_Code-main/ReadDataFiles/hybrid_func2_data.txt")
                else:
                    O_ori = -5 + 10 * np.random.random((10, D))
                if D == 10:
                    MD = np.loadtxt(
                        fname="E:/Expensive optimization/Code/18/CA-LLSO_Code-main (1)/CA-LLSO_Code-main/ReadDataFiles/hybrid_func2_M_D10.txt")
                elif D == 30:
                    MD = np.loadtxt(
                        fname="E:/Expensive optimization/Code/18/CA-LLSO_Code-main (1)/CA-LLSO_Code-main/ReadDataFiles/hybrid_func2_M_D30.txt")
                elif D == 50:
                    MD = np.loadtxt(
                        fname="E:/Expensive optimization/Code/18/CA-LLSO_Code-main (1)/CA-LLSO_Code-main/ReadDataFiles/hybrid_func2_M_D50.txt")
                else:
                    c = [2, 3, 2, 3, 2, 3, 20, 30, 200, 300]
                    # MD = rot_matrix(D, c)
                    MD = np.empty((10 * D, D))
                    for i in range(10):
                        MD[i * D:(i + 1) * D, :] = rot_matrix(D, c[i])
            if fi == 6:
                if D <= 100:
                    O_ori = np.loadtxt(
                        fname="E:/support_data/data_schwefel_102.txt")
                else:
                    O_ori = -100 + 200 * np.random.random((1, D))

            if fi == 7:
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
            best_ever = np.zeros((max_times, max_FEs))
            best_ever_time = np.zeros((max_times, 1))
            average_value = np.zeros((max_times, max_FEs))
            d_train_loss_data_save = []
            g_train_loss_data_save = []
            save_CE = np.zeros((max_times, max_FEs))
            for times in range(max_FEs1):
                pop_x = None
                pop_v = np.zeros((NP, D))
                pop_y = None
                db_x = None
                db_v = np.zeros((NP, D))
                db_y = None
                rank = np.zeros((NP, 1))  # NP个一维数据
                ranked_x = np.zeros((NP, D))
                ranked_x_1 = np.zeros((int(NP / NL), D))
                ranked_v = np.zeros((NP, D))
                ranked_y = np.zeros((NP,))
                CE = np.zeros(max_FEs,)

                selected_function = functions1[fi]
                db_x = lhs(D, samples=NP, criterion='center') * bound * 2 - bound
                db_y = selected_function(db_x)
                print(np.min(db_y))
                database = np.concatenate((db_x, db_y.reshape(db_y.shape[0], 1)), axis=1)
                FEs = np.shape(db_y)[0]
                for ii in range(FEs):
                    y = database[0:ii+1, -1]
                    CE[ii] = np.amin(y)
                best_ever[times, 0:FEs] = np.amin(db_y)
                # net = GAN(D, 32, 0.0001, 200, D)
                net = GAN(D, 32, 0.0001, 100, D)  # self.gp.d, self.batch_size, self.lr, self.epoch, self.n_noise)
                while FEs <= max_FEs1:
                    ORIGIN = database
                    ORIGIN = ORIGIN[ORIGIN[:, -1].argsort()]  # 完成好的排序
                    pop_x = ORIGIN[0:NP, 0:D]  # 提取完成好排列的x(从小到大)
                    pop_y = ORIGIN[0:NP, -1]  # 提取完成好排列的y(从小到大)
                    Y_min = np.amin(pop_y)
                    P = np.zeros((int(NP / NL), D))
                    ranked_x = pop_x
                    ranked_y = pop_y
                    ranked_x_1[0:int(NP / NL), :] = pop_x[0:int(NP / NL), :]  # regarded as real data_x
                    rank[0:int(NP / NL), ] = 1  # regarded as real data
                    # min-max normalization
                    input_dec = (ranked_x - np.tile(-bound, (np.shape(ranked_x)[0], 1))) / \
                                np.tile(bound - (-bound), (np.shape(ranked_x)[0], 1))
                    samples_pool_1 = (ranked_x_1 - np.tile(-bound, (np.shape(ranked_x_1)[0], 1))) / \
                                     np.tile(bound - (-bound), (np.shape(ranked_x_1)[0], 1))
                    samples_pool = ranked_x_1 / np.tile(bound, (np.shape(ranked_x_1)[0], 1))
                    d_train_loss_data, g_train_loss_data = net.train(input_dec, rank, samples_pool)
                    d_train_loss_data_T = d_train_loss_data.transpose()
                    g_train_loss_data_T = g_train_loss_data.transpose()
                    d_train_loss_data_save = np.append(d_train_loss_data_save, d_train_loss_data_T, axis=0)
                    g_train_loss_data_save = np.append(g_train_loss_data_save, g_train_loss_data_T, axis=0)
                    offspring_x = np.zeros((NP, D))
                    # offspring_y = np.zeros((NP, D))
                    # offspring_rank = np.zeros((NP,))
                    # offspring produce
                    # off = net.generate(ref_dec / np.tile(pro.upper, (np.shape(ref_dec)[0], 1)), self.gp.n) * \
                    # np.tile(pro.upper, (self.gp.n, 1))
                    offspring_x = net.generate(samples_pool, np.shape(samples_pool)[0]) * \
                                  np.tile(bound, (np.shape(samples_pool)[0], 1))
                    print("Start GANs")
                    offspring_x_1 = net.generate(samples_pool_1, np.shape(samples_pool_1)[0]) * \
                                    np.tile(bound - (-bound), (np.shape(samples_pool_1)[0], 1))
                    for j in range(int(NP / NL)):
                        for i in range(D):
                            if offspring_x[j, i] < -bound:
                                offspring_x[j, i] = -bound
                            if offspring_x[j, i] > bound:
                                offspring_x[j, i] = bound
                    # offspring_y = selected_function(offspring_x)

                    offspring_y = selected_function(offspring_x)
                    FEs = FEs + np.shape(offspring_y)[0]
                    Last_FEs = FEs - np.shape(offspring_y)[0]
                    if FEs > max_FEs:
                        break
                    tempt = np.concatenate((offspring_x, offspring_y.reshape(offspring_y.shape[0], 1)), axis=1)
                    # tempt_1 = np.concatenate((offspring_x, offspring_y_1.reshape(offspring_y_1.shape[0], 1)), axis=1)
                    database = np.append(database, tempt, axis=0)
                    for ii in range(Last_FEs, FEs):
                        y = database[0:ii+1, -1]
                        CE[ii] = np.amin(y)
                    # database_1 = np.append(database_1, tempt_1, axis=0)
                    y = database[:, -1]
                    # y_1 = database_1[:, -1]
                    current_Y_min = np.amin(y)
                    print(current_Y_min)
                    best_ever[times, FEs - 25:] = current_Y_min
                    best_ever_time[times, :] = best_ever[times, -1]
                    database_sort = database[database[:, -1].argsort()]
                    # if np.random.uniform() <= 0.2:
                    #     select_P = database_sort[0:9, 0:D]
                    #     candidate_offspring, FEs = fc(select_P, selected_function, FEs)
                    #     candidate_F = selected_function(candidate_offspring)
                    #     print(candidate_F)
                    #     tempt_2 = np.concatenate((U, U_F.reshape(U_F.shape[0], 1)), axis=1)
                    if current_Y_min >= Y_min:
                        print("Start DE/best/1")
                        P = database_sort[0:int(NP / NL), 0:D]
                        U = DE(P, F, CR, bound, -bound)
                        U_F = selected_function(U)
                        current_UF_min = np.amin(U_F)
                        print(current_UF_min)
                        FEs = FEs + np.shape(U_F)[0]
                        Last_FEs = FEs - np.shape(U_F)[0]
                        if FEs > max_FEs:
                            break
                        tempt_2 = np.concatenate((U, U_F.reshape(U_F.shape[0], 1)), axis=1)
                        database = np.append(database, tempt_2, axis=0)
                        current_Y_min = np.amin(database[:, -1])
                        print(current_Y_min)
                        for ii in range(Last_FEs, FEs):
                            y = database[0:ii + 1, -1]
                            CE[ii] = np.amin(y)
                        best_ever[times, FEs - 25:] = current_Y_min
                        best_ever_time[times, :] = best_ever[times, -1]
                while FEs <= max_FEs:

                    database_sort = database[database[:, -1].argsort()]
                    x = database_sort[0:int(NP / NL), 0:D]
                    y = database_sort[0:int(NP / NL), -1]
                    rbf = RBF(np.shape(x)[1], 20, np.shape(x)[1])
                    rbf.train(x, y)



                print(D, fi, times, best_ever[times, -1])
                save_CE[times, :] = CE
                # print(best_ever_1)
                # print("Dim:", D, "times:", times, "Function:", fi, "best_value:", best_ever)
            #best_ever_time = best_ever.min(1).transpose()
            average_value = np.mean(save_CE, axis=0)  # draw Convergence curve
            best_average = np.mean(best_ever_time, axis=0)
            best_std = np.std(best_ever_time, axis=0)
            a = d_train_loss_data_save
            d_train_dim = len(d_train_loss_data_save) // max_times
            # d_train_dim = np.shape(d_train_loss_data_save)[1]
            g_train_dim = len(d_train_loss_data_save) // max_times
            d_train_loss_data_save_ave = np.mean(d_train_loss_data_save.reshape(max_times, d_train_dim), axis=0)
            g_train_loss_data_save_ave = np.mean(g_train_loss_data_save.reshape(max_times, g_train_dim), axis=0)
            print("Dim = ", D, "Function =", fi, "average =", best_average, "std =", best_std)
            #best_save = np.concatenate(best_average, best_std, axis=1)
            best_save = [best_average, best_std]
            d_train_t = d_train_loss_data_save_ave.reshape(-1, 1)
            g_train_t = g_train_loss_data_save_ave.reshape(-1, 1)
            train_loss = np.concatenate((d_train_t, g_train_t), axis=1)
            np.savetxt(
               'Gans_average_value_f%d_D%d_eval%d_ite%d.txt' % (fi + 1, D, FEs, max_times),
               best_save)
            print('The function = %s in dim = %d mean = %d, std = %d' %(functions1[fi], D, best_average[-1], best_std[-1]))
            np.savetxt(
                'Gans_convergence_curve_f%d_D%d_eval%d_ite%d.txt' % (fi + 1, D, FEs, max_times),
                average_value)
            np.savetxt(
                'Gans_d_data_loss_f%d_D%d_eval%d_ite%d.txt' % (fi + 1, D, FEs, max_times),
                 train_loss)
    end_time = time.time()
    print('time cost: ', end_time - start_time, 's')
