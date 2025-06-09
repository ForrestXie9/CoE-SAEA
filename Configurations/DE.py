import numpy as np
from Configurations.srgtsPRSEvaluate import srgtsPRSEvaluate

class Pop:
    def __init__(self, X, func, flag):
        self.X = X
        self.func = func
        self.ObjV = None
        self.flag = flag
    def __add__(self, other):
        self.X = np.vstack([self.X, other.X])
        self.ObjV = np.hstack([self.ObjV, other.ObjV])
        return self

    def cal_fitness(self):  #

        if self.flag == 2:
            self.ObjV =  self.func(self.X)
        else:
            self.ObjV  = srgtsPRSEvaluate(self.X, self.func)
            self.ObjV =  self.ObjV

class DE(object):
    def __init__(self, func, max_iter, dim, lb, ub, flag, initX=None):

        self.max_iter = max_iter
        self.initX = initX
        self.flag = flag
        if (self.initX is None):
            self.popsize = 100
        else:
            self.popsize = self.initX.shape[0]

        self.F = 0.5
        self.CR = 0.9

        self.func = func
        self.dim = dim
        self.xmin = np.array(lb)
        self.xmax = np.array(ub)

        self.xbest = None
        self.ybest = None
        self.pop = None

    def initPop(self):
        X = np.zeros((self.popsize, self.dim))
        area = self.xmax - self.xmin
        X = area * np.random.rand(self.popsize, self.dim) + self.xmin


        self.pop = Pop(X, self.func, self.flag)
        self.pop.cal_fitness()

    def mutation(self):
        muX = np.empty((self.popsize, self.dim))
        b = np.argmin(self.pop.ObjV)
        for i in range(self.popsize):  # DE/rand/1
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i  or r2 == r1:
                r1 = np.random.randint(0, self.popsize - 1)
                r2 = np.random.randint(0, self.popsize - 1)

            mutation = self.pop.X[b] + self.F * (self.pop.X[r1] - self.pop.X[r2])

            for j in range(self.dim):
                if self.xmin[j] <= mutation[j] <= self.xmax[j]:
                    muX[i, j] = mutation[j]
                else:
                    rand_value = self.xmin[j] + np.random.random() * (self.xmax[j] - self.xmin[j])
                    muX[i, j] = rand_value
        return muX

    def crossover(self, muX):
        crossX = np.empty((self.popsize, self.dim))
        for i in range(self.popsize):
            rj = np.random.randint(0, self.dim - 1)
            for j in range(self.dim):
                rf = np.random.random()
                if rf <= self.CR or rj == j:
                    crossX[i, j] = muX[i, j]
                else:
                    crossX[i, j] = self.pop.X[i, j]

        return crossX

    def selection(self, crossPop):
        for i in range(self.popsize):
            if crossPop.ObjV[i] < self.pop.ObjV[i]:
                self.pop.X[i] = crossPop.X[i]
                self.pop.ObjV[i] = crossPop.ObjV[i]
                # self.pop.CV[i] = crossPop.CV[i]

    def update_best(self):
        rank = np.argsort(self.pop.ObjV)
        self.xbest = self.pop.X[rank[0]]
        self.ybest = self.pop.ObjV[rank[0]]

    def run(self):
        if self.initX is None:
            self.initPop()
        else:
            self.pop = Pop(self.initX, self.func, self.flag)
            self.pop.cal_fitness()
        self.update_best()

        for i in range(self.max_iter):
            muX = self.mutation()
            crX = self.crossover(muX)

            epop = Pop(crX, self.func, self.flag)
            epop.cal_fitness()

            self.selection(epop)
            self.update_best()

        return self.xbest

