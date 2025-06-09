import numpy as np

def srgtsPRSCreateGramianMatrix(x, NbVariables, PRSDegree, PRSRemovedIdx=None):
    NbPoints = len(x)

    # create X
    X = np.ones((NbPoints, 1))

    if PRSDegree > 0:
        X = np.hstack((X, x))
        X1 = x.copy()
        nc1 = NbVariables
        n_loc = np.arange(1, NbVariables + 1)
        n_loc1 = 1
        for i in range(2, PRSDegree + 1):
            nr, nc = X1.shape
            X2 = np.zeros((nr, 0))
            ctr = 0
            for k in range(NbVariables):
                l_ctr = 0
                for j in range(n_loc[k] - 1, nc):
                    X2 = np.hstack((X2, (x[:, k] * X1[:, j]).reshape(-1, 1)))
                    ctr += 1
                    l_ctr += 1
                n_loc1 += l_ctr
            nc1 = nc
            X = np.hstack((X, X2))
            X1 = X2
            n_loc = n_loc1

    if PRSRemovedIdx is not None:
        X = np.delete(X, PRSRemovedIdx, axis=1)

    return X
