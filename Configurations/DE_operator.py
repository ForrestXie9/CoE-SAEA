import numpy as np


def DEoperator(P, NP, Dim, hisx, F, CR, UB, LB):
    U = np.zeros_like(P)
    for i in range(NP):
        # Mutation
        k0 = np.random.randint(1, NP + 1)
        while k0 == i:
            k0 = np.random.randint(1, NP + 1)
        P1 = P[k0 - 1]

        k1 = np.random.randint(1, NP + 1)
        while k1 == i or k1 == k0:
            k1 = np.random.randint(1, NP + 1)
        P2 = P[k1 - 1]

        Xbest = hisx[0]

        V = Xbest + F * (P1 - P2)
        # Bound
        for j in range(Dim):
            if V[j] > UB[i, j] or V[j] < LB[i, j]:
                V[j] = LB[i, j] + np.random.rand() * (UB[i, j] - LB[i, j])

        # Crossover
        jrand = np.random.randint(1, Dim + 1)
        for j in range(Dim):
            k3 = np.random.rand()
            if k3 <= CR or j + 1 == jrand:
                U[i, j] = V[j]
            else:
                U[i, j] = P[i, j]

    return U
