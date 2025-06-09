import numpy as np
from Configurations.RBFN import RBFN
from Configurations.Reward_design import reward_function

from Configurations.DE import DE

def RBF_ls_arm(ghx, ghf, hx, hf, FUN, NFEs, CE, gfs, LB, UB, num_arm, paras):

    Dim = hx.shape[1]


    sm = RBFN(input_shape=ghx.shape[1], hidden_shape=int(ghx.shape[0]), kernel='gaussian')
    sm.fit(ghx, np.transpose(ghf))

    flag = 2

    lu = np.array([np.min(ghx, axis=0), np.max(ghx, axis=0)])
    LB = lu[0, :]
    UB = lu[1, :]
    ga = DE(max_iter=Dim+10, func=sm.predict, dim=Dim, lb=LB, ub=UB, flag=flag, initX=None)
    candidate_position = ga.run()

    ih = np.where(np.all(hx == candidate_position, axis=1))[0]

    if len(ih) == 0:
        candidate_fit = FUN(candidate_position)
        NFEs += 1

        # Update hx and hf
        hx = np.vstack([hx, candidate_position])
        hf = np.append(hf, candidate_fit)

        # Update CE for plotting
        CE[NFEs - 1, :] = [NFEs - 1, candidate_fit]
        gfs[NFEs-1] = np.min(CE[0:NFEs, 1])

        # Update the low-level arm reward
        Arm = 'RBF_local_search '
        reward = reward_function(ghf, hf, candidate_fit, NFEs, Arm)
        if candidate_fit == np.min(hf):
            print(f"Current optimal obtained by {Arm} arm is: {candidate_fit} NFE={NFEs}")
    else:
        candidate_fit = []
        candidate_position = []
        reward = 0 # Database has not been updated

    return hx, hf, reward, NFEs, CE, gfs, candidate_position, candidate_fit
