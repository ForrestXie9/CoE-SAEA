import numpy as np

from Configurations.Reward_design import reward_function
from Configurations.srgtsPRSEvaluate import srgtsPRSEvaluate
from Configurations.srgtsPRSFit import srgtsPRSFit
from Configurations.srgtsPRSSetOptions import srgtsPRSSetOptions

def PRS_pre_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs, num_arm, paras):

    srgtOPT = srgtsPRSSetOptions(ghx, ghf)
    # Fit the polynomial response surface
    srgtSRGT = srgtsPRSFit(srgtOPT)

    fitnessModel = srgtsPRSEvaluate(offspring, srgtSRGT)

    sidx = np.argmin(fitnessModel)  # Get the best point indexs
    candidate_position = offspring[sidx, :]

    ih = np.where((hx == candidate_position).all(axis=1))[0]

    if len(ih) == 0:
        candidate_fit = FUN(candidate_position)  # Evaluation
        NFEs += 1

        # Save candidate into dataset, and sort dataset
        hx = np.vstack((hx, candidate_position))
        hf = np.append(hf, candidate_fit)

        # Update CE for plotting
        CE[NFEs-1, :] = [NFEs-1, candidate_fit]
        gfs[NFEs-1] = np.min(CE[0:NFEs, 1])

        # Update the low level arm reward
        Arm = 'PRS_prescreening '
        reward = reward_function(ghf, hf, candidate_fit, NFEs, Arm)
        if candidate_fit == np.min(hf):
            print(f"Current optimal obtained by {Arm} arm is: {candidate_fit} NFE={NFEs}")
    else:
        candidate_fit = []
        candidate_position = []
        reward = 0  # current database has not updated

    return hx, hf, reward, NFEs, CE, gfs, candidate_position, candidate_fit
