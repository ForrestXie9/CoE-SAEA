import numpy as np
from smt.surrogate_models import KRG
from Configurations.Reward_design import reward_function

def GP_lcb_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs, num_arm, paras):

    try:
        sm = KRG(poly='constant', corr='squar_exp', print_global=False, n_start = 1)
        sm.set_training_values(ghx, ghf)
        sm.train()
        w = 2
        fitnessModel = sm.predict_values(offspring)
        v = np.sqrt(sm.predict_variances(offspring))
        predict_values = fitnessModel - w * v

        sidx = np.argmin(predict_values)  # Get the best point indexs
        candidate_position = offspring[sidx, :]

        ih = np.where((hx == candidate_position).all(axis=1))[0]

        if len(ih) == 0:
            candidate_fit = FUN(candidate_position)  # Evaluation
            NFEs += 1

            hx = np.vstack((hx, candidate_position))
            hf = np.append(hf, candidate_fit)

            # Update CE for plotting
            CE[NFEs - 1, :] = [NFEs - 1, candidate_fit]
            gfs[NFEs-1] = np.min(CE[0:NFEs, 1])

            # Update the low level arm reward
            Arm = 'GP_lcb '
            reward = reward_function(ghf, hf, candidate_fit, NFEs, Arm)
            if candidate_fit == np.min(hf):
                print(f"Current optimal obtained by {Arm} arm is: {candidate_fit} NFE={NFEs}")

        else:
            candidate_fit = []
            candidate_position = []
            reward = 0  # current database has not updated
    except:
        candidate_fit = []
        candidate_position = []
        reward = 0
    return hx, hf, reward, NFEs, CE, gfs, candidate_position, candidate_fit