
import numpy as np

def reward_function(ghf, hf, candidate_fit, NFEs, Arm):
    # Update the low-level arm reward
    N = len(ghf)
    ghf = np.reshape(ghf, (1, N))
    ghf_sum = np.concatenate((ghf, candidate_fit.reshape(1, -1)), axis=1)
    index = np.argsort(ghf_sum)
    in_ = (np.where(index == N)[1][0]) + 1
    reward = -1 / N * in_ + (N + 1) / N

    # if candidate_fit == np.min(hf):
    #     print(f"Current optimal obtained by {Arm} arm is: {candidate_fit} NFE={NFEs}")

    return reward
