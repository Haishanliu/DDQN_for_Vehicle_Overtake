
# def get_reward(info):
#     if info['crashed']:
#         reward =
import numpy as np



def check_direction(obs):
    penalty = 0
    if obs[0, 3] <= 0:  # to avoid inverse direction
        penalty = -50
    return penalty


def check_idle(obs):
    penalty = 0
    if np.all(obs[1:,:]==0):
        penalty = -10
    return penalty







