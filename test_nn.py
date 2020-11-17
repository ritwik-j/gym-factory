import sys
import gym
import gym_factory
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
# np.set_printoptions(threshold=sys.maxsize)

N_DISCRETE_ACTIONS = 27
MAX_MEM = 50  # 5000

N_COLOURS = 7
SpaceBetweenSkittles = 2
PROB = 0.5

BELT_SPEED = 5
WS_LENGTH = 30 *BELT_SPEED
GEN_LENGTH = 30 *BELT_SPEED
BELT_WIDTH = (((N_COLOURS*2)-1)*BELT_SPEED)+1
BELT_LENGTH = WS_LENGTH + GEN_LENGTH
HEIGHT = 6 *BELT_SPEED
SkittleTypes = BELT_WIDTH/(BELT_SPEED*SpaceBetweenSkittles)
WORLD_ARRAY_SIZE = BELT_LENGTH*BELT_WIDTH*HEIGHT

EPISODES = 20

PATH = Path("""/Users/ritwik/Desktop/gym-factory/MSELoss/""")

def main():

    input1 = T.rand(1, 1, 301, 66, 31)
    print("input1: ", input1.shape)
    
    conv1 = nn.Conv3d(1, 1, (272,47,16), stride = 1)
    
    output1 = conv1(input1)
    print("output1: ", output1.shape)

    conv2 = nn.Conv3d(1,1,(11,6,7),stride=1)

    output2 = conv2(output1)
    print("output2: ", output2.shape)

    output2 = output2.reshape(-1, 10, 300)

    # output2 = T.rand(32, 10, 300)

    print("output2 reshape: ", output2.shape)

    fc1 = nn.Linear(300, 2048) 

    output3 = fc1(output2)
    print("output3: ", output3.shape)

    fc2 = nn.Linear(2048, 256)
    output4 = fc2(output3)
    print("output4: ", output4.shape)

    fc3 = nn.Linear(256, 27) 
    output5 = fc3(output4)
    print("output5: ", output5.shape)

if __name__ == "__main__":
    main()
