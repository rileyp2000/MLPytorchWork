import gym
import numpy as np
import torch

from model import *
from history import *
from visualize import *

algo_name = 'VPG'
num_episodes = 1000
update_iter = 10

#Tracks the previous episodes to be used later
history = History()
#Simulation which we get data from
env = gym.make('CartPole-v1')
#Neural Network
policy = Policy()
#Handles improvement of network
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

past_ep_rewards = []
