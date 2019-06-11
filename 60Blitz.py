import numpy as np
import pickle
import gym

H = 200 #neurons in hidden layer
batch_size = 10 #interval of parameter update
learning_rate = 1e-4
gamma = .99 #Discount factor 
