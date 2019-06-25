import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy

from model import Q,H
from replay_buffer import ReplayBuffer, HumanReplayBuffer
from history import History
from visualize import *
from humanFeedback import Human

algo_name = 'DQN-TAMER'
env = gym.make('LunarLander-v2')
max_ep = 1000
epsilon = .3
gamma = .99
human = Human()
alpha_q = 1
alpha_h = alpha_q
random.seed(5714149178)

#Proportion of network you want to keep
tau = .995

q = Q(env)
q_target = deepcopy(q)

h = H(env)
h_target = deepcopy(h)

q_optim = torch.optim.Adam(q.parameters(), lr=1e-2)
h_optim = torch.optim.Adam(h.parameters(), lr=1e-2)


batch_size = 128
rb = ReplayBuffer(1e6)

h_batch = 16
human_rb = HumanReplayBuffer(1e4)
local_batch = History()

#Training the network
def train():
    explore(10000)
    ep = 0
    while ep < max_ep:
        s = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                #Epsilon greedy exploration
                if random.random() < epsilon:
                    a = int(env.action_space.sample())
                else:
                    adj_q = alpha_q * q(s)
                    adj_h = alpha_h * h(s)
                    a = int(np.argmax(adj_q + adj_h))
            decay()
            #Get the next state, reward, and info based on the chosen action
            s2, r, done, _ = env.step(a)
            rb.store(s, a, r, s2, done)

            if not done:
                local_batch.store(s, a, h(s))
            if random.random() < .2:
                f = human.evaluate(s)
                if f != 0:
                    updateHLocal(f)
            ep_r += r

            #If it reaches a terminal state then break the loop and begin again, otherwise continue
            if done:
                update_viz(ep, ep_r, algo_name)
                ep += 1
                break
            else:
                s = s2

            update()


#Explores the environment for the specified number of timesteps to improve the performance of the DQN
def explore(timestep):
    ts = 0
    while ts < timestep:
        s = env.reset()
        while True:
            a = env.action_space.sample()
            s2, r, done, _ = env.step(int(a))
            rb.store(s, a, r, s2, done)
            ts += 1
            if done:
                break
            else:
                s = s2

#Updates both the Q and H networks according to the DQN-TAMER Algorithm
def update():

    updateQ()
    if human_rb.length() != 0:
        updateH()

    updateTargets()

#Updates the Q by taking the max action and then calculating the loss based on a target
def updateQ():
    s, a, r, s2, m = rb.sample(batch_size)

    with torch.no_grad():
        max_next_q, _ = q_target(s2).max(dim=1, keepdim=True)
        y = r + m*gamma*max_next_q

    loss = F.mse_loss(torch.gather(q(s), 1, a.long()), y)
    optim(loss,q_optim)

#Updates the H network by taking the MSE of the H function and the actual feedback
def updateH():
    #Use the global "replay buffer"  to sample from
    s_h, a_h, f = human_rb.sample(h_batch)
    loss = F.mse_loss(h(s_h), f)
    optim(loss, h_optim)

#Updates the H network using the local batch
def updateHLocal(f):
    s, a, h_pred = local_batch.get_history()
    # print("f: " + str(f))
    # print(len(s))
    discount = f * alpha_h
    feedback_discounted = [0] * len(s)

    for i in reversed(range(len(feedback_discounted))):
        feedback_discounted[i] = discount
        discount = feedback_discounted[i]*alpha_h

    #print(feedback_discounted)

    max_h, _ = h(s).max(dim=1, keepdim=True)
    feedback_discounted = torch.FloatTensor(feedback_discounted)

    loss = F.mse_loss(max_h.squeeze(), feedback_discounted)
    #print(loss)
    optim(loss, h_optim)

#Optimizer encapsulation method
def optim(loss,optim):
    #Update q
    optim.zero_grad()
    loss.backward()
    optim.step()

#Updates the target networks for H and Q
def updateTargets():
    #Update q_target
    for param, target_param in zip(q.parameters(), q_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)

    for param, target_param in zip(h.parameters(), h_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)

#Deacys the epsilon and alpha_h values as the network steps through
def decay():
    global alpha_h, epsilon
    alpha_h *= .9999
    if epsilon > .1:
        epsilon *= .999


train()
