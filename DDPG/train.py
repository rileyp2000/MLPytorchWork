import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from replay_buffer import ReplayBuffer
from visualize import *
from copy import deepcopy


algo_name = 'DDPG'
max_episode = 1000

gamma = .9
learn_rate = 1e-4

env = gym.make('CartPole-v1')
rb = ReplayBuffer(1e5)

actor = Actor(env)
actor_target = deepcopy(actor)
act_optim = torch.optim.Adam(actor.parameters(), learn_rate)

critic = Critic(env)
critic_target = deepcopy(critic)
crit_optim = torch.optim.Adam(critic.parameters(),lr=learn_rate)

def train():
    explore(10000)
    ep = 0
    while ep < max_episode:
        s = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                a = actor(s)
            s2, r, done, _ = env.step(int(a))
            rb.store(s,a,r,s2,done)
            ep_r += r

            if done:
                update_viz(ep, ep_r, algo_name)
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

def update():
    s, a, r, s2, m = rb.sample(batch_size)
    with torch.no_grad():
        max_next_q, _ = critic_target(s2).max(dim=1, keepdim=True)
        y = r + m*gamma*max_next_q
    critic_loss = F.mse_loss(torch.gather(critic(s), 1, a.long()), y)

    policy_loss = -(torch.gather(critic(s),1, actor(s))).mean()

    #Update q
    crit_optim.zero_grad()
    critic_loss.backward()
    crit_optim.step()

    act_optim.zero_grad()
    policy_loss.backward()
    act_optim.step()

    #Update q_target
    for param, target_param in zip(critc.parameters(), critic_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)
    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)


train()
