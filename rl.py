import gym
from collections import deque
import numpy as np
import random

DEFAULT_ENV = 'CartPole-v0'

def episode(agent = None, env = None, render = False):

    """Runs an episode of the agent"""

    # TODO: Agent argument

    if env is None:
        env = gym.make(DEFAULT_ENV)

    observation =  env.reset()

    obs = np.expand_dims(observation, axis=0) 
    state = np.stack((obs, obs), axis=1)


    D = deque() 
    done = False
    
    for _ in range(1000):
        
        env.render()

        action = env.action_space.sample()

        observation_new, reward, done, info = env.step(action) # take a random action
        
        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1) 

        D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
        state = state_new  

        if done:
           break

    env.close()

    return D

if __name__ == '__main__':

    D = episode(render= True)