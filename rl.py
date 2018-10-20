import gym
from collections import deque
import numpy as np
import random
from keras.models import Sequential    
from keras.layers import Dense, Flatten 

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

def initialize_model(obs_space, action_space):

    # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
    model = Sequential()
    model.add(Dense(20, input_shape=(2,) + obs_space.shape, init='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    env = gym.make(DEFAULT_ENV)

    model = initialize_model(env.observation_space, env.action_space)

    D = episode(env = env, render= True)

    state, action, reward, state_new, done = D[0]

    inputs_shape = (1,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((1, env.action_space.n))

    inputs[0:1] = np.expand_dims(state, axis=0)




    
