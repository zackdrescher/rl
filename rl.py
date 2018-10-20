import gym
from collections import deque
import numpy as np
import random
from keras.models import Sequential    
from keras.layers import Dense, Flatten 

DEFAULT_ENV = 'CartPole-v1'

def episode(model, env = None, render = False, epsilon = 0):
    """Runs an episode of the sumulation with the given Q model"""

    if env is None:
        env = gym.make(DEFAULT_ENV)

    observation =  env.reset()

    obs = np.expand_dims(observation, axis=0) 
    state = np.stack((obs, obs), axis=1)

    D = deque() 
    done = False
    
    for _ in range(1000):
        
        env.render()

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            Q = model.predict(state)          # Q-values predictions
            action = np.argmax(Q)  

        observation_new, reward, done, info = env.step(action) # take a random action
        
        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1) 

        D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
        state = state_new  

        if done:
           break

    env.close()

    return D

def observe(num_obs, model, env = None, render = False, epsilon = 0):
    """Runs an episode of the sumulation with the given Q model"""

    print("Observing...")

    if env is None:
        env = gym.make(DEFAULT_ENV)

    observation =  env.reset()

    obs = np.expand_dims(observation, axis=0) 
    state = np.stack((obs, obs), axis=1)

    D = deque() 
    done = False
    
    for _ in range(num_obs):
        
        env.render()

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            Q = model.predict(state)          # Q-values predictions
            action = np.argmax(Q)  

        observation_new, reward, done, info = env.step(action) # take a random action
        
        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1) 

        D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
        state = state_new  

        if done:
            env.reset()           # Restart game if it's finished
            obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
            state = np.stack((obs, obs), axis=1)

    env.close()

    print("Observation complete")
    return D

def initialize_model(obs_space, action_space):

    print("Intilaizing Model...")

    # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
    model = Sequential()
    model.add(Dense(20, input_shape=(2,) + obs_space.shape, init='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    print("Model Initialized")
    return model

if __name__ == '__main__':

    env = gym.make(DEFAULT_ENV)

    model = initialize_model(env.observation_space, env.action_space)

    D = episode(model, env = env, render= True)

    D2 = observe(1000, model, env = env, render= True)




    
