import gym
from collections import deque
import numpy as np
import random
from keras.models import Sequential    
from keras.layers import Dense, Flatten 
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEFAULT_ENV = 'CartPole-v1'

def episode(model, env = None, render = False, epsilon = 0):
    """Runs an episode of the sumulation with the given Q model
    Epsilon is the probability of doing a random move"""

    if env is None:
        env = gym.make(DEFAULT_ENV)

    observation =  env.reset()

    obs = np.expand_dims(observation, axis=0) 
    state = np.stack((obs, obs), axis=1)

    D = deque() 
    done = False
    
    while not done:
        
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
    model.add(Dense(20, input_shape=(2,) + obs_space.shape, kernel_initializer='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(action_space.n, kernel_initializer='uniform', activation='linear'))    # Same number of outputs as possible actions

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    print("Model Initialized")
    return model

def learn(model, D, mini_batch_size = None):

    print("Learning...")

    if mini_batch_size is None:
        mini_batch_size = len(D)

    batch = random.sample(D, mini_batch_size)                              # Sample some moves

    X, Y = bellman(batch, model)

    # Train network to output the Q function
    model.train_on_batch(X, Y)

    print("Learning complete")
    
def bellman(D, model, gamma = 0.9):

    m = len(D)

    state = D[0][0]
    inputs_shape = (m,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((m, model.output_shape[1]))

    for i, (state, action, reward, state_new, done) in enumerate(D):
        
        # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)
        
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

    return inputs, targets

if __name__ == '__main__':

    env = gym.make(DEFAULT_ENV)

    model = initialize_model(env.observation_space, env.action_space)

    D = observe(1000, model, env = env, render= True, epsilon=1)

    learn(model, D)

    D2 = observe(1000, model, env=env, render=True)




    
