import gym
from collections import deque
import numpy as np
import random
from keras.models import Sequential    
from keras.layers import Dense, Flatten 
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEFAULT_ENV = 'CartPole-v1'

def evaluate_model(num_eps, model, env = None, render = False, epsilon = 0):

    r = []
    D = deque()

    for i in tqdm(range(num_eps), desc = 'Evaluating...'):
        d, total_reward = episode(model, env, render, epsilon)
        
        D += d
        r.append(total_reward)

    return r, D

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
    total_reward = 0

    while not done:
        
        if render:
            env.render()

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            Q = model.predict(state)          # Q-values predictions
            action = np.argmax(Q)  

        observation_new, reward, done, info = env.step(action) # take a random action
        total_reward += reward

        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1) 

        D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
        state = state_new  

    env.close()

    return D, total_reward

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
    
    for _ in tqdm(range(num_obs)):
        
        if render:
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

def learn(model, D, sample_size = 2000, epochs = 3, minibatch = 64):

    print("Learning %s expiriences %s times with minibathes %s" % (sample_size, epochs, minibatch))

    batch = random.sample(D, sample_size)                              # Sample some moves

    X, Y = bellman(batch, model)

    # Train network to output the Q function
    model.fit(X, Y, epochs = epochs, batch_size= minibatch)

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

def plot_rewards(d):

    m = [max(v) for v in d.values()]
    m = max(m)
    bins = np.linspace(0, m, 10)

    plt.Figure()
    for k, v in d.items():
        plt.hist(v, bins = bins, alpha = 0.5, label = k)
    
    plt.legend()
    plt.show()

def learn_n_eval(env, model, D, eval_steps = 20, sample_size = None, epochs = None):

    # LEARN
    learn(model, D, sample_size= sample_size, epochs= epochs)

    # EVALUATE
    print('Evaluating Model...')
    model_r, model_d = evaluate_model(20, model, env, True, epsilon=0)
    print('Evaluating random...')
    random_r, random_d = evaluate_model(20, model, env, True, epsilon=1)

    

    d = {'model1' : model_r, 'random' : random_r}
    plot_rewards(d)

    return  model_d + random_d

if __name__ == '__main__':

    # INIT
    env = gym.make(DEFAULT_ENV)

    model = initialize_model(env.observation_space, env.action_space)

    # FIRST RANDOM OBSERVATION
    D = observe(10000, model, env = env, epsilon=1)

    D += learn_n_eval(env, model, D, sample_size= 4 * 1024, epochs=3)

    for i in range(20):
        
        print("Observation %s" % i)
        D += observe(10000, model, env = env, epsilon=0.9 ** i)

        D += learn_n_eval(env, model, D, sample_size= (i+1) * 8 * 1024, epochs=3)





    
