from rl import initialize_model

import gym
from collections import deque
import numpy as np
import random
from keras.models import Sequential    
from keras.layers import Dense, Flatten 
from tqdm import tqdm
import matplotlib.pyplot as plt

# Get rid of warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEFAULT_ENV = 'CartPole-v1'
DEFAULT_MEM = 10000

class DQN:

    def __init__(self, env = DEFAULT_ENV, replay_mem = DEFAULT_MEM, 
                phi_len = 4):

        self.env = gym.make(env)
        self.D = deque(maxlen = DEFAULT_MEM)
        self.phi_len = phi_len

        self.model = self.initialize_model()

    def initialize_model(self):

        print("Intilaizing Model...")

        # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
        model = Sequential()
        model.add(Dense(
            20, input_shape=(self.phi_len,) + self.env.observation_space.shape, 
            kernel_initializer='uniform', activation='relu'))
        model.add(Flatten())       # Flatten input so as to have no problems with processing
        model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(
            self.env.action_space.n, kernel_initializer='uniform', 
            activation='linear'))    # Same number of outputs as possible actions

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        print("Model Initialized")
        return model

    def phi(self, s, x):
        """ s.shape = (phi_len, obs_space)"""

        s = np.array(s)

        # Shift the sequence down
        s[1:] = s[:self.phi_len - 1]

        # add the new state to the top
        s[:1] = x

        return s
    
    def episode(self, epsilon = 0.9, minibatch_size = 32, render = False):

        # intialize sequence
        s = np.array([self.env.reset(),] * self.phi_len)

        done = False
        total_reward = 0
        while not done:

            if render:
                self.env.render()

            # Select action with random action occuring with probability
            # epslion
            if np.random.rand() <= epsilon:
                action = self.env.action_space.sample()
            else:
                Q = self.model.predict(np.expand_dims(s, axis = 0))
                action = np.argmax(Q)  

            # take the action
            x_new, reward, done, info = self.env.step(action)
            total_reward += reward
            # get the new state
            s_new = self.phi(s, x_new)
            self.D.append((s, action, reward, s_new, done))
            s = s_new
            
            if minibatch_size < len(self.D):
                X, Y = self.bellman(minibatch_size)
            else:
                X, Y = self.bellman(len(self.D))

            self.model.fit(X,Y, epochs=1, verbose=0)

        self.env.close()
        #print(total_reward)
        return total_reward

    def bellman(self, batch_size, gamma = 0.9):

        """batch.shape = (len(d_batch), phi_len, obs_space)"""

        d_batch = random.sample(self.D, batch_size)

        inputs = np.zeros((batch_size, self.phi_len, ) 
                            + self.env.observation_space.shape)
        targets = np.zeros((batch_size, self.model.output_shape[1]))

        for i, (state, action, reward, state_new, done) in enumerate(d_batch):

            inputs[i] = state
            targets[i] = self.model.predict(np.expand_dims(state, axis = 0))

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = (reward 
                                     + gamma 
                                     * np.amax(self.model.predict(
                                         np.expand_dims(state_new, axis = 0))))

        return inputs, targets

    def run_episodes(self, num_episodes, min_epsilon = .1, min_prop = .1, render = False):
        d = []
        epsilon = np.linspace(1, min_epsilon, num_episodes * (1-min_prop))

        for i in range(num_episodes):

            if i >= len(epsilon):
                e = min_epsilon
            else:
                e = epsilon[i]

            x = (i, e, self.episode(epsilon= e, render=render))
            print(x)
            d.append(x)

        return d

if __name__ == '__main__':

    m = DQN()

    d = m.run_episodes(1000)

    








