from utils import RLEnvironment, RLDebugger

import random

from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Dense, Conv2D, Flatten, Input, Reshape, Lambda, Add, RepeatVector
from keras.models import Sequential, Model
from keras import backend as K


class DQNAgent(RLDebugger):
    def __init__(self, observation_space, action_space):
        RLDebugger.__init__(self)
        # get size of state and action
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n
        # hyper parameters 
        self.gamma = .995
        self.learning_rate = .01
        self.model = self.build_model()  
        self.target_model = self.model
        
    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        #This is a simple one hidden layer model, thought it should be enough here,
        #it is much easier to train with different achitectures (stack layers, change activation)
        model.add(Dense(30, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        # 1/ You can try different losses. As an logcosh loss is a twice differenciable approximation of Huber loss
        # 2/ From a theoretical perspective Learning rate should decay with time to guarantee convergence 
        return model

    # get action from model using greedy policy. 
    def get_action(self, state):
        q_value = self.model.predict(state)
        best_action = np.argmax(q_value[0]) #The [0] is because keras outputs a set of predictions of size 1
        return best_action

    # train the target network on the selected action and transition
    def train_model(self, action, state, next_state, reward, done):
        target = self.model.predict(state)
        # We use our internal model in order to estimate the V value of the next state 
        target_val = self.target_model.predict(next_state)
        # Q Learning: target values should respect the Bellman's optimality principle
        if done: #We are on a terminal state
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * (np.amax(target_val))

        # and do the model fit!
        loss = self.model.fit(state, target, verbose=0).history['loss'][0]
        self.record(action, state, target, target_val, loss, reward)


