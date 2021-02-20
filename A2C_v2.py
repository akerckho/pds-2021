"""
Originally made for LunarLander-v2
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/ActorCritic
"""

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)

class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, 
                 gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)    
        self.log_prob = log_prob
        
        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()
























































"""
Adaptation of a model initially designed for LunarLander-v2 environment of gym
Thanks to : https://www.youtube.com/watch?v=2vJtbAha3To&ab_channel=MachineLearningwithPhil


from keras import backend as K 
from keras.layers import Dense, Input
from keras.models import Model 
from keras.optimizers import Adam
import numpy as np 

class Agent(object):
    def __init__(self, act_lr, crit_lr, gamma=0.99, n_actions=4, 
                layer1_size=1024, layer2_size=512, input_dims=8):
        self.act_lr = act_lr
        self.crit_lr = crit_lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.input_dims = input_dims

        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]

    def build_actor_critic_network(self):
        inp = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(inp)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)

        actor = Model(inputs=[inp, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.act_lr), loss=custom_loss)

        critic = Model(inputs=[inp], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.crit_lr), loss='mean_squared,error')
        
        policy = Model(inputs=[inp], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action
    
    def learn(self, state, action, reward, new_state, done):
        state = state[np.newaxis, :]
        new_state = new_state[np.newaxis, :]

        new_critic_value = self.critic.predict(new_state)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*new_critic_value*(1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0

        print(actions)
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)
"""










