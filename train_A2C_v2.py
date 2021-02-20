"""
Originally made for LunarLander-v2
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/ActorCritic
"""

import numpy as np
import matplotlib.pyplot as plt

from A2C_v2 import Agent
from car_racing import *

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

SEED = 2
action_choices = [[0,1,0],[1,0,0],[-1,0,0],[0,0,0.8]]


if __name__ == '__main__':
    env = CarRacing(verbose=0)
    env.seed(SEED)
    agent = Agent(gamma=0.99, lr=1e-9, input_dims=[8], n_actions=4,
                  fc1_dims=2048, fc2_dims=512)
    n_games = 1000

    fname = 'ACTOR_CRITIC_' + 'car_racing_' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) +\
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    #eps_greedy = 0.3

    scores = []
    for ep in range(n_games):
        done = False
        observation = env.reset()
        env.seed(SEED)
        score = 0
        while not done:
            env.render()

            action = agent.choose_action(observation)
            #if np.random.random() < eps_greedy:
            #    action = np.random.choice(4)
            
            a = action_choices[action]
            
            observation_, reward, done = env.step(a)
            if not done:
                score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            
            if score < -100:
                done = True

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("Episode {} : Score = {}".format(ep, score))

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)
























































"""
import gym
from A2C_v2 import Agent
import numpy as np


if __name__ == '__main__':
    agent = Agent(act_lr=0.00001, crit_lr=0.00005)

    env = gym.make('LunarLander-v2')
    score_history = []
    num_episodes = 2000

    for ep in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, new_observation, done)
            observation = new_observation
            score += reward
        
        score.history.append(score)
        print("Episode {} : score = {}".format(ep, score))
"""
