"""
Originally made for LunarLander-v2
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/ActorCritic
"""

import numpy as np
from pyglet.window import key
from A2C_v2 import Agent
from car_racing import *


SEED = 2
FRAME_TIME = 4
action_choices = [[0,1.0,0],[1,0.2,0],[-1,0.2,0],[0,0,0.8]]


if __name__ == '__main__':
    env = CarRacing(verbose=0)
    env.seed(SEED)
    agent = Agent(gamma=0.99, lr=3e-4, input_dims=[8], n_actions=4,
                  fc1_dims=256, fc2_dims=48)
    n_games = 2000

    scores = []

    for ep in range(n_games):
        done = False
        observation = env.reset()
        env.seed(SEED)
        score = 0
        frame_counter = 0
        while not done:

            if ep>0:  #commencer à render après x itérations
                env.render()
            if frame_counter == 0:
                action = agent.choose_action(observation)

            else:
                agent.choose_action(observation, action)        
            a = action_choices[action]
            
            observation_, reward, done = env.step(a)

            if frame_counter == 0 and action == 3:
                reward-=2
            else:
                reward+=2
            for i in range(5):
                reward -= 4*int(observation_[i] < observation[i])
            if not done:
                score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            
            #if score < -100:
            #    done = True

            frame_counter +=1
            if frame_counter == FRAME_TIME:
                frame_counter = 0
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("Episode {} : Score = {}, Tiles visited = {}".format(ep, score, env.tile_visited_count))
    agent.save()

























































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
