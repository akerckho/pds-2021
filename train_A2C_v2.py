"""
Originally made for LunarLander-v2
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/ActorCritic
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from A2C_v2 import Agent, OUActionNoise
from car_racing import *

from carbontracker.tracker import CarbonTracker

SEED = 5


def reward_manage(reward, state, action, speed):
    """
    state[0] = right sensor
    state[1] = middle sensor
    state[2] = left sensor

    action 0 = turn right
    action 1 = accelerate
    action 2 = turn left
    action 3 = brake
    """
    #print(speed)
    good_move = np.argmax(state)
    if speed > 20 and action != 3:
        reward -= 20
    elif speed > 20 and action == 3:
        reward += 20
    elif speed < 20 and action == 3:
        reward -= 20
    elif action == good_move:
        reward += 10
    else:
        reward -= 20


    return reward


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

 #SEED = 2
action_choices = [[1,0.3,0],[0,1,0], [-1,0.3,0],[0,0,1]]
inputs = 4

if __name__ == '__main__':
    env = CarRacing(verbose=0)
    env.seed(SEED)
    #l_rate = 1e-9
    l_rate = 0.0000001
    if len(sys.argv) > 1:
    	model_weight = sys.argv[1]
    	agent = Agent(gamma=0.99, lr=l_rate, input_dims=[inputs], n_actions=4,
                  fc1_dims=2048, fc2_dims=1024)
    	agent.load_model(model_weight)
    else: 
    	agent = Agent(gamma=0.99, lr=l_rate, input_dims=[inputs], n_actions=4,
                  fc1_dims=2048, fc2_dims=1024)
    n_games = 10000

    fname = 'ACTOR_CRITIC_' + 'car_racing_' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) +\
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    #std_dev = 0.2
    #ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    
    eps_greedy = 0.2
    frame_number = 0
    scores = []
    greeds =  [0, 0]
    n = 0
    #tracker = CarbonTracker(epochs=n_games, verbose=2)

    for ep in range(n_games):
        #tracker.epoch_start()
        done = False
        observation = env.reset()
        env.seed(SEED)
        score = 0
        
        while not done:
            if ep >= 1000:
                env.render()
            pre = env.tile_visited_count
            if frame_number == 0:
                
                action, bonus = agent.choose_action(observation, eps_greedy)   
                a = action_choices[action]
                if a == 1:
                    reward += 2
                elif a == 2:
                    reward += 2 
                pre = env.tile_visited_count
                observation_, reward, done= env.step(a)
                post = env.tile_visited_count

                #if not done:
                #    score += reward
                progression = False

                if post > pre:
                    progression = True

                reward = reward_manage(reward, observation, action, observation[-1])

                agent.learn(observation, reward, observation_, done)
                observation = observation_
                
                #if score < -100:
                #    done = True
            else:
                pre = env.tile_visited_count
                _, reward, done = env.step(a)
                post = env.tile_visited_count

                #if not done:
                #    score += reward
                progression = False

                if post > pre:
                    progression = True
                    
                reward = reward_manage(reward, observation, action, observation[-1])

            if not done:
                score += reward

            #if eps_greedy > eps_greed_min:
            #    eps_greedy -= 0.0000001  # Je sais pas trop c'est un test
            
            frame_number += 1
            if frame_number == 8:
                frame_number=0
            tiles_visited = env.tile_visited_count
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("Episode {}, e_greedy = {} : Score = {} ".format(ep, eps_greedy, score))
        print("Tiles visited : ", tiles_visited)
        #tracker.epoch_end()

    #tracker.stop()

    x = [i+1 for i in range(n_games)]
    name = "modelAC2v2/model"
    #plot_learning_curve(x, scores, figure_file)
    model = copy.deepcopy(agent.get_model().state_dict())
    torch.save(model, "modelAC2v2/model")
























































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
