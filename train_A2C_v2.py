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

SEED = 2


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

    good_move = np.argmax(state)
    good_move_mapper = [
        [0,1,2,3,4],  # kinda left
        [5,6,7],    # kinda forward
        [8,9,10,11]   # kinda right
    ]
    if speed > 20 and action != 3:
        reward -= 20
    elif speed > 20 and action == 3:
        reward += 20
    elif speed < 20 and action == 3:
        reward -= 20
    elif good_move in good_move_mapper[action]:
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

SEED = 2
action_choices = [[1,0.1,0],[0,1,0],[-1,0.1,0],[0,0,1]]
inputs = 14

if __name__ == '__main__':
    env = CarRacing(verbose=0)
    env.seed(SEED)
    #l_rate = 1e-9
    l_rate = 0.000008
    if len(sys.argv) > 1:
    	model_weight = sys.argv[1]
    	agent = Agent(gamma=0.99, lr=l_rate, input_dims=[inputs], n_actions=4,
                  fc1_dims=2048, fc2_dims=1024)
    	agent.load_model(model_weight)
    else: 
    	agent = Agent(gamma=0.99, lr=l_rate, input_dims=[inputs], n_actions=4,
                  fc1_dims=2048, fc2_dims=1024)
    n_games = 1000

    fname = 'ACTOR_CRITIC_' + 'car_racing_' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) +\
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    frame_number = 0
    tile_visited_history = []
    avg_tile_visited_history = []
    scores = []
    
    tracker = CarbonTracker(epochs=n_games, epochs_before_pred=n_games//10, monitor_epochs=n_games, verbose=2)
    max_tiles = 0
    for ep in range(n_games):
        tracker.epoch_start()
        done = False
        observation = env.reset()
        env.seed(SEED)
        score = 0
        
        while not done:
            if ep >= 0:
                env.render()
            pre = env.tile_visited_count
            if frame_number == 0:
                
                action, bonus = agent.choose_action(observation)   
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
            
            frame_number += 1
            if frame_number == 1:
                frame_number=0
            tiles_visited = env.tile_visited_count
            max_tiles = max(max_tiles, tiles_visited)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("Episode {} : Score = {} ".format(ep, score))
        print("Tiles visited : {} (max {}) ".format(tiles_visited, max_tiles))

        # Data that will be plotted
        tile_visited_history.append(tiles_visited)
        avg_tile_visited = round(np.mean(tile_visited_history[-100:]),2)
        avg_tile_visited_history.append(avg_tile_visited)
        
        tracker.epoch_end()

    tracker.stop()

    print()
    print()
    print()
    print()
    print()
    print("Historique des nombres de tiles visitées par épisode : ")
    print(tile_visited_history)
    print()
    print()
    print()
    print()
    print()
    print("Historique des moyennes de tiles visitées sur 100 époques passées :")
    print(avg_tile_visited_history)
    x = [i+1 for i in range(n_games)]
    name = "modelAC2v2/model"
    #plot_learning_curve(x, scores, figure_file)
    model = copy.deepcopy(agent.get_model().state_dict())
    torch.save(model, "modelA2Cv2/modelAllFrames1000epochs")

