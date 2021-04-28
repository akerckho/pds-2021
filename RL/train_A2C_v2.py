"""
Originally made for LunarLander-v2
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/ActorCritic
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import csv 
from A2C_v2 import Agent
#from carbontracker.tracker import CarbonTracker

import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from car_racing import *


def plot_bar_and_mean(raw_lst, mean_lst, lr, gamma, fig_name):
    x = [i for i in range(len(raw_lst))]
    plt.bar(x, raw_lst)
    plt.plot(x, mean_lst, color="tab:red")
    
    plt.title("Learning rate = {}, gamma = {}".format(lr, gamma))
    plt.xlabel("Epoque")
    plt.ylabel("Nombre de tiles parcourues")
    plt.savefig(fig_name)

def reward_manage(reward, state, action, speed):
    """
    action 0 = turn right
    action 1 = accelerate
    action 2 = turn left
    action 3 = brake
    """
    noise = np.random.normal(0,.1,state.shape) # Ajout de bruit pour différencier les sensors de longueurs égales (en forcant un peu d'exploration)
    new_state = state + noise
    good_move = np.argmax(new_state)
    good_move_mapper = [
        [0,1,2,3,4],      # kinda left
        [5,6,7],          # forward
        [8,9,10,11,12]    # kinda right
    ]

    if action == 1 and ((state[5]+state[6]+state[7]) > 3*44): 
        reward += 10

    if action == 3:
        if speed > 50:   # mollo sur le champignon Michel
            reward += 20
        if speed < 5:
            reward -= 20  # on reste pas à l'arrêt par contre
    elif good_move in good_move_mapper[action]:
        reward += 15
    else:
        reward -= 20

    return reward


ENVIRONMENT_SEED = 2 
action_choices = [[0.2,0.1,0],[0,1,0],[-0.2,0.1,0],[0,0,1]]
inputs = 14

if __name__ == '__main__':
    

    if len(sys.argv) != 3:
        print("A parameter is missing. Please enter seed then obstacle porbability.\nInterupting.")
        sys.exit()
    
    iteration_seed = int(sys.argv[1])
    obstacle_prob = float(sys.argv[2])

    l_rate = 0.000008
    gamma = 0.99
    n_games = 2000

    agent = Agent(gamma=gamma, lr=l_rate, input_dims=[inputs], n_actions=4, seed=iteration_seed, fc1_dims=2048, fc2_dims=1024)
    
    env = CarRacing(obstacle_prob, verbose=0)
    env.seed(ENVIRONMENT_SEED)
    
    tile_visited_history = []
    avg_tile_visited_history = []
    
    #tracker = CarbonTracker(epochs=n_games, epochs_before_pred=n_games//10, monitor_epochs=n_games, components="gpu", verbose=2)
    max_tiles = 0
    for ep in range(n_games):
        #tracker.epoch_start()
        done = False
        observation = env.reset()
        env.seed(ENVIRONMENT_SEED)
        score = 0
        
        while not done:
            
            pre = env.tile_visited_count
            
            action, bonus = agent.choose_action(observation)   
            a = action_choices[action]
            pre = env.tile_visited_count
            observation_, reward, done= env.step(a)
            post = env.tile_visited_count
            reward = reward_manage(reward, observation, action, observation[-1])
            agent.learn(observation, reward, observation_, done)
            observation = observation_
                
            if not done:
                score += reward
            
            tiles_visited = env.tile_visited_count
            max_tiles = max(max_tiles, tiles_visited)

        ## RUNTIME INFORMATION
        print("Episode {} : Score = {} ".format(ep, score))
        print("Tiles visited : {} (max {}) ".format(tiles_visited, max_tiles))

        # Data that will be plotted
        tile_visited_history.append(tiles_visited)
        avg_tile_visited = round(np.mean(tile_visited_history[-100:]),2)
        avg_tile_visited_history.append(avg_tile_visited)
        
        #tracker.epoch_end()

    #tracker.stop()

    ## LOGS  
    with open(str(obstacle_prob) + '_tilesVisited.csv', 'a+', newline ='') as tv_file:    
        write = csv.writer(tv_file)
        write.writerow(tile_visited_history)

    with open(str(obstacle_prob) + '_tilesVisitedAverage.csv', 'a+', newline ='') as tva_file:    
        write = csv.writer(tva_file)
        write.writerow(avg_tile_visited_history)

    ## PLOT
    #savename = "A2C_{}epochs_{}lr_{}gamma.png".format(n_games, l_rate, gamma)
    #plot_bar_and_mean(tile_visited_history, avg_tile_visited_history, l_rate, gamma, savename )

    ## SAVE MODEL
    #model = copy.deepcopy(agent.get_model().state_dict())
    #torch.save(model, "modelA2Cv2/unnamed")

