from typing import Counter
from car_racing import *
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = np.random.randint(1,100000)


def reward_manage(reward, pre_state, action):
    if action != 0:
        reward+=10
    else:
        reward-=21

    return reward

def reward_history_manage(tile_visited_count, tiles, rewards_history):
    progression = tile_visited_count/tiles
    malus = 10*(progression)
    rewards_history -= malus
    return rewards_history


if __name__ == "__main__":
    


    render = False
    

    num_inputs = 5
    num_actions = 4
    num_hidden = 10

    action_choices = [[0,1,0],[0,0,0.8],[1,0.3,0],[-1,0.3,0]]
    if len(sys.argv) < 3:
        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(100, activation="tanh")(inputs)
        common2 = layers.Dense(25, activation="tanh")(common)
        #dropout = layers.Dropout(0.1, noise_shape=None, seed=None)
        action = layers.Dense(num_actions, activation="softmax")(common2)
        critic = layers.Dense(1)(common2)
        model = keras.Model(inputs=inputs, outputs=[action, critic])
    elif len(sys.argv) == 3 and sys.argv[2] == "my-model":
        model = keras.models.load_model("./mon_model") #charger modèle existant
    else:
        print("Mauvais encodage")
        exit()
    gamma = 0.99  # Discount factor for past rewards
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    env = CarRacing()
    if len(sys.argv) < 2:
        greeds = [0, 0.7]
    else:
        greeds = [float(sys.argv[1]), float(sys.argv[1])]
    n = 1

    env.seed(SEED)   # seed the circuit 
    
    if render:
        env.render()

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = np.array([])
    running_reward = 0
    episode_count = 0
    max_episode = 1000
    state = []
    retard = 9
    while episode_count < max_episode:
        checkpoint = 10
        state = env.reset()
        contacts = [0 for i in range(len(env.car.sensors))]
        env.seed(SEED)   # seed the circuit 
        #env.setAngleZero()
        episode_reward = 0
        steps = 0
        
        restart = False
        env.render()
        render = True

        frame_counter=0
        tiles_nb = len(env.track)
        with tf.GradientTape() as tape:
            while True:

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                if frame_counter==0:
                    # Predict action probabilities and estimated future rewards
                    # from environment state
                    action_probs, critic_value = model(state)
                    critic_value_history.append(critic_value[0, 0])
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                    # Sample action from action probability distribution
                    if np.random.random() < greeds[n]: #discoverability
                        action_probs = np.random.random(4)
                        action_probs /= action_probs.sum()
                        action_probs = np.array([action_probs], dtype = float)
                        action_probs = tf.convert_to_tensor(action_probs)
                        action_probs = tf.cast(action_probs, tf.float32)
                        action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                        action =np.random.choice(num_actions)

                    
                    action_probs_history.append(tf.math.log(action_probs[0, action]))
                    #print(action_probs)
                    a = action_choices[action]

                state,reward, done, contacts = env.step(a)
                #print(state)
                """print("Coté droit : ", state[:-3])
                print("Coté Gauche : ", state[-2:])
                print("Droit devant : ", state[1:4])"""
                if action == 0 and np.sum(state[1:4]) < 25:
                    #print("Tu vas te cogner et tu veux avancer ?")
                    reward -= 100
                if action == 2 and np.sum(state[:-3]) < 25:
                    #print("mauvaise idée")
                    reward -= 20
                elif action == 3 and np.sum(state[-2:]) < 25:
                    #print("mauvaise idée")
                    reward -= 20

                visited = env.tile_visited_count
                if visited > checkpoint:
                    reward += 600
                    checkpoint += 5

                #reward += 3000/len(env.track)
                #reward -= 0.1
                reward = reward_manage(reward, state, action)

                #print(reward)
                rewards_history = np.append(rewards_history, reward)
                episode_reward += reward
                """
                if steps % 200 == 0 or done:
                    print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                    print("step {} total_reward {:+0.2f}".format(steps, episode_reward))
                """
                steps += 1
                frame_counter+=1
                if frame_counter==8:
                    frame_counter=0
                if render:
                    isopen = env.render()
                if episode_reward< -5000 and not done:#arrête si le reward total est trop bas (voiture presque immobile)
                    done = True
                if done or restart : 
                    break
            tile_visited_count = env.tile_visited_count

            #rewards_history = reward_history_manage(tile_visited_count, tiles_nb, rewards_history)
            """print("REWARDS !!!!!", rewards_history)
            print("Somme : ", np.sum(rewards_history))"""
            if visited < retard:
                rewards_history -= 30000
            episode_reward = np.sum(rewards_history)
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history = np.array([])
        # Log details
        episode_count += 1
        print("{} tiles visited on run,episode reward {}".format(env.tile_visited_count,episode_reward))
        print("episode {}, greed = {}".format(episode_count, greeds[n]))
        
        if episode_count % 10 == 0:
            template = "running reward: {} at episode {}"
            print(template.format(env.times_succeeded, episode_count))
        if episode_count %100 == 0:
            model.save("./model")
            n = (n+1)%2
        if env.times_succeeded > 5:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break
    model.save("./mon_model")
    env.close()