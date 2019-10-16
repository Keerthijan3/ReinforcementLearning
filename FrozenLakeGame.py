import numpy as np
import gym
import random
import time

env = gym.make("FrozenLake-v0")
q_table = np.zeros((env.observation_space.n, env.action_space.n))

totalepisodes = 10000
stepsperepisode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001

rewards_all_episodes = []
for episode in range(totalepisodes):
    state = env.reset()
    done = False
    rewards = 0
    for step in range(stepsperepisode):
        thres = random.random()
        if thres > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action)
        q_table[state, action] = q_table[state, action]*(1-learning_rate) + learning_rate*(reward + discount_rate*np.max(q_table[new_state, :]))
        state = new_state
        rewards += reward       
        if done:
            break

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode) 
    rewards_all_episodes.append(rewards)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), totalepisodes/1000)
count = 1000

print('Average Reward Per Thousand Episodes')
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

print('Q TABLE')
print(q_table)
num_tests = 100
print('Running Policy ', num_tests, ' Times')
correct = 0
for test in range(num_tests):
    state = env.reset()
    done = False
    for step in range(stepsperepisode):
        action = np.argmax(q_table[state,:])
        new_state, reward, done, _ = env.step(action)
        if done:
            if reward==1:
                print('Reached Goal')
                correct+=1         
            else:
                print('Fell Into Hole')
            break
        state=new_state
env.close()
print('Total Correct: ', correct/num_tests)

