"""
@author: Ziyang Zhang
"""

from PGagent import PGagent
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env = env.unwrapped

agent = PGagent(env.observation_space.shape[0],
                env.action_space.n)
rewards_trend = []
rewards_trend.append(10) # average random reward

for episode in range(10000):
    observ = env.reset()
    observ = observ[:, np.newaxis]
    episode_reward = 0
    agent.episode_start()
    while True:        
        action = agent.choose_action(observ)
        observ_, reward, done, _ = env.step(action)        
        observ_= observ_[:, np.newaxis]
        agent.learn(reward, observ, observ_, action, done)
        episode_reward += reward
        if done:
            break
        observ = observ_
    if episode % 100 == 0:
        print('episode:', episode, 'reward:', episode_reward)
    exp_r = episode_reward * 0.01 + rewards_trend[episode] * 0.99
    rewards_trend.append(exp_r)
    
plt.plot(np.arange(len(rewards_trend)), rewards_trend)
plt.ylabel('reward')
plt.xlabel('episode')
plt.show()