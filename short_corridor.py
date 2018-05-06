"""
@author: Ziyang Zhang
"""

import numpy as np
import matplotlib.pyplot as plt

class short_corridor:
    def __init__(self):
        self.state = None
        
    def reset(self):
       self.state = 0
       return self.state
    
    def step(self, action):
        
        if self.state == 0:
            if action == 1: state = 1
            else:           state = 0
                
        elif self.state == 1:
            if action == 0: state = 2
            else:           state = 0
                
        elif self.state == 2:
            if action == 1: return None, 0, True
            else:           state = 1
                
        self.state = state
                
        return self.state, -1, False
        
class Agent:
    
    def __init__(self):
        self.alpha = 2 ** -12
        self.theta = np.array([0.0,0.0])
        self.feature = np.array([[0.0, 1.0],
                                 [1.0, 0.0]])
        self.clear_memory()
        
    def clear_memory(self):
        self.actions = []
        self.observs = []
        self.rewards = []
             
    def choose_action(self, observ):
        h = np.dot(self.theta, self.feature)
        probs = np.exp(h) / np.sum(np.exp(h))
        if np.random.rand() <= probs[0]:
            return 0
        else:
            return 1
    
    def store(self, observ, action, reward):
        self.observs.append(observ)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def learn(self):
        G = np.sum(self.rewards)
        episode = len(self.rewards)
        for t in range(episode):
#            observ = self.observs[t]
            action = self.actions[t]
            h = np.dot(self.theta, self.feature)
            probs = np.exp(h) / np.sum(np.exp(h))
            print(probs)
            dlnp = self.feature[:,action] - np.dot(probs, self.feature.T)
            self.theta += self.alpha * G * dlnp
            G -= self.rewards[t]
        self.clear_memory()
        

env = short_corridor()
agent = Agent()
rewards_trend = []

for episode in range(1000):
    observ = env.reset()
    episode_rewards = 0
    while episode_rewards > -1000:
        action = agent.choose_action(observ)
        observ_, reward, done = env.step(action)
        agent.store(observ, action, reward)
        episode_rewards += reward
        if done:
            break
        observ = observ_
    print('episode:', episode, 'reward:', episode_rewards)
    try:
        rewards_trend.append(0.99*rewards_trend[episode-1]+0.01*episode_rewards)
    except:
        rewards_trend.append(episode_rewards)       
    agent.learn()
    
plt.plot(np.arange(len(rewards_trend)), rewards_trend)
plt.ylabel('Rewards')
plt.xlabel('episode')
plt.show()