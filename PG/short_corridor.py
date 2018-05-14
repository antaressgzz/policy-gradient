"""
This is an realization of example 13.1 on 'Reinforcement Learning: An Introduction'
by Sutton and Barto(v2018), a 3 grid corridor. In state 0 and 1, action 0 leads
to left(if in state 0 then no change happens) and action 1 leads to right, 
while in state 2, the result of the two actions are reversed. All 3 states are 
indistinguishable to the agent(same observation). The optimal result is stocastic,
probability of 0.59 turn right(action 1) and reward is about 11(on average).

@author: Ziyang Zhang
"""

import numpy as np
import matplotlib.pyplot as plt

class short_corridor:       
    def reset(self):
       self.state = 0
       return self.state
    
    def step(self, action):      
        if self.state == 0:
            if action == 1: state = 1 # action 1 leads to right
            else:           state = 0               
        elif self.state == 1:
            if action == 0: state = 2
            else:           state = 0 # action 1 leads to left               
        elif self.state == 2:
            if action == 1: return None, 0, True
            else:           state = 1               
        self.state = state                
        return self.state, -1, False
        
class Agent:
    
    def __init__(self):
        self.alpha = 2 ** -12
        self.theta = np.zeros(2)
        self.feature = np.array([[0.0, 1.0],
                                 [1.0, 0.0]])
        self.probs = []
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
            action = self.actions[t]
            h = np.dot(self.theta, self.feature)
            probs = np.exp(h) / np.sum(np.exp(h))
            self.probs.append(probs[1]) # record the policy(prob to choose action 1)
            dlnp = self.feature[:,action] - np.dot(probs, self.feature.T)
            self.theta += self.alpha * G * dlnp
            G -= self.rewards[t]
        self.clear_memory()


env = short_corridor()
agent = Agent()
rewards_trend = []
moving_average = []

for episode in range(3000):
    observ = env.reset()
    episode_rewards = 0
    while True:
        action = agent.choose_action(observ)
        observ_, reward, done = env.step(action)
        agent.store(observ, action, reward)
        episode_rewards += reward
        if done:
            break
        observ = observ_
    print('episode:', episode, 'reward:', episode_rewards)
    rewards_trend.append(episode_rewards)
    if episode > 100:
        moving_average.append(np.sum(rewards_trend[-100:]) / 100)
    agent.learn()
    
plt.plot(np.arange(len(moving_average)), moving_average)
plt.ylabel('Rewards')
plt.xlabel('episode')
plt.show()

plt.plot(np.arange(len(agent.probs)), agent.probs)
plt.ylabel('probs to right')
plt.xlabel('episode')
plt.show()