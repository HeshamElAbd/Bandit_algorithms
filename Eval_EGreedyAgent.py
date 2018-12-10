# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:18:24 2018

@author: Hesham El Abd
@description: Evaluating the performance of a greedy bandit
"""
from EGreedyAgent import Egreedy_agent,env_EGreedyAgent
# construct a greedy agent :
# comparing differet initial reward estimation values: 
# first a=0
# running the random agent 10 times for a 100 time step:
print('Epsilon Greedy with different Values for Epsilon:')
print('epsilon=0')
agent_SumRewards_zero=[] 
for j in range(10):
    greedyAgent =Egreedy_agent(10,0)
    Environment=env_EGreedyAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(100):
        action=greedyAgent.do_action()
        greedyAgent.memory(Environment.return_reward(action))
    agent_SumRewards_zero.append(greedyAgent.sum_rewards())
    greedyAgent.analysis()
print('average reward: \t',
      sum(agent_SumRewards_zero)/len(agent_SumRewards_zero))
input("Press Enter to continue...")

# second a=0.1
print('epsilon=0.01')
agent_SumRewards_p_one=[] 
for j in range(10):
    greedyAgent =Egreedy_agent(10,0.01)
    Environment=env_EGreedyAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(100):
        action=greedyAgent.do_action()
        greedyAgent.memory(Environment.return_reward(action))
    agent_SumRewards_p_one.append(greedyAgent.sum_rewards())
    greedyAgent.analysis()
print('average reward: \t',
      sum(agent_SumRewards_p_one)/len(agent_SumRewards_p_one))
input("Press Enter to continue...")
# third a=1

print('epsilon=0.1')
agent_SumRewards_one=[] 
for j in range(10):
    greedyAgent =Egreedy_agent(10,0.1)
    Environment=env_EGreedyAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(100):
        action=greedyAgent.do_action()
        greedyAgent.memory(Environment.return_reward(action))
    agent_SumRewards_one.append(greedyAgent.sum_rewards())
    greedyAgent.analysis()
print('average reward: \t',
      sum(agent_SumRewards_one)/len(agent_SumRewards_one))
input("Press Enter to continue...")

# fourth a=10:
print('epsilon=0.5')
agent_SumRewards_ten=[] 
for j in range(10):
    greedyAgent =Egreedy_agent(10,0.5)
    Environment=env_EGreedyAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(100):
        action=greedyAgent.do_action()
        greedyAgent.memory(Environment.return_reward(action))
    agent_SumRewards_ten.append(greedyAgent.sum_rewards())
    greedyAgent.analysis()
print('average reward: \t',
      sum(agent_SumRewards_ten)/len(agent_SumRewards_ten))
input("Press Enter to continue...")