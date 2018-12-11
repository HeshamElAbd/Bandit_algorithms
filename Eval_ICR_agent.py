# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:40:07 2018

@author: ESC
"""

from Incremental_greedy_agent import ICR_greedy_agent,env_ICR_GreedyAgent
# construct a greedy agent :
# comparing differet initial reward estimation values: 
# first a=0
# running the random agent 10 times for a 100 time step:
print('Incremental Greedy agent with different Values for the reward:')
print('initial rewards=0')
agent_SumRewards_zero=[] 
for j in range(10):
    greedyAgent =ICR_greedy_agent(10,0)
    Environment=env_ICR_GreedyAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(100):
        action=greedyAgent.do_action()
        greedyAgent.update_memory(Environment.return_reward(action))
    agent_SumRewards_zero.append(greedyAgent.sum_rewards())
print('average reward: \t',
      sum(agent_SumRewards_zero)/len(agent_SumRewards_zero))
input("Press Enter to continue...")

# second a=0.1
print('initial rewards=0.1')
agent_SumRewards_p_one=[] 
for j in range(10):
    greedyAgent =ICR_greedy_agent(10,0.1)
    Environment=env_ICR_GreedyAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(100):
        action=greedyAgent.do_action()
        greedyAgent.update_memory(Environment.return_reward(action))
    agent_SumRewards_p_one.append(greedyAgent.sum_rewards())
print('average reward: \t',
      sum(agent_SumRewards_p_one)/len(agent_SumRewards_p_one))
input("Press Enter to continue...")
# third a=1

print('initial rewards=1')
agent_SumRewards_one=[] 
for j in range(10):
    greedyAgent =ICR_greedy_agent(10,1)
    Environment=env_ICR_GreedyAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(100):
        action=greedyAgent.do_action()
        greedyAgent.update_memory(Environment.return_reward(action))
    agent_SumRewards_one.append(greedyAgent.sum_rewards())
print('average reward: \t',
      sum(agent_SumRewards_one)/len(agent_SumRewards_one))
input("Press Enter to continue...")

# fourth a=10:
print('initial rewards=10')
agent_SumRewards_ten=[] 
for j in range(10):
    greedyAgent =ICR_greedy_agent(10,10)
    Environment=env_ICR_GreedyAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(100):
        action=greedyAgent.do_action()
        greedyAgent.update_memory(Environment.return_reward(action))
    agent_SumRewards_ten.append(greedyAgent.sum_rewards())
print('average reward: \t',
      sum(agent_SumRewards_ten)/len(agent_SumRewards_ten
         ))
input("Press Enter to continue...")