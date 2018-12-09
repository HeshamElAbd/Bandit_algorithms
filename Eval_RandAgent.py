# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:03:56 2018

@author: Hesham El Abd
Evalating the performance of a random bandit agent 
"""
from RandomAgent import env_RandAgent,random_agent
###############################################################################
# construct the agent and the environment 
Rand_agent=random_agent(5)
Environment=env_RandAgent(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99])
# running the random agent for a 1000 time step
for i in range(1000):
    action=Rand_agent.do_action()
    Rand_agent.memory(Environment.return_reward(action))
# the sum of reward by playing randomly 
Randomagent_SumRewards=Rand_agent.sum_rewards()
