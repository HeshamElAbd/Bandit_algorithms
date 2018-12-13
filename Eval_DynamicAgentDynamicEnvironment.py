# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:35:20 2018
@author: Hesham El Abd
"""
from DynamicAgentDynamicEnvironment import DynamicAgent,DynamicEnv
# comparing differet initial reward estimation values: 
# first a=0
# running the random agent 10 times for a 100 time step:
print('Incremental Greedy agent with different learning step size: ')
print('initial reward estimation=1 and step size of 0.1, default parameter '+
      'The Environment')
agent_SumRewards_StepSize_p_one=[] 
for j in range(10):
    dynamicAgent =DynamicAgent(10,1,0.1)
    Environment=DynamicEnv(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(1000):
        action=dynamicAgent.do_action()
        dynamicAgent.update_memory(Environment.return_reward(action))
    agent_SumRewards_StepSize_p_one.append(dynamicAgent.sum_rewards())
print('average reward: \t',
      sum(agent_SumRewards_StepSize_p_one)/len(agent_SumRewards_StepSize_p_one))
input("Press Enter to continue...")
## step size= 0.5 
print('initial reward estimation=1 and step size of 0.1, default parameter '+
      'The Environment')
agent_SumRewards_StepSize_p_five=[] 
for j in range(10):
    dynamicAgent =DynamicAgent(10,1,0.5)
    Environment=DynamicEnv(binnary_reward=True,
                 binnary_reward_vector=[0.1,0.5,0.9,0.01,0.99,0.02,0,0.74,0.3,0.17])
    for i in range(1000):
        action=dynamicAgent.do_action()
        dynamicAgent.update_memory(Environment.return_reward(action))
    agent_SumRewards_StepSize_p_one.append(dynamicAgent.sum_rewards())
print('average reward: \t',
      sum(agent_SumRewards_StepSize_p_one)/len(agent_SumRewards_StepSize_p_one))
input("Press Enter to continue...")