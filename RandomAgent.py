# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 10:54:21 2018
@author: Hesham El Abd
@description: A random bandit agent,i.e. an agent that choose its actions 
randomly
"""
from Agent_Framework import bandit_agent, environment
import numpy as np 
##########################
class random_agent(bandit_agent):
    def __init__(self,num_actions):
        """
        # inputs:
                num_action: is the number of actions that agent has to choose
                from, a positive integer bigger than or equal to two.
        """
        if num_actions<2:
            raise ValueError('The number of actions the agent has to choose from'+
                             'should be at least two')
        self.num_actions=num_actions
        self.memory_actions=[] 
        self.memory_rewards=[] 
        return
    
    def do_action(self):
        actions=np.array(range(self.num_actions))
        action=int(np.random.choice(actions,size=1))
        self.memory_actions.append(action)
        return action
    
    def estimate_reward(self):
        """
        The agent is compeletly random so, it does not need to estimate the 
        rewards for its own actions.
        """
        return 
    
    def memory(self,reward): 
        """
        A memory for the agent, this simple memory only contian the reward 
        history.
        """
        self.memory_rewards.append(reward)
        return
    
    def sum_rewards(self):
        return(sum(self.memory_rewards))
        
    def extract_memories(self):
        """
        A function to extract the whole histroy of the agent, it returns a 2D
        array of size (time_steps,2) where the first column is the actions and
        the second column is the reward
        """
        action_history=np.array(self.memory_actions).reshape(
                len(self.memory_actions),1)
        
        reward_history=np.array(self.memory_rewards).reshape(
                len(self.memory_rewards),1)
        return np.concatenate((action_history,reward_history), axis=1)
        
class env_RandAgent(environment):
    
    def __init__(self,binnary_reward=True,continous_reward=False,
                 binnary_reward_vector=None,continous_reward_matrix=None):
        """
        The env_RandAgent is a simple environment where at each time step 
        the environment recieves an action from the agent and returns a reward.
        For each action, the environment has a pre-defined probability distrbution 
        for the reward, theses distrbutions are constant across all time steps.
        
        # inputs: 
                -binnary_reward: The reward is a bernoulli random variable 
                i.e. R(a_i)~bern(pi) where pi is the probability of getting a 
                reward by playing the ith action. default: True. 
                
                -continous_reward: The reward is a normally distrbuted 
                random variable. i.e. R(a_i)~N(mean_i,sd_). default is false
                
                -binnary_reward_vector: a numpy vector of the same length 
                as the number of actions. each element in the array is the 
                probability of success associated with playing each action.
                numpy array each element, pi, is between zero and one.
                
                -continous_reward_matrix: a 2D numpy array with size equal to
                (num_actions,2), where each row is the mean and sd for the ith 
                action reward distrbution.
        """
        if binnary_reward and binnary_reward_vector is None:
            raise ValueError('Please provide a reward vector')
            
        if binnary_reward and binnary_reward_vector is not None:
            if len(binnary_reward_vector) < 2:
                raise ValueError('The binary vector should be at least of ' +
                                 'length two.')
            else: 
                for reward_pro in binnary_reward_vector:
                    if not reward_pro >=0 and reward_pro <=0:
                        raise ValueError('Probability vector should be a value'+
                                         'between zero and one.')
       
        if binnary_reward and continous_reward:
            raise ValueError('The reward can follow only one mode.')
        
        if continous_reward and continous_reward_matrix is None:
            raise ValueError('Please provide a reward Matrix')
        
        self.binnary_reward=binnary_reward
        self.binnary_reward_vector=binnary_reward_vector
        self.continous_reward=continous_reward
        self.continous_reward_matrix=continous_reward_matrix
        return
    
    def return_reward(self,action):
        """
        Sample a reward for an action from it's associated distrbution.
        # input: 
                -action: the index for each action, goes from zero to num_action
        # output: 
                Reward, a scaler value from the action associated distrbution
        """
        if self.binnary_reward:
            action_prob=self.binnary_reward_vector[action]
            return np.random.binomial(1,action_prob)
        else: 
            action_prob=self.continous_reward_matrix[action,]
            return np.np.random.normal(action_prob[0],action_prob[1])
###############################################################################
