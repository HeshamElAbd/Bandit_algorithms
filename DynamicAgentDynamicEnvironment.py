# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:25:06 2018
@author: Hesham El Abd
@description: an efficent implementation for a greedy agent 
"""
from Agent_Framework import bandit_agent, environment
import numpy as np 
###############################################################################
class DynamicAgent(bandit_agent):
    def __init__(self,num_actions,init_reward_estimation,step_size):
        """
        # inputs:
                num_action: is the number of actions that agent has to choose
                from, a positive integer bigger than or equal to two.
                
                step_size: weights the difference betwenn the observed reward 
                and the estimation of the reward. step_size is bigger that zero
                and smaller or equal to one.
        """
        if num_actions<2:
            raise ValueError('The number of actions the agent has to choose from'+
                             'should be at least two')
        if step_size ==0 or step_size > 1:
            raise ValueError('The step size should be bigger than zero and '+
                             ' smaller smaller than or equal to 1' + 
                             ' found step_size '+ str(step_size)+ ' !!!')
        self.num_actions=num_actions
        self.actions=np.array(range(num_actions))
        self.last_action=None
        self.age=0
        self.actions_rewards=np.ones(
                shape=(num_actions,1))*init_reward_estimation
        self.total_rewards=0
        self.step_size=step_size
        return
    
    def do_action(self):
        action=np.argmax(self.actions_rewards)
        self.last_action=action
        self.age+=1
        return action
    
    def update_memory(self,reward): 
        """
        A memory for the agent, this simple memory only contian the reward 
        history.
        """
        self.actions_rewards[self.last_action]=(
                self.actions_rewards[self.last_action]+(self.step_size*(
                        reward-self.actions_rewards[self.last_action]
                )))
        self.total_rewards+=reward
        return
    
    def sum_rewards(self):
        return self.total_rewards
        
class DynamicEnv(environment):
    
    def __init__(self,binnary_reward=True,continous_reward=False,
                 binnary_reward_vector=None,continous_reward_matrix=None,
                 noise_mean=0,noise_sd=0.1):
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
        self.noise_mean=noise_mean
        self.noise_sd=noise_sd
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
            noise=np.random.normal(self.noise_mean,self.noise_sd,1)
            action_prob=np.clip(self.binnary_reward_vector[action]+noise,0,1)
            return np.random.binomial(1,action_prob)
        else: 
            action_prob=self.continous_reward_matrix[action,]
            return np.np.random.normal(action_prob[0],action_prob[1])
