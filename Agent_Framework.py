# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 09:25:58 2018

@author: Hesham El Abd
@description: 
        The file contians a parent class for all Bandit agents and environment 
        classes
"""
class bandit_agent(object):
    """
    any bandit agent must have at least two methods, the first is do_action
    which enables the agent to take an action from the set of possible actions.
    The second function is estimate_reward which enables the agent to estimate
    the reward for each action.
    """
    def do_action(self):
        raise NotImplementedError('Inheriting classes must override do_action.')
        
    def estimate_reward(self):
        raise NotImplementedError('Inheriting classes must override estimate_reward.')
    
class environment(object):
    """
    Here, return_reward respresent the scaler value that the environment
    return when the agent take an action.
    """
    def return_reward(self):
        raise NotImplementedError('Inheriting classes must override return_reward')
