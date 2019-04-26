#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:37:51 2019

@author: joe
"""

from tetris import Tetris
import torch
import numpy as np
import copy

# tetris interface
class TetrFace(Tetris):
    def __init__(self, width=4):
        super().__init__(width)
        
    def state_space(self, initial=False):
        state_space = []
        state = self.get_state()
        for action in self.action_space():
            state_space.append(self.successor(state, action))
        return state_space
            
        
    def get_state(self):
        ground = torch.tensor(self.ground)
        # location of active shape
        active_loc = self.shape_loc
        # rotation of active shape
        active_rot = self.shape_position
        active_sqs = torch.tensor(list(self.active_squares()), dtype=torch.float).numpy()
        return torch.tensor(np.array(active_loc)+np.array(active_rot)+np.array(active_sqs), dtype=torch.float).flatten()
    
    def action_space(self, state=None):
        # you can always move, it just might not do anything
        return ['a','w','s','d']
    
    def get_action(self, state, action_index):
         action_space = self.action_space()
         if action_index in range(0, len(action_space)):
            return action_space[action_index]
         else:
            return None
        
    # state is unused here but this is the required format for monte-carlo
    def successor(self, state, action):
        if action not in self.action_space():
            action = self.step
        next_tetris = copy.deepcopy(self)
        next_tetris.take_action(action)
        next_state = next_tetris.get_state()
        reward = next_tetris.score - self.score
        return next_state, reward
    
    def terminal_state(self, state=None):
        return self.score==0
    
    
        