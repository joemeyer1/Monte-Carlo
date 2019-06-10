#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:55:30 2019

@author: joe
"""

from maze import Maze
from policy import Policy
from reinforce_mcpg import Reinforcer
import torch
import random



class Tester:
    def __init__(self):
        self.main()
        
    def main(self):
        self.maze = Maze()
        self.make_policy()
        self.make_reinforcer()
        self.reinforce()
        self.evaluate()
        
        
        
    def make_policy(self):
        in_dim = 2
        out_dim = 4
        H = 64
        non_io_hidden_layers = 5
        self.policy = Policy(in_dim, out_dim, H, non_io_hidden_layers)
    
    def make_reinforcer(self):
        step_size = .1
        discount = .9
        self.reinforcer = Reinforcer(self.maze, self.policy, step_size, discount)
        
    def reinforce(self):
        self.policy = self.reinforcer.reinforce()
        print("policy:",self.policy)
        
    def evaluate(self):
        init_state = random.choice(self.maze.state_space(initial=True))
        
        state = init_state
        print("initial state:", state)
        for i in range(200):
            if not self.maze.terminal_state(state):
                state_t = torch.tensor(state, dtype=torch.float)
                action_vec = self.policy(state_t)
                print("action vec:", action_vec)
                action_index = self.reinforcer.max_action(action_vec, state)
                print("action_index:", action_index)
                action = self.maze.get_action(state, action_index)
                state = self.maze.successor(state, action)[0]
                print("state:", state)
            else:
                print("terminal state found")
                break
        
    
t = Tester()
