#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:38:46 2019

@author: joe
"""

from tetris_interface import TetrFace
from policy import Policy
from reinforce_mcpg import Reinforcer


# make mdp
mdp = TetrFace()

# make policy
D_in = len(mdp.get_state())
D_out = 4
H = 100
M = 15
pol = Policy(D_in, D_out, H, M)

# make reinforcer
r=Reinforcer(mdp,pol,.1,.9)
r.reinforce()


# test
t=TetrFace()
for i in range(500):
    state_t = t.get_state()
    action_vec = pol(state_t)
#    print("action vec:", action_vec)
    action_index = r.max_action(action_vec, state_t[0])
#    print("action_index:", action_index)
    action = t.get_action(state_t[0], action_index)
    t.take_action(action)
#    print("state:", state)







