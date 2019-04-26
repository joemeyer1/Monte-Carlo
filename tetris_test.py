#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:38:46 2019

@author: joe
"""

from tetris_interface import TetrFace
from policy import Policy
from reinforce_mcpg import Reinforcer

def main(visualize=False):
	# make mdp
	mdp = TetrFace()

	# make policy
	D_in = len(mdp.get_state())
	D_out = 4
	H = 1000
	M = 50
	pol = Policy(D_in, D_out, H, M)
	step_size = .1
	discount = .9
	num_episodes=50000
	max_episode_length=10000

	# make reinforcer
	r=Reinforcer(mdp, pol, step_size, discount, num_episodes, max_episode_length)
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
	    if visualize:
	    	t.print_board()
	    else:
	    	print("Score: ", t.score)
	#    print("state:", state)







