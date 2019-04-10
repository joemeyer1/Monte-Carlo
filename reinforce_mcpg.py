


import random
import torch
from math import inf



# REINFORCE: Monte-Carlo policy-Gradient Control (episodeic) for optimal policy


# policy: a nn with initially randomized paramters

class Reinforcer:
	def __init__(self, mdp, policy, step_size, discount, num_episodes=100, max_episode_length=100):
		self.mdp = mdp
		self.policy = policy
		self.step_size, self.discount = step_size, discount
		self.num_episodes, self.max_episode_length = num_episodes, max_episode_length
		


	def reinforce(self):
		# get parameters
		policy_params = self.policy.params()

		for i in range(self.num_episodes):
			# episode is a list of form [state_0, action_0, reward_1, ... , state_t-1, action_t-1, reward_t]
			episode, episode_length = self.gen_episode()

			rwd_vec, action_vec, state_vec = self.parse_episode(episode.copy())
			discount_vec = torch.tensor([self.discount**i for i in range(episode_length)], dtype=torch.float)
			

			for t in range(episode_length):
				G = torch.sum(self.discounted_rwds(rwd_vec, discount_vec, t, episode_length))
				state_t = state_vec[t]
				action_t = action_vec[t]
				param_update = self.step_size * (self.discount**t) * G * self.log_policy_gradient(state_t, action_t, policy_params.copy())
				with torch.no_grad():
					for i in range(len(policy_params)):
						policy_params[i] += param_update[i]
					self.update_policy(policy_params)


		return self.policy




	# HELPERS for self.reinforce() :


	# generates episode under self.policy; returns episode, episode_length
	def gen_episode(self):
		# randomly choose initial state, action
		state = rand_choice(self.mdp.state_space(initial=True))

		# episode so far
		episode = []
		# episode length so far (number of time-steps, not directly length of episode list)
		episode_length = 0

		while (episode_length < self.max_episode_length) and (not self.mdp.terminal_state(state)):
			action_index = self.max_action(self.policy.model(torch.tensor(state, dtype=torch.float)))
			action = self.mdp.get_action(state, action_index)
			episode.append(state)
			episode.append(action_index)
			state, reward = self.mdp.successor(state, action)
			episode.append(reward)

			episode_length += 1

		return episode, episode_length



	# returns rwd_vec, action_vec, state_vec tensors
	def parse_episode(self, episode):

		rwd_vec, action_vec, state_vec = [], [], []

		# pop off/update with each block in episode
		while episode:
			# pop off last rwd, action, state
			rwd_vec.append(episode.pop())
			action_vec.append(episode.pop())
			state_vec.append(episode.pop())

		return torch.tensor(rwd_vec, dtype = torch.float), torch.tensor(action_vec, dtype = torch.float), torch.tensor(state_vec, dtype = torch.float)


	# returns vector of discounted rewards from time t+1
	def discounted_rwds(self, rwd_vec, discount_vec, t, episode_length):
		discounted_rwd_vec = discount_vec[:episode_length - (t+1)] * rwd_vec[t+1:]
		return discounted_rwd_vec


	# return gradient of P(action under policy) at state (use autograd)
	def log_policy_gradient(self, state, action, policy_params):
		# convert action to index
		action_index = self.normalize_action(action, state)
		w1, w2, w3 = policy_params
		w1.requires_grad_(True)
		w2.requires_grad_(True)
		w3.requires_grad_(True)
		self.update_policy([w1,w2,w3])
		self.policy.model.zero_grad()
		y = torch.log(self.policy.model(state.float())[action_index])
		with torch.no_grad():
			# zero gradients if they are 'none'
			w1grad = self.normalize_grad(w1.grad)
			w2grad = self.normalize_grad(w2.grad)
			w3grad = self.normalize_grad(w3.grad)

		y.backward()
		return torch.tensor([w1grad, w2grad, w3grad], dtype=torch.float)


	# sets self.policy params to 'policy_params'
	def update_policy(self, policy_params):
		self.policy.update(policy_params)


	# returns integer index corresponding w best action
	def max_action(self, action_vec):
		best_action = [None]
		best_value = -inf

		for action in range(len(action_vec)):
			val = action_vec[action]
			if val == best_value:
				best_action.append(action)
			elif val > best_value:
				best_action = [action]
				best_value = val

		return random.choice(best_action)


	# if the action is invalid this makes it '-1' instead of 'None'
	# allows for its inclusion in tensor
	def normalize_action(self,action, state):
		valid_action = action in range(0, len(self.mdp.action_space(state)))
		if not valid_action:
			return 0
		return action

	def normalize_grad(self, grad):
		if grad:
			return grad
		return torch.tensor([0.])

	# # returns integer index corresponding w action
	# def index_of(self, action, state):
	# 	return self.mdp.action_space(state).index(action)

def rand_choice(ls):
	if not ls:
		return None
	return random.choice(ls)



