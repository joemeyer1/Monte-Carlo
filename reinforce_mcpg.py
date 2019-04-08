


import random
import torch



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

		for i in range(num_episodes):
			# episode is a list of form [state_0, action_0, reward_1, ... , state_t-1, action_t-1, reward_t]
			episode, episode_length = self.gen_episode(self.max_episode_length)

			rwd_vec, action_vec, state_vec = self.parse_episode(episode)
			

			for t in range(episode_length):
				G, action_t, state_t = self.sum_from_tp1(t, episode_length, rwd_vec)
				policy_params += self.step_size * (self.discount**t) * G * self.gradient_of( math.log( self.prob_under_policy(action_t, state_t) ) )
				self.update_policy(policy_params)


		return self.policy, policy_params





	def sum_from_tp1(self, t, episode_length, rwd_vec):
		discount_vec = torch.tensor([discount**i for i in range(t+1, episode_length)])
		discounted_rwd_vec = discount_vec * rwd_vec[t+1:]
		return discounted_rwd_vec



	# generates episode under self.policy; returns episode, episode_length
	def gen_episode(self):
		# randomly choose initial state, action
		state = rand_choice(self.mdp.state_space(initial=True))
		action = self.max_action(self.policy.predict(state))

		# episode so far
		episode = []
		# episode length so far (number of time-steps, not directly length of episode list)
		episode_length = 0

		while (episode_length < self.max_episode_length) and (not self.mdp.terminal_state(state)):
			episode.append(state)
			episode.append(action)
			state, reward = self.mdp.successor(state, action)
			episode.append(reward)
			if not self.mdp.terminal_state(state):
				action_index = self.max_action(self.policy.predict(state))
				action = self.mdp.get_action(state, action_index)

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

		return torch.tensor(rwd_vec), torch.tensor(action_vec), torch.tensor(state_vec)

	# return gradient of equation (use autograd)
	def gradient_of(self, equation, point):
		x = torch.tensor(point, requires_grad=True)
		y = equation(x)
		y.backward()
		return x.grad

	# returns P(action | self.policy, policy_params, state)
	def prob_under_policy(self, action, state)
		action_vec = self.policy.predict(state)
		action_index = self.index_of(action)
		return action_vec[action_index]


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


	# returns integer index corresponding w action
	def index_of(self, action, state):
		return self.mdp.action_space(state).index(action)



