


import random
import torch



# REINFORCE: Monte-Carlo Policy-Gradient Control (episodeic) for optimal policy


# policy: a nn
# num_params: num of params of policy

def reinforce(mdp, policy, step_size, discount, num_params = 2**8, num_episodes=100, max_episode_length=100):
	# initialize parameters randomly
	policy_params = torch.tensor([random.random() for i in range(num_params)])

	for i in range(num_episodes):
		# episode is a list of form [state_0, action_0, reward_1, ... , state_t-1, action_t-1, reward_t]
		episode, episode_length = gen_episode(mdp, policy, policy_params, max_episode_length)

		rwd_vec, action_vec, state_vec = parse_episode(episode)
		

		for t in range(episode_length):
			G, action_t, state_t = sum_from_tp1(t, episode_length, rwd_vec, discount_vec)
			policy_params += step_size * (discount**t) * G * gradient_of( math.log( prob_under(policy, action_t, state_t, policy_params) ) )
			update_policy(policy, policy_params)


	return policy, policy_params





def sum_from_tp1(t, episode_length, rwd_vec):
	discount_vec = torch.tensor([discount**i for i in range(t+1, episode_length)])
	discounted_rwd_vec = discount_vec * rwd_vec
	return discount_vec * rwd_vec



# generates episode under policy; returns episode, episode_length
def gen_episode(mdp, policy, max_episode_length):
	# randomly choose initial state, action
	state = rand_choice(mdp.state_space(initial=True))
	action = policy.predict(state)

	# episode so far
	episode = []
	# episode length so far (number of time-steps, not directly length of episode list)
	episode_length = 0

	while (episode_length < max_episode_length) and (not mdp.terminal_state(state)):
		episode.append(state)
		episode.append(action)
		state, reward = mdp.successor(state, action)
		episode.append(reward)
		if not mdp.terminal_state(state):
			action = policy.predict(state)

		episode_length += 1

	return torch.tensor(episode), episode_length



	

# returns rwd_vec, action_vec, state_vec tensors
def parse_episode(episode):

	rwd_vec, action_vec, state_vec = [], [], []

	# pop off/update with each block in episode
	while episode:
		# pop off last rwd, action, state
		rwd_vec.append(episode.pop())
		action_vec.append(episode.pop())
		state_vec.append(episode.pop())

	return torch.tensor(rwd_vec), torch.tensor(action_vec), torch.tensor(state_vec)

# return gradient of equation (use autograd)
def gradient_of(equation, point):
	x = torch.tensor(point, requires_grad=True)
	y = equation(x)
	y.backward()
	return x.grad

# returns P(action | policy, policy_params, state)
def prob_under_policy(policy, action, state, policy_params)
	# TODO
	pass


# sets policy params to 'policy_params'
def update_policy(policy, policy_params):
	# TODO
	pass








