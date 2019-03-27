

import random
from math import inf


def monte_carlo(mdp, discount = .9999, epsilon = .4, max_episode_length = 20, num_episodes = 10000):
	# policy_map = {state : action from state under policy}
	policy_map = {state : rand_choice(mdp.action_space(state)) for state in mdp.state_space(non_terminal=True)}

	# q_val_map = {(state, action):(state-action average return, number of times state-action visited)
	q_val_map = {}
	for state in mdp.state_space():
		for action in mdp.action_space(state):
			q_val_map[(state, action)] = (0,0)
	
	for i in range(num_episodes):
		# episode is a list of form [state_0, action_0, reward_1, ... , state_t-1, action_t-1, reward_t]
		episode, episode_length = gen_episode(mdp, policy_map, max_episode_length, epsilon)

		G = 0

		# pop off/update with each block in episode
		while episode:
			# pop off last rwd, action, state
			reward = episode.pop()
			action = episode.pop()
			state = episode.pop()
			G = (discount * G) + reward

			if not pre_visited(episode, state, action):
				value_avg, times_visited = q_val_map[(state, action)]
				# online mean calculation
				q_val = value_avg + ((1. / episode_length) * (reward - value_avg))
				#update q map
				q_val_map[(state, action)] = (q_val, times_visited + 1)
				# update policy_map
				policy_map[state] = max_action(q_val_map, state, mdp)

	return policy_map, q_val_map




def gen_episode(mdp, policy_map, max_episode_length, epsilon):


	# randomly choose initial state, action
	state = rand_choice(mdp.state_space(initial=True))
	if epsilon > 0:
		action = rand_choice(mdp.action_space(state))
	else:
	# ow we want a greedy run
		action = compute_action(mdp, policy_map, state, epsilon)

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
			action = compute_action(mdp, policy_map, state, epsilon)

		episode_length += 1

	return episode, episode_length


# helper for gen_episode
def compute_action(mdp, policy_map, state, epsilon):
	if flip_coin(epsilon):
		return rand_choice(mdp.action_space(state))
	else:
		return policy_map[state]




def pre_visited(episode, state, action):
	i = 0
	while i < len(episode) - 1:
		if episode[i] == state:
			if episode[i+1] == action:
				return True
		i += 3
	return False



def max_action(q_val_map, state, mdp):

	best_action = None
	best_value = -inf

	for action in mdp.action_space(state):
		q_val, _ = q_val_map[(state, action)]
		if q_val == best_value:
			best_action.append(action)
		elif q_val > best_value:
			best_action = [action]
			best_value = q_val

	return random.choice(best_action)




def rand_choice(ls):
	if not ls:
		return None
	return random.choice(ls)


# flips a weighted bool coin (P(1) =: p)
def flip_coin(p):
    r = random.random()
    return r < p




