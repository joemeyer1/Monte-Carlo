

import random
from math import inf


def monte_carlo(mdp, discount = .9, max_episode_length = 1000, max_episodes = 100000):
	# policy_map = {state : action from state under policy}
	policy_map = {state : random.choice(mpd.action_space(state)) for state in mdp.state_space()}

	# q_val_map = {(state, action):(state-action average return, number of times state-action visited in episode)
	q_val_map = {(state, action):(0,0) for action in mdp.action_space(state) for state in mpd.state_space()}
	
	for i in range(max_episodes):
		# episode is a list of form [state_0, action_0, reward_1, ... , state_t-1, action_t-1, reward_t]
		episode, episode_length = gen_episode(mpd, init_state, init_action, max_episode_length)

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
				policy_map[state] = max_action(q_val_map, state, mdp)

	return policy_map, q_val_map




def gen_episode(mpd, max_episode_length):

	# randomly choose initial state, action
	state = random.choice(mdp.state_space(initial=True))
	action = random.choice(mpd.action_space(state))

	# episode so far
	episode = []
	# episode length so far (number of time-steps, not directly length of episode list)
	episode_length = 0

	while (episode_length < len(max_episode_length)) and (not mpd.terminal_state(state)):
		episode.append(state)
		episode.append(action)
		state, reward = mdp.successor(state, action)
		episode.append(reward)
		action = policy_map[state]
		episode_length += 1

	return episode, episode_length



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
		if q_val > best_value:
			best_action = action
			best_value = q_val
			
	return best_action












