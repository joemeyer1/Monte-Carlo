
import maze
import monte_carlo

# get mdp
mdp = maze.Maze()

# run monte-carlo
def run():
	return monte_carlo.monte_carlo(mdp)

policy_map, q_val_map = run()
simple_q_val_map = {key:q_val_map[key][0] for key in q_val_map}


def print_episode():
	return monte_carlo.gen_episode(mdp, policy_map, max_episode_length=20, epsilon=0)

def print_episode_states(episode=print_episode()[0]):
	states = []
	i = 0
	while i < len(episode):
		states.append(episode[i])
		i += 3
	rwd = episode[-1]
	return states, rwd

def avg_reward(episodes = 10, runs = 1):
	for r in range(runs):
		run()
		tot_rwd = 0
		for i in range(episodes):
			tot_rwd += print_episode_states()[1]
		return tot_rwd/float(episodes)



# shortcuts:

# p for convenience
p = print_episode
# ps for convenience
ps = print_episode_states
# pol, q for convenience
pol, q = policy_map, q_val_map
# a for convenience
a = avg_reward