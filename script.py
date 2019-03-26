
import maze
import monte_carlo

mdp = maze.Maze()
policy_map, q_val_map = monte_carlo.monte_carlo(mdp)
# p, q for convenience
p, q = policy_map, q_val_map

def print_episode():
	return monte_carlo.gen_episode(mdp, policy_map, max_episode_length=1000, epsilon=0)