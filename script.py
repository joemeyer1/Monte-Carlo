
import maze
import monte_carlo

mdp = maze.Maze()
policy_map, q_val_map = monte_carlo.monte_carlo(mdp)