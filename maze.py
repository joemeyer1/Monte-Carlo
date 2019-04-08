
import numpy as np



# init state same for all
INITIAL_STATE = [(0, 0)]


# SMALL_MAZE dimensions: x = 3, y = 3

SMALL_MAZE = [[0,0,0],
			  [0,0,0],
			  [0,0,999]]

SMALL_TERMINAL = [(2,2)]



ONE_PATH_MAZE = \
		[[0,0, -4, 0, 0, 0, 0],
		[-1,1, 1, 2, 0, 0, 0],
		[0, -2, 0, 4, 8, 16,0],
		[0,  0,-4, 0, 0, 32,0],
		[0,  0, 0,-8, 0, 64,0],
		[0,  0, 0, 0,-16, 0,0]]		

def one_path_terminal():
	# For one-path, most states are terminal
	# so compile list of all states
	one_path_term = [(x, y) for x in range(len(ONE_PATH_MAZE[0])) for y in range(len(ONE_PATH_MAZE))]
	# then remove the non-terminal ones
	path_non_term = [(0,0), (1,0), (1, 1), (2, 1), (3, 1), (2,2), (3, 2), (4, 2), (5, 2), (5, 3)]
	for nont in path_non_term:
		one_path_term.remove(nont)
	return one_path_term

ONE_PATH_TERMINAL = one_path_terminal()



# MAZE/HARD_MAZE dimensions: x = 9, y = 9

MAZE = [[0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,2**9,0],
		[0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0]]



HARD_MAZE = [[0, -10, -10, -1, 0, 10, 50, 100, 500],
			[-10, -10, -10,   0, 0, 10, 50, 100, 500],
			[-10, -10, -10,   0, 0, 10, 50, 100, 500],
			[-10, -10, -10,   0, 0, 10, 50, 100, 500],
			[0,     0,  50,   0, 0,  0,  0,   0,   0],
			[0,     0,   0,  50, 0,  0, -10, -10, -10],
			[0,    50,  -10,  0, 0, 50, -10, 2**30, 10],
			[0,     0,  -10,  0, 0,  0,  0,   0,   0],
			[0,     0,   0,  10, 0,  0,  0,   0,   0]]

# maze terminal state same for MAZE, HARD_MAZE
MAZE_TERMINAL = [(7, 6)]



# data just compiles the relevant info
SMALL_MAZE_DATA = 	(SMALL_MAZE, 	INITIAL_STATE,	SMALL_TERMINAL)
ONE_PATH_DATA = 	(ONE_PATH_MAZE, INITIAL_STATE,	ONE_PATH_TERMINAL)
MAZE_DATA = 		(MAZE, 			INITIAL_STATE,	MAZE_TERMINAL)
HARD_MAZE_DATA = 	(HARD_MAZE, 	INITIAL_STATE,	MAZE_TERMINAL)




class Maze:
	def __init__(self, data = ONE_PATH_DATA):
		self.maze, self.init_states, self.end_states = data


	def state_space(self, initial=False, non_terminal=False):
		if initial:
			return [s for s in self.init_states]
		# non-initial: get state space
		state_space = [(x, y) for x in range(len(self.maze[0])) for y in range(len(self.maze))]
		# if non-terminal remove terminal states
		if non_terminal:
			for end in self.end_states:
				state_space.remove(end)

		return state_space.flatten()


	def action_space(self, state):
		actions = []
		offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		for offset in offsets:
			new_x, new_y = np.array(state)+np.array(offset)
			if 0 <= new_x < len(self.maze[0]) and 0 <= new_y < len(self.maze):
				actions.append(offset)
		return actions

	def get_action(self, state, action_index):
		action_space = self.action_space(state)
		if action_index < len(action_space):
			return action_space[action_index]
		else:
			return None



	def successor(self, state, action):
		if action not in self.action_space(state):
			action = (0, 0)
		new_x, new_y = np.array(state)+np.array(action)
		next_state = (new_x, new_y)
		reward = self.maze[new_y][new_x]
		return next_state, reward



	def terminal_state(self, state):
		return state in self.end_states


























