
import numpy as np


# SMALL_MAZE dimensions: x = 3, y = 3

SMALL_MAZE = [[0,0,0],
			  [0,0,0],
			  [0,0,999]]


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


# init state same for all
INITIAL_STATE = (0, 0)
SMALL_TERMINAL = (2,2)
# maze terminal state same for MAZE, HARD_MAZE
MAZE_TERMINAL = (7, 6)


# start: (0, 0)
# end:   (7, 6)

MAZE_TERMINAL = (7, 6)


maze = SMALL_MAZE
terminal_state = SMALL_TERMINAL

class Maze:
	def __init__(self, maze = maze, init_states = [INITIAL_STATE], end_state = terminal_state):
		self.maze = SMALL_MAZE
		self.init_states = init_states
		self.end_states = [end_state]


	def state_space(self, initial=False, non_terminal=False):
		if initial:
			return [s for s in self.init_states]
		# non-initial: get state space
		state_space = [(x, y) for x in range(len(self.maze[0])) for y in range(len(self.maze))]
		# if non-terminal remove terminal states
		if non_terminal:
			for end in self.end_states:
				state_space.remove(end)

		return state_space


	def action_space(self, state):
		actions = []
		offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		for offset in offsets:
			new_x, new_y = np.array(state)+np.array(offset)
			if 0 <= new_x < len(self.maze[0]) and 0 <= new_y < len(self.maze):
				actions.append(offset)
		return actions



	def successor(self, state, action):
		new_x, new_y = np.array(state)+np.array(action)
		next_state = (new_x, new_y)
		reward = self.maze[new_y][new_x]
		return next_state, reward



	def terminal_state(self, state):
		return state in self.end_states


