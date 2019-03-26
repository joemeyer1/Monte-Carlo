
import numpy as np


MAZE = [[0, -10, -10, -10, 0, 10, 50, 100, 500],
		[-10, -10, -10,   0, 0, 10, 50, 100, 500],
		[-10, -10, -10,   0, 0, 10, 50, 100, 500],
		[-10, -10, -10,   0, 0, 10, 50, 100, 500],
		[0,     0,  50,   0, 0,  0,  0,   0,   0],
		[0,     0,   0,  50, 0,  0, -10, -10, -10],
		[0,    50,  -10,  0, 0, 50, -10, 9999, 10],
		[0,     0,  -10,  0, 0,  0,  0,   0,   0],
		[0,     0,   0,  10, 0,  0,  0,   0,   0]]

# MAZE dimensions: x = 9, y = 9
# start: (0, 0)
# end:   (7, 6)


class Maze:


	def state_space(self, initial=False):
		if initial:
			return [(0, 0)]
		else:
			return [(x, y) for x in range(len(MAZE[0])) for y in range(len(MAZE))]


	def action_space(self, state):
		actions = []
		offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		for offset in offsets:
			new_x, new_y = np.array(state)+np.array(offset)
			if 0 <= new_x < len(MAZE[0]) and 0 <= new_y < len(MAZE):
				actions.append(offset)
		return actions



	def successor(self, state, action):
		new_x, new_y = np.array(state)+np.array(action)
		next_state = new_x, new_y
		reward = MAZE[new_y][new_x]
		return next_state, reward



	def terminal_state(self, state):
		return state == (7, 6)


