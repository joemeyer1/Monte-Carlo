


# so far just general pseudocode

import random

# end_condition_mcts: when to stop improving policy (i.e. performance is good enough)
# end_condition_select: when a leaf is reached (all children unvisited)
# end_condition_sim: when to end episode (i.e. max episode length or terminal state reached)

def main(tree, end_condition_mcts, end_condition_select, end_condition_sim):
	policy = initialize_policy(tree)
	while not end_condition_mcts:
		result = select(tree, policy, end_condition_select)
		result = expand(result, end_condition_select)
		result = simulate(result, policy, end_condition_sim)
		tree = backup(tree, result, policy)
		end_condition_mcts = check_finish(tree)


# traverse tree to select leaf
def select(tree, policy, end_condition):
	cur = tree
	while not end_condition:
		action = policy(cur)
		cur, end_condition = cur.move(action)
	return cur


# explore new actions
def expand(tree, end_condition):
	policy = rand_unvisited(tree)
	return select(tree, policy, end_condition)


# run greedy episode
def simulate(tree, policy, end_condition):
	return select(tree, policy, end_condition)


# update node with result
def backup(node, result, policy)
	policy.update(node, result)



	
	

# helper for expand
def rand_unvisited(tree):
	return random.choice(tree.action_space(unvisited=True))

# helpers for main
# TODO
def check_finish(tree):
	pass


def initialize_policy(tree):
	# this depends on whether it's tabular or gradient
	pass



class Policy:
	def __init__(self):
		pass

	def __call__(self, node):
		pass

	def update(self, node, result):
		pass







