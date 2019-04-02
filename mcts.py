


# so far just pseudocode

import random


def main(tree, end_condition):
	policy = initialize_policy()
	while not end_condition:
		tree = select(tree, policy, end_condition)
		result = expand(tree, end_condition)
		result = simulate(tree, result, policy)
		backup(tree, result, policy)
		end_condition = check_finish(tree)


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
	update(node, result, policy)







# helper for backup
def update(node, result, policy):
	policy.update(node, result)

# helper for expand
def rand_unvisited(tree):
	return random.choice(tree.action_space(unvisited=True))

# helper for main
# TODO
def check_finish(tree):
	pass