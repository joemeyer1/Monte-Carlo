


# so far just pseudocode






import random

# traverse tree to select leaf
def select(tree, policy, end_condition):
	cur = tree
	while not end_condition:
		action = policy(cur)
		cur, end_condition = cur.move(action)


# explore new actions
def expand(tree, end_condition):
	policy = rand_unvisited(tree)
	select(tree, policy, end_condition)


# run greedy episode
def simulate(tree, policy end_condition):
	select(tree, policy, end_condition)


# update node with result
def backup(node, result, policy)
	update(node, result, policy)








def update(node, result, policy):
	policy.update(node, result)

# helper for expand

def rand_unvisited(tree):
	return random.choice(tree.action_space(unvisited=True))