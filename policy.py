
import torch
from torch import tensor



class Policy:


	def __init__(self, D_in, D_out, H, datatype=torch.float):
		# w1: in(D_in) - sigmoid -> H
		self.w1 = torch.randn(D_in, H, dtype=datatype)
		# w2: H - relu-sigmoid- -> H
		self.w2 = torch.randn(H, H, dtype=datatype)
		# w3: H - relu-sigmoid- -> D_out
		self.w3 = torch.randn(H, D_out, dtype=datatype)
		# during computation we will then run results through softmax


	# state is tensor of same dtype as Policy weights
	def __call__(self, state, params = None):
		if not params:
			params = self.params()
		w1, w2, w3 = params
		# first layer sigmoid
		result = torch.mv(torch.t(w1), state)
		result = torch.sigmoid(result)
		# second layer relu
		result = torch.mv(torch.t(w2), result)
		result = result.clamp(min=0)
		result = torch.sigmoid(result)
		# third layer relu
		result = torch.mv(torch.t(w3), result)
		result = result.clamp(min=0)
		result = torch.sigmoid(result)
		# finally, softmax it
		return torch.nn.functional.softmax(result, dim=result.dim()-1)


	# params is a tensor of initial policy parameters
	def update(self, params):
		self.w1, self.w2, self.w3 = params

	# returns policy parameters
	def params(self):
		return [self.w1, self.w2, self.w3]

	# return gradient of P(action under policy) at state (use autograd)
	def log_policy_gradient(self, state, action, policy_params, mdp):
		# convert action to index
		action_index = self.normalize_action(action, state, mdp)
		w1, w2, w3 = policy_params
		w1.requires_grad_(True)
		w2.requires_grad_(True)
		w3.requires_grad_(True)
		y = torch.log(self(state.float(), [w1, w2, w3])[action_index])
		y.backward()
		return [w1.grad, w2.grad, w3.grad]

	# if the action is invalid this makes it '-1' instead of 'None'
	# allows for its inclusion in tensor
	def normalize_action(self,action, state, mdp):
		valid_action = action in range(0, len(mdp.action_space(state)))
		if not valid_action:
			action = torch.tensor([-1])
		return action.int().item()














