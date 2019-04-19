
import torch



class Policy:


	def __init__(self, D_in, D_out, H, non_io_hidden_layers = 2, datatype=torch.float):
		# w_in: in(D_in) - relu-sigmoid -> H
		w_in = torch.randn(D_in, H, dtype=datatype)
		self.layer_weights = [w_in]
		# w_hidden: H - relu-sigmoid- -> H
		for layer in range(non_io_hidden_layers):
			hidden_layer = torch.randn(H, H, dtype=datatype)
			self.layer_weights.append(hidden_layer)
		# w_out: H - relu-sigmoid- -> D_out
		w_out = torch.randn(H, D_out, dtype=datatype)
		self.layer_weights.append(w_out)
		# during computation we will then run results through softmax


	# state is tensor of same dtype as Policy weights
	def __call__(self, state, params = None):
		if not params:
			params = self.layer_weights
		# relu-sigmoid
		result = state
		i = 0
		while i < len(params):
			w = params[i]
			result = torch.mv(torch.t(w), result)
			result = result.clamp(min=0)
			result = torch.sigmoid(result)
			i += 1
		# finally, softmax it
		return torch.nn.functional.softmax(result, dim=result.dim()-1)


	# params is a tensor of initial policy parameters
	def update(self, params):
		self.layer_weights = params



	# return gradient of P(action under policy) at state (use autograd)
	def log_policy_gradient(self, state, action, mdp):
		# convert action to index
		action_index = self.normalize_action(action, state, mdp)
		params = self.layer_weights.copy()
		for param in params:
			param.requires_grad_(True)
		y = torch.log(self(state.float(), params)[action_index])
		y.backward()
		return [w.grad for w in self.layer_weights]

	# if the action is invalid this makes it '-1' instead of 'None'
	# allows for its inclusion in tensor
	def normalize_action(self,action, state, mdp):
		valid_action = action in range(0, len(mdp.action_space(state)))
		if not valid_action:
			action = torch.tensor([-1])
		return action.int().item()














