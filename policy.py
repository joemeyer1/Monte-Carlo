
import torch
from torch import tensor



class Policy:


	def __init__(self, D_in, D_out, H, datatype=torch.float):
		# w1: in(D_in) - linear -> H
		self.w1 = torch.randn(D_in, H, dtype=datatype)
		# w2: H - relu -> H
		self.w2 = torch.randn(H, H, dtype=datatype)
		# w3: H - relu -> D_out
		self.w3 = torch.randn(H, D_out, dtype=datatype)
		# during computation we will then run results through softmax


	# state is tensor of same dtype as Policy weights
	def __call__(self, state, params = None, sh=True):
		if not params:
			params = self.params()
		w1, w2, w3 = params
		# first layer linear
		result = torch.mv(torch.t(w1), state)
		if not sh: print("res1:", result)
		# second layer relu
		result = torch.mv(torch.t(w2), result)
		if not sh: print("res2:", result)
		result = result.clamp(min=0)
		if not sh: print("res3:", result)
		# third layer relu
		result = torch.mv(torch.t(w3), result)
		if not sh: print("res4:", result)
		result = result.clamp(min=0)
		if not sh: print("res5:", result)
		if not sh: print("res6:", softmax(result))
		# finally, softmax it
		return softmax(result)


	# params is a tensor of initial policy parameters
	def update(self, params):
		self.w1, self.w2, self.w3 = params

	# returns policy parameters
	def params(self):
		return [self.w1, self.w2, self.w3]







def softmax(ls):
	tot = torch.sum(torch.exp(ls))
	ftot = torch.tensor([tot,tot,tot,tot], dtype=ls.dtype)
	return torch.exp(ls)/ftot
	# return torch.tensor([torch.exp(x)/tot for x in ls])







