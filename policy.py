
import torch
from torch import tensor



class Policy:


	def __init__(self, D_in, H, dtype=torch.float):
		# w1: in(D_in) - linear -> H
		self.w1 = torch.randn(D_in, H, dtype=dtype)
		# w2: H - relu -> H
		self.w2 = torch.randn(H, H, dtype=dtype)
		# w3: H - linear -> D_out
		self.w3 = torch.randn(H, D_out, dtype=dtype)
		# during computation we will then run results through softmax


	# input is tensor of same dtype as Policy weights
	def __call__(self, input):
		# first layer linear
		result = input.mm(self.w1)
		# second layer relu
		result = result.mm(self.w2)
		result = result.clamp(min=0)
		# third layer relu
		result = result.mm(self.w3)
		result = result.clamp(min=0)
		# finally, softmax it
		return softmax(result)


	# params is a tensor of initial policy parameters
	def update(self, params):
		self.w1, self.w2, self.w3 = params

	# returns policy parameters
	def params(self):
		return torch.tensor([self.w1, self.w2, self.w3])







def softmax(ls):
	tot = torch.sum([torch.exp(x) for x in ls])
	return torch.tensor([torch.exp(x)/tot for x in ls])







