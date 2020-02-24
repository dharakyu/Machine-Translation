from __future__ import print_function
import torch
from net import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def tensor_operations():
	# initialize tensor of zeros
	x = torch.zeros(5, 3, dtype=torch.long)
	print(x)

	# get size
	print(x.size())

	# addition
	y = torch.rand(5, 3)
	print(x + y)

	# add in place
	y.add_(x)
	print(y)

	# index using numpy syntax
	print(x[:, 1])

	# resizing
	x = torch.randn(4, 4)
	y = x.view(16)
	z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
	print(x.size(), y.size(), z.size())

	# getting element from one-element tensor
	x = torch.randn(1)
	print(x)
	print(x.item())

def autograd_operations():
	# create tensor and track operations
	x = torch.ones(2, 2, requires_grad=True)
	print(x)

	# tensor operation
	y = x + 2
	print(y) # y will will have a grad_fn that references the function that created it

	# more operations
	z = y * y * 3
	out = z.mean()
	print(z, out)

	# default is for requires_grad = False
	a = torch.randn(2, 2)
	a = ((a * 3) / (a - 1))
	print(a.requires_grad) # False
	a.requires_grad_(True)
	print(a.requires_grad) # True
	b = (a * a).sum()
	print(b.grad_fn)

	# backprop on out, has to be size 1
	out.backward()

	# d(out)/dx
	print(x.grad)

	# backprop for non-scalar output
	x = torch.randn(3, requires_grad=True)
	y = x * 2
	while y.data.norm() < 1000:
   		y = y * 2
	print(y)

	v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
	y.backward(v) # since y is not a scalar, calculate the vector-Jacobian product
	print(x.grad)

def neural_network():
	my_net = Net()
	print(my_net)

	params = list(my_net.parameters())
	print(len(params))
	print(params[0].size())  # conv1's .weight

	input = torch.randn(1, 1, 32, 32) # what do these parameters represent??
	out = my_net(input)
	print(out)

	# zero out the gradient (to avoid accumulating??)
	my_net.zero_grad()
	out.backward(torch.randn(1, 10))

	# define loss function
	output = my_net(input)
	target = torch.randn(10)  # a dummy target, for example
	target = target.view(1, -1)  # make it the same shape as output
	criterion = nn.MSELoss()

	loss = criterion(output, target)
	print(loss)

	# doing backprop
	my_net.zero_grad()     # zeroes the gradient buffers of all parameters

	print('conv1.bias.grad before backward')
	print(my_net.conv1.bias.grad)

	loss.backward()

	print('conv1.bias.grad after backward')
	print(my_net.conv1.bias.grad)

	# updating weights
	# simple SGD implementation weight = weight - learning_rate * gradient
	learning_rate = 0.01
	for f in my_net.parameters():
    	f.data.sub_(f.grad.data * learning_rate)

    # picking an optimizer
	optimizer = optim.SGD(net.parameters(), lr=0.01)

	# in your training loop:
	optimizer.zero_grad()   # zero the gradient buffers
	output = net(input)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()    # Does the update

#autograd_operations()
neural_network()