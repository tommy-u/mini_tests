import numpy as np
import math
h = 10**-2

#Simple approximation of 1st deriv.
def sym_der(f, x): return (f(x+(h/2))-f(x-(h/2))) / h

#Deriv, not symmetric.
def der(f,x): return (f(x+(h))-f(x)) / h

#Sigmoid fn.
def sig(x): return 1/(1 + np.exp(-x))

#Create 1 layer net with bias.
def create_net(net, layers):
	for i in range(len(layers)-1):
		weights = np.random.normal(0, 1, [ layers[i+1], layers[i] ])
		biases = np.random.normal(0, 1, [ layers[i+1], 1 ])
		net.append([weights, biases])

#Turn [[]] into [np.array]
def to_list_arr(l):
	tmp = []
	for e in l:
		tmp.append(np.array(e).reshape(len(e),1) )
	return tmp

#Forward motion through net.
def forward_layer_sig(net, x, i):  return sig ( (net[i][0] @ x) + net[i][1] )
#def forward_layer_tanh(net, x, i): return tanh( (net[i][0] @ x) + net[i][1] )

#Move input all the way forward through net.
def forward_prop_sig(net, x):
	for i in range(len(net)):
		if i == 0:
			h = forward_layer_sig(net, x, i)
		else:
			h = forward_layer_sig(net, h, i)
	return h

#Get predictions for each input.
def get_predictions(net, input):
	y = []
	for e in my_inputs:
		y.append(forward_prop_sig(net, e))
	return y

#Cost summed over all outputs.
def mse_cost(l, y): return sum((l-y)**2)

#Gets num params.
def count_net_params(net):
	size = 0
	for l in net:
		size += l[0].size + l[1].size
	return size

#Get int indexed param.
def peek_net_param(net, i):
	bottom = 0
	assert(i>=0)
	assert(i<count_net_params(net))
	assert(i==math.floor(i))
	for l in net:
		top = bottom + l[0].size + l[1].size
		if i >= bottom and i < top:
			if i >= bottom and i < bottom + l[0].size:
				return(l[0].item(i-bottom)) #NO PARTICULAR ORDERING
			else:
				return(l[1].item(i-(bottom + l[0].size)))
		bottom = top
	return None

#Set int indexed param.
def poke_net_param(net, i, v):
	bottom = 0
	assert(i>=0)
	assert(i<count_net_params(net))
	assert(i==math.floor(i))
	for l in net:
		top = bottom + l[0].size + l[1].size
		if i >= bottom and i < top:
			if i >= bottom and i < bottom + l[0].size:
				l[0].itemset(i-bottom, v) 
				return(net) #NO PARTICULAR ORDERING
			else:
				l[1].itemset(i-(bottom + l[0].size), v)
				return(net)
		bottom = top
	return None

#Calc deriv numerically.
def deriv_param(net, l, inp, i):
	#Get old val param.
	param = peek_net_param(net, i)
	#Get old cost.
	old_cost = mse_cost(l, forward_prop_sig(net, inp))
	#Poke epsilon diff on param into net
	net = poke_net_param(net, i, param+h)
	#Get new cost.
	new_cost = mse_cost(l, forward_prop_sig(net, inp))
	#Reset param to old val
	net = poke_net_param(net, i, param)
	diff = new_cost - old_cost
	return diff / h

def all_params_grad(net, l, inp):
	num_params = count_net_params(net)
	grad = np.zeros(num_params)
	for i in range(num_params):
		grad[i] = deriv_param(net, l, inp, i)
	return grad

def update_params(net, grad):
	for i in range(count_net_params(net)):
		net = poke_net_param(net, i, (peek_net_param(net, i) - grad[i]))

np.random.seed(seed=2)

net = []
layers = [2, 2, 1]
create_net(net, layers)	

my_inputs = [ [0, 0], [0, 1], [1,0], [1,1] ]
my_labels = [ [0, 0], [0, 1], [1,0], [1,1] ]
my_labels2 = [ [0], [1], [1], [0] ]
my_labels = my_labels2
my_inputs = to_list_arr(my_inputs)
my_labels = to_list_arr(my_labels)

print("error before")
print("error for 3rd input random init")
print(mse_cost(my_labels[3], forward_prop_sig(net, my_inputs[3])))

print("gradient for all params using 3rd input")

for i in range(1000):
	for j in range(len(my_inputs)):
		grad = all_params_grad(net, my_labels[j], my_inputs[j])
		update_params(net, grad)
		print(mse_cost(my_labels[j], forward_prop_sig(net, my_inputs[j])))

for i in range(len(my_inputs)):
	print(forward_prop_sig(net, my_inputs[i]))

#Turn network params into 1d vec
# def flatten_net(net):
# 	flag = 0
# 	for l in net:
# 		if flag == 0:
# 			a = np.concatenate((l[0].flatten(), l[1].flatten()), axis=0)
# 		else:
# 			a = np.concatenate((a, l[0].flatten(), l[1].flatten()), axis=0)
# 		flag = 1
# 	return a

#Turn 1d vec into net
# def expand_flat_net(net, f):
# 	count = 0
# 	for l in net:
# 		s = l[0].shape
# 		sz = s[0] * s[1]
# 		l[0] = f[count: count + sz].reshape(s)
# 		count += sz

# 		s = l[1].shape
# 		sz = s[0] * s[1]
# 		l[1] = f[count: count + sz].reshape(s)
# 		count += sz
# 	return net

#l and y must be in corresponding order
#Get cost corresponding to one input label pair.

