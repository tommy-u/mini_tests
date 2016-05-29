'''
Naive Neural Network. Numerical differentiation of all network parameters
very slow, but trains just fine.
'''
import numpy as np
import math

H = 10**-4

def sig(x): 
    '''Apply sigmoid elementwise.'''
    return 1/(1 + np.exp(-x))

def forward_layer_sig(l, x):
    '''Pass input, x through linear transform and sigmoid fn.'''
    return sig((l[0].dot(x)) + l[1])

def create_net(net, layers):
    '''Create net with bias.'''
    for i in range(len(layers)-1):
        weights = np.random.normal(0, 1, [ layers[i+1], layers[i] ])
        biases = np.random.normal(0, 1, [ layers[i+1], 1 ])
        net.append([weights, biases])

def to_list_arr(l):
    '''Turn [[]] into [np.array]'''
    return [ np.array(e).reshape(len(e),1) for e in l ]

def forward_prop_sig(net, x):
    '''Move input all the way forward through net.'''
    for l in net: 
        x = forward_layer_sig(l, x)
    return x

def mse_cost(l, y): 
    '''Cost summed over all outputs.'''
    return sum((l-y)**2)

def count_net_params(net):
    '''Gets num params.'''
    return sum([l[0].size + l[1].size for l in net])

def peek_net_param(net, i):
    '''Get int indexed param.'''
    bottom = 0
    assert i >= 0
    assert i < count_net_params(net)
    assert i == math.floor(i)
    for l in net:
        top = bottom + l[0].size + l[1].size
        if i >= bottom and i < top:
            if i >= bottom and i < bottom + l[0].size:
                return l[0].item(i-bottom) #NO PARTICULAR ORDERING
            else:
                return l[1].item(i-(bottom + l[0].size))
        bottom = top
    return None

def poke_net_param(net, i, v):
    '''Modify network param.'''
    bottom = 0
    assert i >= 0
    assert i < count_net_params(net)
    assert i == math.floor(i)
    for l in net:
        top = bottom + l[0].size + l[1].size
        if i >= bottom and i < top:
            if i >= bottom and i < bottom + l[0].size:
                l[0].itemset(i-bottom, v)
                return net #NO PARTICULAR ORDERING
            else:
                l[1].itemset(i-(bottom + l[0].size), v)
                return net
        bottom = top
    return None

def deriv_param(net, l, inp, i):
    '''Find deriv of cost wrt param.'''
    #Get old val param.
    param = peek_net_param(net, i)
    #Get old cost.
    old_cost = mse_cost(l, forward_prop_sig(net, inp))
    #Poke epsilon diff on param into net
    net = poke_net_param(net, i, param+H)
    #Get new cost.
    new_cost = mse_cost(l, forward_prop_sig(net, inp))
    #Reset param to old val
    net = poke_net_param(net, i, param)
    diff = new_cost - old_cost
    return diff / H

def all_params_grad(net, l, inp):
    '''Get grad of all params.'''
    return np.array([ deriv_param(net, l, inp, i) for i in range(count_net_params(net))])

def update_params(net, grad):
    '''Apply SGD to update net params'''
    for i in range(count_net_params(net)):
        net = poke_net_param(net, i, (peek_net_param(net, i) - grad[i]))

def test_net():
    '''Unit test.'''
    np.random.seed(seed=2)

    net = []
    layers = [2, 2, 4, 1]
    create_net(net, layers)
  
    my_inputs =  [ [0, 0], [0, 1], [1,0], [1,1] ]
    my_labels =  [ [0, 0], [0, 1], [1,0], [1,1] ]
    my_labels2 = [    [0],    [1],   [1],   [0] ]
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

test_net()
