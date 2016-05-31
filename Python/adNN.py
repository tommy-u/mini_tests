'''
Naive Neural Network. Numerical differentiation of all network parameters
very slow, but trains just fine.
'''
import numpy as np
# import ad
import ad.admath
ONE = ad.adnumber(np.array([1.0]))
H = 10**-4

def sig(x): 
    '''Apply sigmoid elementwise.'''
    return 1/( ONE + ad.admath.exp(-x))

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

def forward_prop_sig(net, x, f=None):
    if f is not None:
        net = expand_flat_net(net,f)
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
                return l[0].item(i-bottom) #Order?
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

def flatten_net(net):
    '''Turns net into flat vec of params.'''
    flag = 0
    for l in net:
        if flag == 0:
            a = np.concatenate((l[0].flatten(), l[1].flatten()), axis=0)
        else:
            a = np.concatenate((a, l[0].flatten(), l[1].flatten()), axis=0)
        flag = 1
    return a

def expand_flat_net(net, f):
    '''Reconstructs net from flat vec of params.'''
    count = 0
    for l in net:
        s = l[0].shape
        sz = s[0] * s[1]
        l[0] = f[count: count + sz].reshape(s)
        count += sz

        s = l[1].shape
        sz = s[0] * s[1]
        l[1] = f[count: count + sz].reshape(s)
        count += sz
    return net

def update_params(net, grad):
    '''Apply SGD to update net params'''
    f = flatten_net(net)
    #Move against grad
    return expand_flat_net(net, f - grad)

def get_all_outputs(net, inp):
    '''Get outputs for each input.'''
    return [forward_prop_sig(net, e) for e in inp]

def train_epoch(net, l, inp):
    '''One round of online training for each element in input set.'''
    for j in range(len(inp)):
        ad_f = ad.adnumber(flatten_net(net))
        mse = mse_cost( l[j], forward_prop_sig(net, inp[j], ad_f))
        update_params(net, mse[0].gradient(ad_f))
        # print(mse_cost(l[j], forward_prop_sig(net, inp[j])))

def unit_test():
    '''Unit test.'''
    np.random.seed(seed=2)

    net = []
    layers = [2, 3, 1]
    create_net(net, layers)

    #Some data.
    my_inputs =  [ [0, 0], [0, 1], [1,0], [1,1] ]
    my_labels =  [ [0, 0], [0, 1], [1,0], [1,1] ]
    my_labels2 = [    [0],    [1],   [1],   [0] ]
    my_labels = my_labels2
    my_inputs = to_list_arr(my_inputs)
    my_labels = to_list_arr(my_labels)
    ad_f = ad.adnumber(flatten_net(net))
    print("Before training")
    # print(get_all_outputs(net, my_inputs))
    print(mse_cost(my_labels[3], forward_prop_sig(net, my_inputs[3], ad_f))[0])
    # for i in range(100):
        # train_epoch(net, my_labels, my_inputs)

    #train on one pair

    for i in range(2):
        for j in range(4):
            ad_f = ad.adnumber(flatten_net(net))
            all_params_grad = mse_cost(my_labels[j], forward_prop_sig(net, my_inputs[j], ad_f))[0]
            update_params(net, all_params_grad)
            print("h2")
        print("h1")


    print("After training")
    print(mse_cost(my_labels[3], forward_prop_sig(net, my_inputs[3], ad_f))[0])
    # print(get_all_outputs(net, my_inputs))
    
unit_test()

















