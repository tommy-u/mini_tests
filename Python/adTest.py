'''Hackery'''


import numpy as np
import ad.admath
ONE = ad.adnumber(np.array([1.0]))
def to_list_arr(l):
    '''Turn [[]] into [np.array]'''
    return [ np.array(e).reshape(len(e),1) for e in l ]

def create_net(net, layers):
    '''Create net with bias.'''
    for i in range(len(layers)-1):
        weights = np.random.normal(0, 1, [ layers[i+1], layers[i] ])
        biases = np.random.normal(0, 1, [ layers[i+1], 1 ])
        net.append([weights, biases])

def mse_cost(l, y): 
    '''Cost summed over all outputs.'''
    return sum((l-y)**2)

def mse(net, x, lab):
    '''Move input all the way forward through net.'''
    for l in net: 
        x = 1/(ONE + ad.admath.exp(-(l[0].dot(x)) + l[1]))
    return sum((lab-x)**2)

def flat_forward_prop_sig_np(net, x):
    '''Move input all the way forward through net.'''
    for l in net: 
        x = 1/(1 + np.exp(-(l[0].dot(x)) + l[1]))
    return x

def my_sq(x, a):
    return x**2

def works():
    np.random.seed(seed=2)
    net = []
    layers = [2, 2]
    create_net(net, layers)
    #Some data.
    my_inputs =  [ [0, 0], [0, 1], [1,0], [1,1] ]
    my_labels =  [ [0, 0], [0, 1], [1,0], [1,1] ]
    my_inputs = to_list_arr(my_inputs)
    my_labels = to_list_arr(my_labels)
    f = flat_forward_prop_sig_np(net, my_inputs[3])
    print(f)
    # print(type(ad_net))


def test():
    np.random.seed(seed=2)
    net = []
    layers = [2, 2]
    create_net(net, layers)
    #Some data.
    my_inputs =  [ [0, 0], [0, 1], [1,0], [1,1] ]
    my_labels =  [ [0, 0], [0, 1], [1,0], [1,1] ]
    my_inputs = to_list_arr(my_inputs)
    my_labels = to_list_arr(my_labels)
    ad_net = ad.adnumber(net)
    # ad_my_inputs = ad.adnumber(my_inputs)
    in_x = ad.adnumber(my_inputs[3])
    in_l = ad.adnumber(my_labels[3])
    m = mse(ad_net, in_x, in_l)
    print(m)
    print(m[0].gradient(in_x))




def exp():
    a = np.array([1,2,3])
    ad_a = ad.adnumber(a)
    # print(a)
    # print(ad_a)
    print(np.exp(a))
    print(ad.admath.exp(a))



def my_sum(x):
    print(x)
    return sum(x)

# exp()
# works()
test()
exit()



z = np.array([1,2,3])
ad_z = ad.adnumber(z)

out = my_sum(ad_z)
print(out)

print(out.gradient(ad_z))

















