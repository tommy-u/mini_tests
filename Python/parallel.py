#http://scottsievert.com/blog/2014/07/30/simple-python-parallelism/
from functools import partial
import numpy as np
from multiprocessing import Pool

N = 10000
serial = []
def ez_para(f, sequence):
    """ assumes f takes sequence as input, easy w/ Python's scope """
    pool = Pool(processes=4) # depends on available cores
    result = pool.map(f, sequence) # for i in sequence: result[i] = f(i)
    cleaned = [x for x in result if not x is None] # getting results
    cleaned = np.asarray(cleaned)
    pool.close() # not optimal! but easy
    pool.join()
    return cleaned

def test_prime(n):
    prime = True
    for i in range(2, n):
        if n/i % 1 == 0:
            prime = False
    return prime

for i in range(3, N):
    serial.append(test_prime(i))

#Returns a partial object which behaves like ez_para
#called with test prime as an arg.
my_f = partial(ez_para, test_prime)

#Runs test_prime on all elements in list.
#Result is [test_prime(3), ... text_prime(N)]
parallel_result = my_f(list(range(3,N)))



