from timeit import timeit

import numpy as np
arr = np.random.rand(1, 100)

def test_floor():
    np.floor(arr).view(np.int32)

def test_as():
    arr.astype(np.int32)

def test_as_nocpy():
    arr.astype(int)

time_a = timeit(test_floor, number=10000)
time_b = timeit(test_as, number=10000)
time_c = timeit(test_as_nocpy, number=10000)

print('Time A: ' + str(time_a))
print('Time B: ' + str(time_b))
print('Time C: ' + str(time_c))

