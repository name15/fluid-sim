import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([2, 3])

a[:] = b[:, np.newaxis]

print(a)