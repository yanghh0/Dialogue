

import numpy as np


x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
print(np.split(x, 3))
print(np.split(x, [3, 5, 6, 9]))
print(np.split(x, [3, 5, 6, 8]))

a = np.array([[1, 2, 3],
              [1, 2, 5],
              [4, 6, 7]])
print (np.split(a, [2, 3], axis=0))
print (np.split(a, [1, 2], axis=1))