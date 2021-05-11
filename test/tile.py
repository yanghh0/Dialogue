

import numpy as np


mat = np.array([[1, 2],
                [3, 4]])
x = np.tile(mat, (1, 4))
print(x)
x = np.tile(mat, (2, 1))
print(x)
