import numpy as np
from numpy import linalg as la
m1 = np.matrix([[2, -1, 0],
               [-1, 2, -1],
               [0, -1, 2]])

eigenvalues = la.eigvals(m1)

print(eigenvalues)
print(la.matrix_rank(m1))



m2 = np.matrix([[1, 1, 1],
                [2, 2, 2],
                [0, -1, 3]])
print(la.matrix_rank(m2))