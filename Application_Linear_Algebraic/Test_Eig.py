
import numpy as np
from numpy import linalg as la

m1 = [[2, -1, 0],[-1, 2, -1],[0, -1, 2]]
print(la.eigvals(m1))
print(la.matrix_rank(m1))
