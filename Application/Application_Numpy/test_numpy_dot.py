
# This script tests if numpy array can store and compute matrices with object data type, that is, dtype=object

import sympy as sy
import numpy as np

# x_SampleDim_FeatureDim
# y_FeatureDim_OutputDim
# Define symbolic system states
# This is a very important way to initialize numpy array to put object
X = np.array([[sy.symbols('x_{}_{}'.format(i, j)) for j in range(3)] for i in range(10)])
Y = np.array([[sy.symbols('y_{}_{}'.format(i, j)) for j in range(1)] for i in range(3)])
B = np.array([sy.symbols('b_{}_{}'.format(0, j)) for j in range(1)])
Z = np.dot(X, Y) + B
print(Z)


