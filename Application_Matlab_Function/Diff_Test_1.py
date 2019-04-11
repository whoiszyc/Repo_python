from sympy import *
import numpy as np
from numpy import linalg as la


x, y, z= symbols('x y z')

symvar=[x, y, z]

f= []
f.append(5*x + 8*y + 8*z)
f.append(2*x + 3*z + 2*y + 5*x)
f.append(0*x + 0*z + y)

A=[]
for i in f:
    a=[]
    print(i)
    for j in symvar:
        a.append(float(diff(i,j)))
    A.append(a)
    print(A)

# before using linear algebraic tool, convert the data type first
xx = la.eig(A)
print(xx)


# change symbolic variable to string
x_name = str(x)
type(x_name)