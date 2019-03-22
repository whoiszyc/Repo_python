from sympy import *
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
        a.append(diff(i,j))
    A.append(a)
    print(A)