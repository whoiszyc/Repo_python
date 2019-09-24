import sympy as sy

x1 = sy.Symbol('x1')
x2 = sy.Symbol('x2')

# cost coefficients
coe_1=[0.02533, 25.5472, 24.3891]
coe_2=[0.00660, 19.0800, 230.9689]

cost_1=coe_1[0]*x1**2 + coe_1[1]*x1 + coe_1[2]
cost_2=coe_2[0]*x2**2 + coe_2[1]*x2 + coe_2[2]

Dcost_1=sy.diff(cost_1,x1)
Dcost_2=sy.diff(cost_2,x2)

mc1=Dcost_1.subs(x1,500)

sy.plot(cost_1, (x1, 25, 100))


