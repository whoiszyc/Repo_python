from pyomo.environ import *

model = ConcreteModel()

N = [1,2,3]

a = {101:1, 201:3.1, 301:4.5}  # If the keyword is a number, then use it as a number
b = {1:1, 2:2.9, 3:3.1}

model.y = Var(N, within=NonNegativeReals, initialize=0.0)

def CoverConstr_rule(model, i):
    return a[i] * model.y[i] >= b[i]

model.CoverConstr= Constraint(N)

model.pprint()

