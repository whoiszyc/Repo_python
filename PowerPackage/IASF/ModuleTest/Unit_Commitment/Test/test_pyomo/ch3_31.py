from pyomo.environ import *

model = ConcreteModel()

model.a = Var(bounds=(0.0, None))
lower = {1:2, 2:4, 3:6}
upper = {1:3, 2:4, 3:7}

def f(model, i):
    return (lower[i], upper[i])

    
model.b = Var(model.a, bounds=f)


