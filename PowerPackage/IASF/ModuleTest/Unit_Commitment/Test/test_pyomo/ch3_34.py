from pyomo.environ import *


model = ConcreteModel()
model.x = Var([1,2])


# Three methods to define the objective

# (1)
model.f = Objective(expr=model.x[1] + 2*model.x[2])

# (2)
def TheObjective(model):
    return model.x[1] + 2*model.x[2]
model.g = Objective(rule=TheObjective)

# (3)
def gg_rule(model):
    return model.x[1] + 2*model.x[2]
model.gg = Objective()