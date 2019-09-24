from pyomo.environ import *


model = ConcreteModel()
model.x = Var([1,2])


# Three methods to define the constraints

# (1)
model.Diff= Constraint(expr=model.x[2]-model.x[1] <= 7.5)

# (2)
def Diff(model):
    return model.x[2] - model.x[1] <= 7.5
model.Diff = Constraint(rule=Diff)

# (3)
def Diff_rule(model):
    return model.x[2] - model.x[1] <= 7.5
model.Diff = Constraint()


model.pprint()






