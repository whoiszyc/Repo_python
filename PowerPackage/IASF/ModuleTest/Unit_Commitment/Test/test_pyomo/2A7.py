from pyomo.environ import *
import numpy as np

# Test of a hybrid model
# Abstract model with concrete parameters


test_coef=[1,1]
iter_coef=np.arange(0,2)

# Define an abstract model
model = AbstractModel()
model.N = Set()
model.M = Set()
model.c = Param(model.N)
model.a = Param(model.N, model.M)
model.b = Param(model.M)
model.x = Var(model.N, within=NonNegativeReals)

def obj_rule(model):
    return sum(model.c[i]*model.x[i] for i in model.N)
model.obj = Objective(rule=obj_rule)


def con_rule(model, m, k):
    return sum(model.a[i,m]*model.x[i] for i in model.N) \
           >= model.b[m]*test_coef[k]
model.con = Constraint(model.M, iter_coef,rule=con_rule)


#  creates a model instance from the abstract Pyomo model using data7
instance = model.create_instance('data7.dat')
instance.pprint()
opt = SolverFactory("glpk")
results = opt.solve(instance)
results.write()

print("Variable x are {} and {}".format(value(instance.x["gen1"]),value(instance.x["gen2"]))  )