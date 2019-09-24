from pyomo.environ import *

model = ConcreteModel()
model.x_1 = Var(within=NonNegativeReals)
model.x_2 = Var(within=NonNegativeReals)
model.obj = Objective(expr=model.x_1 + 2*model.x_2)
model.con1 = Constraint(expr=3*model.x_1 + 4*model.x_2 >= 1)
model.con2 = Constraint(expr=2*model.x_1 + 5*model.x_2 >= 2)

# model.pprint()

opt = SolverFactory("glpk")
results = opt.solve(model)
results.write()

print("First value equals to {}".format(value(model.x_1)))
print("Second value equals to {}".format(value(model.x_2)))

print(model.x_1.value)
print(model.x_2.value)