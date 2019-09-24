from pyomo.environ import *

model = ConcreteModel()

model.A = Set(initialize=[1,2,3])
model.Y = Param(within=model.A)
model.X = Param(within=Reals)
model.W = Param(within=Boolean)

