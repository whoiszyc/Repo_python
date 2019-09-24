# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 21:44:37 2018

@author: Yichen Zhang
"""

#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# This example formulates and solves the following simple QP model:
#  minimize
#      x^2 + x*y + y^2 + y*z + z^2 + 2 x
#  subject to
#      x + 2 y + 3 z >= 4
#      x +   y       >= 1
#
# It solves it once as a continuous model, and once as an integer model.

from gurobipy import *

# Create a new model
m = Model("UC")

# Create variables
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")
z = m.addVar(vtype=GRB.BINARY, name="z")

# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
# obj = x*x + x*y + y*y + y*z + z*z + 2*x
obj=x + y + 2 * z
m.setObjective(obj,GRB.MAXIMIZE)

# Add constraint: x + 2 y + 3 z <= 4
m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

# Add constraint: x + y >= 1
m.addConstr(x + y >= 1, "c1")

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())

"""
x.vType = GRB.INTEGER
y.vType = GRB.INTEGER
z.vType = GRB.INTEGER

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())
"""