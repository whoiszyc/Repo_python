# example to compute the irreduciable inconsistent subsystems
# min x + y
#   x >= 6
#   y >= 6
#   x + y <= 11


import sys
import os
import time
import pandas as pd
import numpy as np
import math
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *
from gurobipy import *
import os
import csv
import ast
import networkx as nx


model = Model('IIS_test')

x = model.addVar(0, vtype = GRB.CONTINUOUS, name = 'x')
y = model.addVar(0, vtype = GRB.CONTINUOUS, name = 'y')

a = model.addConstr(x + y >= 2)
model.addConstr(x + y <= 11)
model.addConstr(x >= 6)
model.addConstr(y >= 6)

model.setObjective(x + y, GRB.MINIMIZE)

model.update()
model.optimize()
model.computeIIS()
model.write("model.ilp")
print("IIS written to file 'model.ilp'")