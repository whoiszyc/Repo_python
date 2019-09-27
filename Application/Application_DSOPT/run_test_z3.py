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
import os
import ast
import networkx as nx
import operator
from collections import OrderedDict
import gurobipy as gb
import pyomo.environ as pm

dir_path = os.path.dirname(os.path.realpath(__file__)) # add Z3 binary distribution to the system path
# sys.path.append(dir_path+'/SatEX/solver/z3/z3-4.4.1-x64-osx-10.11/bin/')  # version 4.4
sys.path.append(dir_path+'/z3/z3-4.6.0-x64-osx-10.11.6/bin/python/')  # version 4.6
import z3

import Fun_IEEETestCase as case
import Fun_Crew_Dispatch as c_d
import formulation_pyomo as fm
import formulation_z3 as fz

ppc = case.case33_noDG_tieline()
vrp = c_d.crew_dispatch_determ()



mp1 = fz.OutageManageZ3()
mp1.data_preparation(ppc, vrp)
mp1.form_mp(ppc)
solver_start = time.time()
print(mp1.s.check())
solver_complete = time.time()
print('solver time {}'.format(solver_complete - solver_start))
# get results
m = mp1.s.model()
# get objective function value of SAT (also need to add the served energy)
ObjVal_SAT = float(m[mp1.EnergyServed].as_long())
print(ObjVal_SAT)
# get the scenario of crew dispatch
Route_scenario = [m[mp1.Route[i]].as_long() for i in range(mp1.number_vertex)]
print(Route_scenario)


