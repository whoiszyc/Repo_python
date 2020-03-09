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
# sys.path.append(dir_path+'/z3/z3-4.6.0-x64-osx-10.11.6/bin/python/')  # version 4.6
# import z3

import Fun_IEEETestCase as case
import Fun_Crew_Dispatch as c_d
import formulation_pyomo as fm
import formulation_gurobipy as fg
# import formulation_z3 as fz

ppc = case.case33_noDG_tieline()
vrp = c_d.crew_dispatch_determ()




# # # ----------------------------------------
# # #           co-optimization model
# # # ----------------------------------------
# test_1 = fg.OutageManageGurobi()
# test_1.data_preparation(ppc, vrp)
# test_1.form_cop()
#
# # # set solver parameters
# test_1.model.Params.MIPGap = 0.02
#
# # # solve problem
# test_1.model.update()
# test_1.model.optimize()
#
# # # # get objective function value of MIP
# ObjVal_cop = test_1.model.objVal
# print('objective is {}'.format(ObjVal_cop))
# test_1.get_solution_route()
# load_status = test_1.get_solution_2d('rho', test_1.iter_bus, test_1.iter_time)
# load_status.plot_bin_2d()
# line_status = test_1.get_solution_2d('ul', test_1.iter_line, test_1.iter_time)
# line_status.plot_bin_2d()
# line_flow = test_1.get_solution_2d('P', ['line_1'], test_1.iter_time)
# line_flow.plot_step_2d()
# repair_status = test_1.get_solution_2d('z', test_1.ordered_vertex, test_1.iter_time)
# repair_status.plot_bin_2d()




# # # # ----------------------------------------
# # # #           test master problem
# # # # ----------------------------------------
mp = fg.OutageManageGurobi()
mp.data_preparation(ppc, vrp)
mp.form_mp()

# # set solver parameters
# test_1.model.Params.MIPGap = 0.02

# # solve problem
mp.model.update()
mp.model.optimize()

# # # get objective function value of MIP
ObjVal = mp.model.objVal
print('objective is {}'.format(ObjVal))
mp.get_solution_route()
load_status = mp.get_solution_2d('rho', mp.iter_bus, mp.iter_time)
load_status.plot_bin_2d()
line_status = mp.get_solution_2d('ul', mp.iter_line, mp.iter_time)
line_status.plot_bin_2d()
repair_status = mp.get_solution_2d('z', mp.ordered_vertex, mp.iter_time)
repair_status.plot_bin_2d()
line_flow = mp.get_solution_2d('P', ['line_1'], mp.iter_time)
line_flow.plot_step_2d()



# # # ----------------------------------------
# # #           test crew dispatch problem
# # # ----------------------------------------
# mp = fg.OutageManageGurobi()
# mp.data_preparation(ppc, vrp)
# mp.test_crew_dispatch()
#
# # # set solver parameters
# # test_1.model.Params.MIPGap = 0.02
#
# # # solve problem
# mp.model.update()
# mp.model.optimize()
#
# # # # get objective function value of MIP
# ObjVal = mp.model.objVal
# print('objective is {}'.format(ObjVal))
# mp.get_solution_route()
# repair_status = mp.get_solution_2d('z', mp.ordered_vertex, mp.iter_time)
# repair_status.plot_bin_2d()
# # line_flow = test_1.get_solution_2d('P', ['line_1'], test_1.iter_time)
# # line_flow.plot_step_2d()




# # # ----------------------------------------
# # #           test network operation
# # # ----------------------------------------
# mp = fg.OutageManageGurobi()
# mp.data_preparation(ppc, vrp)
# mp.test_network_operation()
#
# # # set solver parameters
# # test_1.model.Params.MIPGap = 0.02
#
# # # solve problem
# mp.model.update()
# mp.model.optimize()
#
# # # # get objective function value of MIP
# ObjVal = mp.model.objVal
# print('objective is {}'.format(ObjVal))
# load_status = mp.get_solution_2d('rho', mp.iter_bus, mp.iter_time)
# load_status.plot_bin_2d()
# line_status = mp.get_solution_2d('ul', mp.iter_line, mp.iter_time)
# line_status.plot_bin_2d()




# # # ----------------------------------------
# # #   test network operation integer version
# # # ----------------------------------------
# mp = fg.OutageManageGurobi()
# mp.data_preparation(ppc, vrp)
# mp.test_network_operation_int()

# # set solver parameters
# test_1.model.Params.MIPGap = 0.02

# # # solve problem
# mp.model.update()
# mp.model.optimize()
#
# # # # get objective function value of MIP
# ObjVal = mp.model.objVal
# print('objective is {}'.format(ObjVal))
# load_status = mp.get_solution_2d('rho', mp.iter_bus, mp.iter_time)
# load_status.plot_bin_2d()
# line_status = mp.get_solution_2d('ul', mp.iter_line, mp.iter_time)
# line_status.plot_bin_2d()