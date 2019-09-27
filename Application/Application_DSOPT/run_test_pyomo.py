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
# import gurobipy as gb
import pyomo.environ as pm

dir_path = os.path.dirname(os.path.realpath(__file__)) # add Z3 binary distribution to the system path
# sys.path.append(dir_path+'/SatEX/solver/z3/z3-4.4.1-x64-osx-10.11/bin/')  # version 4.4
sys.path.append(dir_path+'/z3/z3-4.6.0-x64-osx-10.11.6/bin/python/')  # version 4.6
import z3

import Fun_IEEETestCase as case
import Fun_Crew_Dispatch as c_d

ppc = case.case33_noDG_tieline()
vrp = c_d.crew_dispatch_determ()



# # test co-optimization
# cop = fm.OutageManagePyomo()
# cop.data_preparation(ppc, vrp)
# cop.form_cop()
# opt = pm.SolverFactory("cplex", executable = '/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
# opt.options['mipgap'] = 0.02  # if gap=b, then it is (b*100) %
# results = opt.solve(cop.model, tee = True)
# print(results['solver'][0]['Termination condition'])
# cop.get_solution_route()
# load_status = cop.get_solution_2d('rho', cop.iter_bus, cop.iter_time)
# load_status.plot_bin_2d()
# line_status = cop.get_solution_2d('ul', cop.iter_line, cop.iter_time)
# line_status.plot_bin_2d()
# line_flow = cop.get_solution_2d('P', ['line_1'], cop.iter_time)
# line_flow.plot_step_2d()



# # test master problem
# cop = fm.OutageManagePyomo()
# cop.data_preparation(ppc, vrp)
# cop.form_mp()
# # opt = pm.SolverFactory("cplex", executable = '/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex') # solver_io="python" or "nl"
# # opt.options['mipgap'] = 0.02  # if gap=b, then it is (b*100) %
# opt = pm.SolverFactory("gurobi", solver_io="python")
# results = opt.solve(cop.model, tee = True)
# print(results['solver'][0]['Termination condition'])
# print('objective is {}'.format(-cop.model.obj.value()))
# cop.get_solution_route()
# # load_status = cop.get_solution_2d('rho', cop.iter_bus, cop.iter_time)
# # load_status.plot_bin_2d()
# # line_status = cop.get_solution_2d('ul', cop.iter_line, cop.iter_time)
# # line_status.plot_bin_2d()
# # line_flow = cop.get_solution_2d('P', ['line_1'], cop.iter_time)
# # line_flow.plot_step_2d()
# # repair_status = cop.get_solution_2d('z', cop.ordered_vertex, cop.iter_time)
# # repair_status.plot_bin_2d()



# t1 = fm.OutageManagePyomo()
# t1.data_preparation(ppc, vrp)
# t1.test_network_operation()
# opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
# # opt.options['mipgap'] = 0.02  # if gap=b, then it is (b*100) %
# results = opt.solve(t1.model, tee=True)
# print(results['solver'][0]['Termination condition'])
# load_status = t1.get_solution_2d('rho', t1.iter_bus, t1.iter_time)
# load_status.plot_bin_2d()
# line_status = t1.get_solution_2d('ul', t1.iter_line, t1.iter_time)
# line_status.plot_bin_2d()
# line_flow = t1.get_solution_2d('P', ['line_1'], t1.iter_time)
# line_flow.plot_step_2d()



# t2 = fm.OutageManagePyomo()
# t2.data_preparation(ppc, vrp)
# t2.test_crew_dispatch()
# opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
# # opt.options['mipgap'] = 0.02  # if gap=b, then it is (b*100) %
# results = opt.solve(t2.model, tee=True)
# print(results['solver'][0]['Termination condition'])
#
# t2.get_solution_route()
# repair_status = t2.get_solution_2d('z', t2.ordered_vertex, t2.iter_time)
# repair_status.plot_bin_2d()