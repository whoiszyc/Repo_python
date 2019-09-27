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

# dir_path = os.path.dirname(os.path.realpath(__file__)) # add Z3 binary distribution to the system path
# # sys.path.append(dir_path+'/SatEX/solver/z3/z3-4.4.1-x64-osx-10.11/bin/')  # version 4.4
# sys.path.append(dir_path+'/z3/z3-4.6.0-x64-osx-10.11.6/bin/python/')  # version 4.6
# import z3

from gurobipy import *
import _Restoration.Fun_IEEETestCase as case
import _Restoration.Fun_Crew_Dispatch as c_d
import _Restoration.formulation_pyomo as fm
import _Restoration.formulation_gurobipy as fg
import _Restoration.formulation_z3 as fz

ppc = case.case33_noDG_tieline()
vrp = c_d.crew_dispatch_determ()


def branch_cut(model, where):
    if where == GRB.Callback.MIPSOL:
        # # MIP node callback
        # # retrieve master problem objective
        ObjVal_mp = model.cbGetSolution(model._ObjVar)
        print('***** New Incumbent Nodes with Objective {} *****'.format(ObjVal_mp))

        # # retrieve crew dispatch results
        X = model.cbGetSolution(model._x)
        # # identify the indices that x equals to one
        cut_indices = []
        for i in range(len(X)):
            if X[i] == 1:
                cut_indices.append(i)
        print('***** New Incumbent Nodes with Dispatch {} *****'.format(cut_indices))

        # # retrieve line availability indicator
        Z = model.cbGetSolution(model._z)

        # # dictionary for constraint indices
        CutList_loop = OrderedDict()

        # # Damaged line cannot function before it is repaired
        for i in sp.line_damaged:
            CutList_loop[i] = OrderedDict()
            # get the index from the ordered vertex
            idd = vrp['ordered_vertex'].index(i)

            for t in model._sp.iter_time:
                # compute corresponding list index
                idd_t = idd * model._sp.Total_Time + t

                if len(model._CutList) == 0:  # indicate this is the first iteration, only add cuts
                    CutList_loop[i][t] = model._sp.model.addConstr(sp.ul[i, t] <= Z[idd_t])
                else:
                    # add new cuts
                    CutList_loop[i][t] = model._sp.model.addConstr(sp.ul[i, t] <= Z[idd_t])
                    # delete old cuts
                    model._sp.model.remove(model._CutList[i][t])

        # # update and solve subproblem
        model._sp.model.update()
        model._sp.model.Params.OutputFlag = False
        model._sp.model.optimize()
        ObjVal_sp = model._sp.model.objVal
        print('***** Subproblem Solved with Objective {} *****'.format(ObjVal_sp))

        # # add optimality combinatorial Benders cut
        model.cbLazy(model._ObjVar >= ObjVal_sp * (sum(model._x[i] for i in cut_indices) - sum(X) + 1))

        # # rule out current dispatch scenarios
        # #  It is simplily just a not-euqal constraint, however, ...
        # # (1) These type of constraints (x != y) are not supported in Gurobi, nor are they supported in Linear Programming
        # # / Mixed-Integer Programming in general. This type of constraint doesn't work natively because the feasible set consists of two feasible regions.
        # # (2) Strictly inequality constraints are against the nature of the linear programming. Your feasible set needs to be closed.
        # for i in cut_indices:
        #     model.cbLazy(model._x[i] - X[i] <= -0.5 + 5000 * model._yy[model._CutCounter])
        #     model.cbLazy(model._x[i] - X[i] >= 0.5 - (1 - model._yy[model._CutCounter]) * 5000)
        #
        # model._CutCounter = model._CutCounter + 1





# # master problem
mp = fg.OutageManageGurobi()
mp.data_preparation(ppc, vrp)
mp.form_mp()
mp.model.Params.LazyConstraints = 1
mp.model.update()

# # subproblem
sp = fg.OutageManageGurobi()
sp.data_preparation(ppc, vrp)
sp.form_sp()
sp.model.update()

# # add new attribute to the model class of gurobi for callback
# # In gurobipy.Model, new attribute has to start with _
# # define a vector of variables to express not-equal constraint
mp.yy = mp.model.addVars(1000, vtype=gb.GRB.BINARY)

# # add callback list
# mp.model._yy = mp.yy.values()
mp.model._ObjVar = mp.ObjVar
mp.model._z = mp.z.values()
mp.model._x = mp.x.values()

# # add a cut count
# mp.model._CutCounter = 0

# # pass subproblem object
mp.model._sp = sp
# # pass subproblem constraint list
mp.model._CutList = OrderedDict()


# # solve problem with callback
mp.model.update()
mp.model.optimize(branch_cut)
# mp.model.optimize()




# # # get objective function value of MIP
# ObjVal = mp.model.objVal
# print('objective is {}'.format(ObjVal))
# mp.get_solution_route()
# load_status = mp.get_solution_2d('rho', mp.iter_bus, mp.iter_time)
# load_status.plot_bin_2d()
# line_status = mp.get_solution_2d('ul', mp.iter_line, mp.iter_time)
# line_status.plot_bin_2d()
# repair_status = mp.get_solution_2d('z', mp.ordered_vertex, mp.iter_time)
# repair_status.plot_bin_2d()
# line_flow = cop.get_solution_2d('P', ['line_1'], cop.iter_time)
# line_flow.plot_step_2d()



