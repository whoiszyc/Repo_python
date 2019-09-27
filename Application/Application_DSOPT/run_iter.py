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
import Fun_Build_Results as b_r
import Fun_Crew_Dispatch as c_d
# import formulation_pyomo as fm
import formulation_gurobipy as fg
# import formulation_z3 as fz

## get data
ppc = case.case33_noDG_tieline()
vrp = c_d.crew_dispatch_determ()

## formulate master problem
print('Begin to formulate master problem')
mp = fg.OutageManageGurobi()
mp.data_preparation(ppc, vrp)
mp.form_mp()
mp.model.update()

## formulate subproblem
print('Begin to formulate subproblem')
# # subproblem
sp = fg.OutageManageGurobi()
sp.data_preparation(ppc, vrp)
sp.form_sp()
sp.model.update()


# # supplementary variables to rule out the crew dispatch scenarios
mp.yy = mp.model.addVars(1000, vtype=gb.GRB.BINARY)
CutCounter = 0

# # starting loops
print('Start the iteration')
loop_count = 1
CutList = OrderedDict()
ObjVal_mp = 500000
ObjVal_sp = 0
res = {}
while True:
    # ---------------------------------------------------
    #       Solve master problem for repairing decision
    # ---------------------------------------------------
    res['program_time_start_solve_MP'] = time.time()
    mp.model.optimize()
    res['program_time_complete_solve_MP'] = time.time()

    # ---------------------------------------------------
    #       MP data processing
    # ---------------------------------------------------
    res['program_time_start_process_MP'] = time.time()

    ObjVal_mp = mp.ObjVar.x
    repair_status = mp.get_solution_2d('z', mp.ordered_vertex, mp.iter_time)

    # # get route indices
    CutIndices = []
    for k in mp.iter_crew:
        for i in mp.ordered_vertex:
            for j in mp.ordered_vertex:
                if mp.x[k, i, j].x == 1:
                    CutIndices.append((k, i, j))


    res['program_time_complete_process_MP'] = time.time()

    # -----------------------------------------------------------------------------------
    #   Update the sub-problem
    #   Fix line status to check feasibility of power flow
    # -----------------------------------------------------------------------------------
    # --------- Note ------------
    # If "model_loop = model_flow", any change in the new model will change the original model as well
    # Option 1: add and remove constraints
    # --------- Note End------------

    res['program_time_start_update_SP'] = time.time()

    CutList_loop = OrderedDict()
    # Damaged line cannot function before it is repaired
    for t in sp.iter_time:
        CutList_loop[t] = OrderedDict()
        for i in sp.line_damaged:
            # get the matrix index from the component name
            if len(CutList) == 0: # indicate this is the first iteration, only add cuts
                CutList_loop[t][i] = sp.model.addConstr(sp.ul[i, t] <= repair_status[i][t], 'cut')
            else:
                # add new cuts
                CutList_loop[t][i] = sp.model.addConstr(sp.ul[i, t] <= repair_status[i][t], 'cut')
                # delete old cuts
                sp.model.remove(CutList[t][i])

    res['program_time_complete_update_SP'] = time.time()


    # ---------------------------------------------------
    #       Solve the MIP subproblem for load decision
    # ---------------------------------------------------
    res['program_time_start_solve_SP'] = time.time()

    # solve problem
    sp.model.update()
    sp.model.optimize()

    # get objective function value of MIP
    ObjVal_sp = sp.model.objVal

    res['program_time_complete_solve_SP'] = time.time()

    # # ---------------------------------------------------
    # #       Check break condition
    # # ---------------------------------------------------
    # ##### check gap
    IterGap = abs(ObjVal_mp - ObjVal_sp)/ObjVal_mp
    if IterGap <= 1e-4:
        print('Break in {}th loop with gap {}'.format(loop_count, IterGap))
        break


    # #---------------------------------------------------
    # #       SP data processing
    # #---------------------------------------------------
    # # get line status data
    # plt.figure(figsize=(15, 8))
    # plt.xlabel('Time (step)')
    # plt.ylabel('Line index')
    # y_axis = np.arange(0, len(sp.iter_line))
    # k = 0
    # for i in sp.iter_line:
    #     for t in sp.iter_time:
    #         if sp.ul[i,t].x == 0:
    #             plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
    #         else:
    #             plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
    #     k = k + 1
    # plt.title('SP line status')
    # # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    # plt.show()
    #
    # # get load status data
    # plt.figure(figsize=(15, 8))
    # plt.xlabel('Time (step)')
    # plt.ylabel('Bus index')
    # y_axis = np.arange(0, len(sp.iter_bus))
    # k = 0
    # for i in sp.iter_bus:
    #     for t in sp.iter_time:
    #         if sp.rho[i,t].x == 0:
    #             plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
    #         else:
    #             plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
    #     k = k + 1
    # plt.title('SP load status')
    # # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    # plt.show()

    # res['program_time_start_process_SP'] = time.time()

    # # get load status
    # RHO_MIP = OrderedDict()
    # for i in iter_bus:
    #     RHO_MIP[i] = []
    #     for t in iter_time:
    #         RHO_MIP[i].append(rho[i, t].x)

    # # Get the time instant of load availability from MIP results
    # Load_ava_cut = []
    # for i in range(number_load):
    #     # match the index of load between MIP and SAT
    #     # print(load_initial.iloc[1,i])
    #     # print(type(load_initial.iloc[1, i]))
    #     mip_key = load_initial.iloc[1, i]
    #     Load_ava_cut.append(Total_Time - np.count_nonzero(RHO_MIP[mip_key]))

    ###### print load availability cut
    # print(Load_ava_cut)

    # res['program_time_complete_process_SP'] = time.time()



    # #---------------------------------------------------
    # #       update MP problem
    # #       Add optimality cut
    # #---------------------------------------------------
    res['program_time_start_update_MP'] = time.time()

    ### Add optimality cut as the following form:
    ### If Route = Route_scenario, then load_ava[i] >= Load_ava_cut[i]
    mp.model.addConstr(mp.ObjVar >= ObjVal_sp * (sum(mp.x[i[0], i[1], i[2]] for i in CutIndices) - len(CutIndices) + 1))

    for i in CutIndices:
        mp.model.addConstr(mp.x[i[0], i[1], i[2]] - mp.x[i[0], i[1], i[2]].x <= -0.5 + 5000 * mp.yy[CutCounter])
        mp.model.addConstr(mp.x[i[0], i[1], i[2]] - mp.x[i[0], i[1], i[2]].x >= 0.5 - (1 - mp.yy[CutCounter]) * 5000)

    res['program_time_complete_update_MP'] = time.time()

    # ---------------------------------------------------
    #      Print
    # ---------------------------------------------------
    print('MP solve {}'.format(res['program_time_complete_solve_MP'] - res['program_time_start_solve_MP']))
    print('MP process {}'.format(res['program_time_complete_process_MP'] - res['program_time_start_process_MP']))
    print('SP update {}'.format(res['program_time_complete_update_SP'] - res['program_time_start_update_SP']))
    print('SP solve {}'.format(res['program_time_complete_solve_SP'] - res['program_time_start_solve_SP']))
    print('SAT objective {}'.format(ObjVal_mp))
    print('MIP objective {}'.format(ObjVal_sp))
    print('***********************************')
    print('***********************************')
    print('***********************************')
    print('***********************************')
    print('***********************************')
    print('***********************************')

    # ------------------------------------------------------------
    #       Pass coupling constraint indices for next iteration
    # ------------------------------------------------------------
    loop_count = loop_count + 1
    CutList = CutList_loop
    CutCounter = CutCounter + 1




# # get results
# m = mp1.s.model()
# # get objective function value of SAT (also need to add the served energy)
# ObjVal_SAT = float(m[mp1.EnergyServed].as_long())
# print(ObjVal_SAT)
# # get the scenario of crew dispatch
# Route_scenario = [m[mp1.Route[i]].as_long() for i in range(mp1.number_vertex)]
# print(Route_scenario)



# load_status = cop.get_solution_2d('rho', cop.iter_bus, cop.iter_time)
# load_status.plot_bin_2d()
# line_status = cop.get_solution_2d('ul', cop.iter_line, cop.iter_time)
# line_status.plot_bin_2d()
# line_flow = cop.get_solution_2d('P', ['line_1'], cop.iter_time)
# line_flow.plot_step_2d()
# repair_status = cop.get_solution_2d('z', cop.ordered_vertex, cop.iter_time)
# repair_status.plot_bin_2d()

