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

import _Restoration.Fun_IEEETestCase as case
import _Restoration.Fun_Build_Results as b_r
import _Restoration.Fun_Crew_Dispatch as c_d
import _Restoration.formulation_pyomo as fm
import _Restoration.formulation_gurobipy as fg
import _Restoration.formulation_z3 as fz

## get data
ppc = case.case33_noDG_tieline()
vrp = c_d.crew_dispatch_determ()

## formulate master problem
print('Begin to formulate master problem')
mp = fz.OutageManageZ3()
mp.data_preparation(ppc, vrp)
mp.form_mp(ppc)

## formulate subproblem
print('Begin to formulate subproblem')
sp = fg.OutageManageGurobi()
sp.data_preparation(ppc, vrp)
sp.form_sp()
sp.model.setObjective(sp.obj, gb.GRB.MAXIMIZE)
sp.model.update()


## starting loops
print('Start the iteration')
loop_count = 1
CutList = OrderedDict()
ObjVal_mp = 500000
ObjVal_sp = 0
res = {}
while loop_count < 2:
    # ---------------------------------------------------
    #       Solve master problem for repairing decision
    # ---------------------------------------------------
    res['program_time_start_solve_SAT'] = time.time()
    print(mp.s.check())
    res['program_time_complete_solve_SAT'] = time.time()

    # ---------------------------------------------------
    #       SAT data processing
    # ---------------------------------------------------
    res['program_time_start_process_SAT'] = time.time()

    # get results
    m = mp.s.model()

    # get objective function value of SAT (also need to add the served energy)
    ObjVal_mp = float(m[mp.EnergyServed].as_long())

    # get the scenario of crew dispatch
    Route_scenario = [m[mp.Route[i]].as_long() for i in range(mp.number_vertex)]

    # # get line binary vector
    Line_bin_SAT = OrderedDict()
    for i in range(mp.number_vertex):
        Line_bin_SAT[i] = []
        for t in mp.iter_time:
            if m[mp.Line_bin[i][t]] == True:
                Line_bin_SAT[i].append(1)
            else:
                Line_bin_SAT[i].append(0)
    b_r.plot_binary_evolution(Line_bin_SAT, range(mp.number_vertex), 'MP line status binary vector')

    # # get line status
    Ul_SAT = OrderedDict()
    for i in range(mp.number_line):
        Ul_SAT[i] = []
        for t in mp.iter_time:
            if m[mp.Ul[i][t]] == True:
                Ul_SAT[i].append(1)
            else:
                Ul_SAT[i].append(0)
    b_r.plot_binary_evolution(Ul_SAT, range(mp.number_line), 'MP line status')

    # # print repairing crew dispatch
    print(Route_scenario)

    # print line repairing availability
    for i in range(mp.number_vertex):
        print("line ava time {}".format(m[mp.Line_ava[i]].as_long()))

    # get load status
    RHO_SAT = OrderedDict()
    for i in range(mp.number_bus):
        RHO_SAT[i] = []
        for t in mp.iter_time:
            if m[mp.Rho[i][t]]==True:
                RHO_SAT[i].append(1)
            else:
                RHO_SAT[i].append(0)
    b_r.plot_binary_evolution(RHO_SAT, range(mp.number_bus), 'MP load status')
    # order the vertex by arrival time
    plt.figure(figsize=(7, 5))
    plt.xlabel('Node')
    plt.ylabel('Time')
    plt.bar(np.arange(mp.number_vertex), [m[mp.Time_accum[i]].as_long() for i in range(mp.number_vertex)])
    plt.xticks(np.arange(mp.number_vertex),
               [vrp['ordered_vertex'][m[mp.Route[i]].as_long()] for i in range(mp.number_vertex)])
    plt.title('Available time of components (SMT)')
    plt.show()

    res['program_time_complete_process_SAT'] = time.time()

    # -----------------------------------------------------------------------------------
    #   Update the sub-problem
    #   Fix line status to check feasibility of power flow
    # -----------------------------------------------------------------------------------
    # --------- Note ------------
    # If "model_loop = model_flow", any change in the new model will change the original model as well
    # Option 1: add and remove constraints
    # --------- Note End------------

    res['program_time_start_update_MIP'] = time.time()

    CutList_loop = OrderedDict()
    # Damaged line cannot function before it is repaired
    for t in sp.iter_time:
        CutList_loop[t] = OrderedDict()
        for i in sp.line_damaged:
            # get the matrix index from the component name
            id = int(i[i.find('_') + 1:]) - 1

            # get the index from the ordered vertex
            idd = vrp['ordered_vertex'].index(i)

            if len(CutList) == 0: # indicate this is the first iteration, only add cuts
                CutList_loop[t][i] = sp.model.addConstr(sp.ul[i, t] <= Line_bin_SAT[idd][t], 'cut')
            else:
                # add new cuts
                CutList_loop[t][i] = sp.model.addConstr(sp.ul[i, t] <= Line_bin_SAT[idd][t], 'cut')
                # delete old cuts
                sp.model.remove(CutList[t][i])

    res['program_time_complete_update_MIP'] = time.time()

    # ---------------------------------------------------
    #       Solve the MIP subproblem for load decision
    # ---------------------------------------------------
    res['program_time_start_solve_MIP'] = time.time()

    # solve problem
    sp.model.update()
    sp.model.optimize()

    # get objective function value of MIP
    ObjVal_sp = sp.model.objVal

    res['program_time_complete_solve_MIP'] = time.time()

    # # ---------------------------------------------------
    # #       Check break condition
    # # ---------------------------------------------------
    # ##### check gap
    # if ObjVal_SAT - ObjVal_MIP <= 200:
    #     print('Break in {}th loop'.format(loop_count))
    #     print('SAT solve {}'.format(res['program_time_complete_solve_SAT'] - res['program_time_start_solve_SAT']))
    #     print('SAT process {}'.format(res['program_time_complete_process_SAT'] - res['program_time_start_process_SAT']))
    #     print('MIP update {}'.format(res['program_time_complete_update_MIP'] - res['program_time_start_update_MIP']))
    #     print('MIP solve {}'.format(res['program_time_complete_solve_MIP'] - res['program_time_start_solve_MIP']))
    #     print('MIP process {}'.format(res['program_time_complete_process_MIP'] - res['program_time_start_process_MIP']))
    #     print('SAT objective {}'.format(ObjVal_SAT))
    #     print('MIP objective {}'.format(ObjVal_MIP))
    #     break



    #---------------------------------------------------
    #       MIP data processing
    #---------------------------------------------------
    # get line status data
    plt.figure(figsize=(15, 8))
    plt.xlabel('Time (step)')
    plt.ylabel('Line index')
    y_axis = np.arange(0, len(sp.iter_line))
    k = 0
    for i in sp.iter_line:
        for t in sp.iter_time:
            if sp.ul[i,t].x == 0:
                plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
            else:
                plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
        k = k + 1
    plt.title('SP line status')
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    plt.show()

    # get load status data
    plt.figure(figsize=(15, 8))
    plt.xlabel('Time (step)')
    plt.ylabel('Bus index')
    y_axis = np.arange(0, len(sp.iter_bus))
    k = 0
    for i in sp.iter_bus:
        for t in sp.iter_time:
            if sp.rho[i,t].x == 0:
                plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
            else:
                plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
        k = k + 1
    plt.title('SP load status')
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    plt.show()




    # res['program_time_start_process_MIP'] = time.time()

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

    res['program_time_complete_process_MIP'] = time.time()



    # #---------------------------------------------------
    # #       update SAT problem
    # #       Add optimality cut
    # #---------------------------------------------------
    res['program_time_start_update_SAT'] = time.time()

    ### Add optimality cut as the following form:
    ### If Route = Route_scenario, then load_ava[i] >= Load_ava_cut[i]
    # s.add(z3.If(Route == Route_scenario, EnergyServed >= ObjVal_sp))

    res['program_time_complete_update_SAT'] = time.time()



    # #---------------------------------------------------
    # #       Loop should be going back from here
    # #       The following part is for test
    # #---------------------------------------------------
    # print(s.check())
    # m = s.model()
    # # get the scenario of crew dispatch
    # Route_scenario_1 = [m[Route[i]].as_long() for i in range(number_vertex)]

    # ---------------------------------------------------
    #      Print
    # ---------------------------------------------------
    print('SAT solve {}'.format(res['program_time_complete_solve_SAT'] - res['program_time_start_solve_SAT']))
    print('SAT process {}'.format(res['program_time_complete_process_SAT'] - res['program_time_start_process_SAT']))
    print('MIP update {}'.format(res['program_time_complete_update_MIP'] - res['program_time_start_update_MIP']))
    print('MIP solve {}'.format(res['program_time_complete_solve_MIP'] - res['program_time_start_solve_MIP']))
    print('SAT objective {}'.format(ObjVal_mp))
    print('MIP objective {}'.format(ObjVal_sp))

    loop_count = loop_count + 1
    CutList = CutList_loop





# # get results
# m = mp1.s.model()
# # get objective function value of SAT (also need to add the served energy)
# ObjVal_SAT = float(m[mp1.EnergyServed].as_long())
# print(ObjVal_SAT)
# # get the scenario of crew dispatch
# Route_scenario = [m[mp1.Route[i]].as_long() for i in range(mp1.number_vertex)]
# print(Route_scenario)



# load_status = test_1.get_solution_2d('rho', test_1.iter_bus, test_1.iter_time)
# load_status.plot_bin_2d()
# line_status = test_1.get_solution_2d('ul', test_1.iter_line, test_1.iter_time)
# line_status.plot_bin_2d()
# line_flow = test_1.get_solution_2d('P', ['line_1'], test_1.iter_time)
# line_flow.plot_step_2d()
# repair_status = test_1.get_solution_2d('z', test_1.ordered_vertex, test_1.iter_time)
# repair_status.plot_bin_2d()

