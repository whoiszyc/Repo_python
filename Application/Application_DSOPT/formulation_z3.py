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


dir_path = os.path.dirname(os.path.realpath(__file__)) # add Z3 binary distribution to the system path
# sys.path.append(dir_path+'/SatEX/solver/z3/z3-4.4.1-x64-osx-10.11/bin/')  # version 4.4
sys.path.append(dir_path+'/z3/z3-4.6.0-x64-osx-10.11.6/bin/python/')  # version 4.6
import z3
from formulation_general import *



class OutageManageZ3(OutageManage):
    """
    crew dispatch and flow-driven crew dispatch in z3
    """


    ## define If-then constraints for total time
    def time_fn(self, c1, c2):
        t = 9999999  # default value
        for i in range(self.number_vertex):
            for j in range(self.number_vertex):
                t = z3.If(z3.And(c1 == i, c2 == j), self.ordered_total_time[i, j], t)  # +ordered_repair_time[i]
        # print(t)
        return t



    ## Define the crew dispatch availability indicator
    ## define find index function using If-then
    def time_index_fn(self, c):
        t = 9999999  # default value
        for i in range(self.number_vertex):
            t = z3.If(self.Route[i] == c, self.Time_accum[i], t)
        # print(t)
        return t



    def form_crew_dispatch(self, Route_Example=None):
        """

        :return:
        """

        ## optimal satisfiability object
        self.s = z3.Optimize()

        ## Which vertex is visited on each step of route? (starting from 0 and end at 7)
        self.Route = [z3.Int('route_%d' % i) for i in self.index_vertex]

        ## Time instants that the vehicle leave the vertex corresponding to Route
        self.Time_accum = [z3.Int('time_accum_%d' % i) for i in range(self.number_vertex)]

        ## Available time for each damaged component
        self.Line_ava = [z3.Int('line_ava_{}'.format(i)) for i in range(self.number_vertex)]

        ## line functioning indicator coupled with network constraint
        self.Line_bin = [[z3.Bool('l_{}_time_{}'.format(i, j)) for j in self.iter_time] for i in range(self.number_vertex)]

        ## all ints in route[] must be in [0..cities_t) range:
        for r in self.Route:
            self.s.add(r >= 0, r < self.number_vertex)

        self.s.add(self.Route[0] == 0)
        self.s.add(self.Route[self.number_vertex - 1] == self.number_vertex - 1)

        ## no city may be visited twice:
        self.s.add(z3.Distinct(self.Route))

        ## define total travel time
        self.s.add(self.Time_accum[0] == 0)

        for i in range(1, self.number_vertex):
            self.s.add(self.Time_accum[i] == self.Time_accum[i - 1] + self.time_fn(self.Route[i - 1], self.Route[(i) % self.number_vertex]))

        ## get the available time in a list in the order of component index
        self.s.add(self.Line_ava[0] == 0)

        for i in range(1, self.number_vertex):
            self.s.add(self.Line_ava[i] == self.time_index_fn(i))

        ## Convert the line time availability instant into a vector
        for i in range(self.number_vertex):
            for j in self.iter_time:
                self.s.add(self.Line_bin[i][j] == z3.If(self.Line_ava[i] >= j, False, True))

        ## Add the crew dispatch valid inequalities
        ## Minimum time that line k could be available if the crew repair k first
        for i in self.line_damaged:
            idd = self.ordered_vertex.index(i)
            self.s.add(self.Line_ava[idd] >= self.ordered_total_time[0, idd])

        if Route_Example != None:
            print('Route example is given')
            for i in range(len(Route_Example)):
                 self.s.add(self.Route[i] == Route_Example[i])



    def form_network_operation(self, ppc):
        ## line switch variables
        self.Ul = [[z3.Bool('ul_{}_time_{}'.format(i, t)) for t in self.iter_time] for i in range(self.number_line)]

        ## fictitious power flow variables
        self.Pl = [[z3.Int('pl_{}_time_{}'.format(i, t)) for t in self.iter_time] for i in range(self.number_line)]

        ## tree topology constraint variables
        self.Beta = [[[z3.Bool('beta_{}_{}_time_{}'.format(i, j, t)) for t in self.iter_time] for j in range(self.number_bus)] for i in range(self.number_bus)]

        ## load status variables
        self.Rho = [[z3.Bool('rho_{}_time_{}'.format(i, t)) for t in self.iter_time] for i in range(self.number_bus)]

        ## Once a load is picked up, it cannot be shed
        for t in range(1, self.Total_Time):  # use set type as iterators are easy to implement N\{m} type of constraints
            for i in range(self.number_bus):
                self.s.add(z3.If(self.Rho[i][t], 1, 0) >= z3.If(self.Rho[i][t - 1], 1, 0))

        ## static line cannot change
        for t in self.iter_time:
            for i in self.line_static:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1
                self.s.add(self.Ul[id][t] == True)

        ## Damaged line can be closed if and only if it is repaired first
        for t in self.iter_time:
            for i in self.line_damaged:
                # get the matrix index from the component name
                id_line = int(i[i.find('_') + 1:]) - 1
                # get the index from the component name corresponding to the index in the crew dispatch problem
                id_route = self.ordered_vertex.index(i)
                self.s.add(z3.If(self.Ul[id_line][t], 1, 0) <= z3.If(self.Line_bin[id_route][t], 1, 0))

        # # # # Once a damaged line is closed, it cannot be opened
        for t in range(1, self.Total_Time):  # use set type as iterators are easy to implement N\{m} type of constraints
            for i in self.line_damaged:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1
                self.s.add(z3.If(self.Ul[id][t], 1, 0) >= z3.If(self.Ul[id][t - 1], 1, 0))

        # # substation bus does not have parent bus
        for t in self.iter_time:
            self.s.add(self.Beta[1][0][t] == False)  # hard coded

        # # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in range(self.number_bus):
                self.s.add(sum(z3.If(self.Beta[i][j][t], 1, 0) for i in range(self.number_bus)) <= 1)

        for t in self.iter_time:
            for i in range(self.number_line):
                # for this line, get the bus index
                f_bus = int(ppc['line'][i, 1]) - 1
                t_bus = int(ppc['line'][i, 2]) - 1
                self.s.add(z3.If(self.Beta[f_bus][t_bus][t], 1, 0) + z3.If(self.Beta[t_bus][f_bus][t], 1, 0) == z3.If(
                    self.Ul[i][t], 1, 0))

        ## Fictitious power flow constraint
        ### Here we will use a power flow-like constraint, where each line contains an integer variable to represent its line flow line flow limit
        ### The logic is to set line flow to zero if Ul is zero
        for t in self.iter_time:
            for i in range(self.number_line):
                self.s.add(self.Pl[i][t] <= z3.If(self.Ul[i][t], 1, 0) * self.FicFlowBigM)
                self.s.add(self.Pl[i][t] >= -z3.If(self.Ul[i][t], 1, 0) * self.FicFlowBigM)

        ### DistFlow constraint: power balance at bus i
        for t in self.iter_time:
            for i in range(1, self.number_bus):  # looping from bus 1, substation bus will be considered specifically
                P_out = 0
                P_in = 0

                # create bus name
                bus_name = 'bus_{}'.format(i + 1)

                # get power flow variables flowing out from this bus
                iter_out = np.arange(0, len(self.bus_line[bus_name]["line_from_this_bus"]))

                if iter_out.shape[0] == 0:
                    P_out = P_out
                else:
                    for k in iter_out:
                        aaa = int(self.bus_line[bus_name]["line_from_this_bus"][k]) - 1
                        P_out = P_out + self.Pl[aaa][t]

                # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[bus_name]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                else:
                    for k in iter_in:
                        aaa = int(self.bus_line[bus_name]["line_to_this_bus"][k]) - 1
                        P_in = P_in + self.Pl[aaa][t]

                # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][i, 1] == 1:
                    self.s.add(P_in == P_out + z3.If(self.Rho[i][t], 1, 0))

                # if this is a bus with only load
                else:
                    self.s.add(P_in == P_out + z3.If(self.Rho[i][t], 1, 0))



    def define_objective(self, ppc):
        # # # objective 1: minimize total time
        # obj_1 = Time_accum[number_vertex - 1]

        # # objective 2: all damaged components should be repaired
        # obj_2 = sum(sum(Hazard * z3.If(Line_bin[i][j], 1, 0) for i in range(number_vertex)) for j in iter_time)

        # objective 3: served energy should be maximized
        self.EnergyServed = z3.Int('EnergyServed')
        self.s.add(self.EnergyServed == sum(
            sum(ppc['bus'][i, 4] * 1000 * z3.If(self.Rho[i][j], 1, 0) for i in range(self.number_bus)) for j in self.iter_time))

        # summation the objectives
        self.s.maximize(self.EnergyServed)



    def form_mp(self, ppc, Route_Example=None):
        """
        formulate the master problem in Z3
        """
        self.form_crew_dispatch(Route_Example)
        self.form_network_operation(ppc)
        self.define_objective(ppc)




if __name__=="__main__":

    # import testcase data
    import Fun_IEEETestCase as case
    import Fun_Crew_Dispatch as c_d
    ppc = case.case33_noDG_tieline()
    vrp = c_d.crew_dispatch_determ()

    mp1 = OutageManageZ3()
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