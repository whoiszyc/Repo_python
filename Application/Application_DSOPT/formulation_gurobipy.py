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
from formulation_general import *


class OutageManageGurobi(OutageManage):
    """
    define outage management using gurobipy interface
    """


    def define_problem_object(self):
        """
        Define optimization problem objects
        :return:
        """
        #
        self.model = gb.Model()



    def form_crew_dispatch(self, Route_Example=None):
        """
        formulate crew dispatch problem
        :param Route_Example:
        :return:
        """

        vrp = self.vrp

        # # define variables
        self.AT = self.model.addVars(self.iter_crew, self.ordered_vertex, lb=0, ub=100, vtype=gb.GRB.CONTINUOUS, name='AT')
        self.f = self.model.addVars(self.ordered_vertex, self.iter_time, vtype=gb.GRB.BINARY, name='f')
        self.x = self.model.addVars(self.iter_crew, self.ordered_vertex, self.ordered_vertex, vtype=gb.GRB.BINARY, name='x')
        self.y = self.model.addVars(self.iter_crew, self.ordered_vertex, vtype=gb.GRB.BINARY, name='y')
        self.z = self.model.addVars(self.ordered_vertex, self.iter_time, vtype=gb.GRB.BINARY, name='z')

        # # crew visited constraints
        # # 30 pathflow conservation constraint ensures that a crew arriving at a damaged component leaves it after finishing the repair
        for c in self.iter_crew:
            for m in set(self.iter_vertex) - {'0', 'd'}:
                self.model.addConstr(sum(self.x[c, m, n] for n in set(self.iter_vertex) - {m}) -
                                     sum(self.x[c, n, m] for n in set(self.iter_vertex) - {m}) == 0)

        # # 31 ensures that the crews start from depots
        for c in self.iter_crew:
            self.model.addConstr(sum(self.x[c, '0', n] for n in set(self.iter_vertex) - {'0'}) - sum(
                self.x[c, n, '0'] for n in set(self.iter_vertex) - {'0'}) == 1)

        # 32 indicates that all crews return to the depots
        self.model.addConstr(sum(sum(self.x[c, n, 'd'] for n in set(self.iter_vertex) - {'d'}) for c in self.iter_crew) == 1)

        # 33 indicates that a damaged component is fixed by only one crew to ensure that no crew visits a repaired component
        for m in set(self.iter_vertex) - {'0', 'd'}:
            self.model.addConstr(sum(self.y[c, m] for c in self.iter_crew) == 1)

        # 34 couples the binary variables x and y, i.e., if a crew c takes the traveling path xmm, then ym;c = 1
        for c in self.iter_crew:
            for m in set(self.iter_vertex) - {'d'}:
                self.model.addConstr(self.y[c, m] == sum(self.x[c, m, n] for n in set(self.iter_vertex) - {'0', m}))

        # # enforcement
        for c in self.iter_crew:
            for i in self.iter_vertex:
                for j in self.iter_vertex:
                    if i == j:
                        self.model.addConstr(self.x[c, i, j] == 0)


        # # crew repairing constraints
        # 39 represents the arrival time
        for c in self.iter_crew:
            for n in self.iter_vertex:
                for m in set(self.iter_vertex) - {'d'}:
                    self.model.addConstr(
                        self.AT[c, m] + vrp['repair'][c][m] + vrp['travel'][m][n] <= self.AT[c, n]
                        + (1 - self.x[c, m, n]) * self.BigM)

        # # 40-42: the time a damaged component is repaired
        for m in self.iter_vertex:
            self.model.addConstr(sum(self.f[m, t] for t in self.iter_time) == 1)
            self.model.addConstr(sum(t * self.f[m, t] for t in self.iter_time) >=
                                 sum(self.AT[c, m] + vrp['repair'][c][m] * self.y[c, m] for c in self.iter_crew))
            self.model.addConstr(sum(t * self.f[m, t] for t in self.iter_time) <=
                                 sum(self.AT[c, m]+ vrp['repair'][c][m] * self.y[c, m] for c in self.iter_crew) + 1 - self.epsilon)

        # # 43 If the damaged component is not repaired by a crew c then the arrival time and repair time
        # # for this crew should not affect constraints (41) and (42), which is realized by using constraint (43)
        for c in self.iter_crew:
            for m in self.iter_vertex:
                self.model.addConstr(self.AT[c, m] <= self.y[c, m] * self.BigM)

        # # 44 indicates that the restored component becomes available in all subsequent time periods
        for t in self.iter_time:
            for m in self.iter_vertex:
                self.model.addConstr(self.z[m, t] <= sum(self.f[m, tau] for tau in np.arange(0, t)))



    def form_coupling(self):
        """
        Coupling: line availability and repairing indicator
        :return:
        """
        for t in self.iter_time:
            for m in self.line_damaged:
                self.model.addConstr(self.ul[m, t] <= self.z[m, t])



    def form_network_operation(self):
        """
        Define variables for convex optimization problems
        :param ppc:
        :return:
        """
        ppc = self.ppc

        self.p = self.model.addVars(self.iter_gen, self.iter_time, lb=-5, ub=5, vtype=gb.GRB.CONTINUOUS, name='p')
        self.q = self.model.addVars(self.iter_gen, self.iter_time, lb=-5, ub=5, vtype=gb.GRB.CONTINUOUS, name='q')
        self.P = self.model.addVars(self.iter_line, self.iter_time, lb=-5, ub=5, vtype=gb.GRB.CONTINUOUS, name='P')
        self.Q = self.model.addVars(self.iter_line, self.iter_time, lb=-5, ub=5, vtype=gb.GRB.CONTINUOUS, name='Q')
        self.V = self.model.addVars(self.iter_bus, self.iter_time, lb=0, ub=1.5, vtype=gb.GRB.CONTINUOUS, name='V')
        self.rho = self.model.addVars(self.iter_bus, self.iter_time, vtype=gb.GRB.BINARY, name='rho')
        self.ul = self.model.addVars(self.iter_line, self.iter_time, vtype=gb.GRB.BINARY, name='ul')
        self.beta = self.model.addVars(self.iter_bus, self.iter_bus, self.iter_time, vtype=gb.GRB.BINARY, name='beta')

        # ------------------ Notes -----------------
        # It is important to define the bound in gurobi-py
        # It seems the default lower bound of gurobi is 0, that is, it assumes positive values
        # So for negative values, it is important to define the lower bound, as in this case, the line flow.
        # Bound for variables in "addVars" together with vtype define types like NonNegativeReals

        # # Line flow limits
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1

                self.model.addConstr(self.P[i, t] <= self.ul[i, t] * ppc['line'][id, 5], name='line_upper_P_' + i)
                self.model.addConstr(self.P[i, t] >= -self.ul[i, t] * ppc['line'][id, 5], name='line_lower_P_' + i)
                self.model.addConstr(self.Q[i, t] <= self.ul[i, t] * ppc['line'][id, 6], name='line_upper_Q_' + i)
                self.model.addConstr(self.Q[i, t] >= -self.ul[i, t] * ppc['line'][id, 6], name='line_lower_Q_' + i)

        # # Voltage limits
        for t in self.iter_time:
            for i in self.iter_bus:
                if i == 'bus_1':
                    self.model.addConstr(self.V[i, t] == self.Voltage_Substation, name='Voltage_substation')
                else:
                    self.model.addConstr(self.V[i, t] >= 1 - self.Voltage_Variation, name='Voltage_upper_' + i)
                    self.model.addConstr(self.V[i, t] <= 1 + self.Voltage_Variation, name='Voltage_lower_' + i)

        # # Power balance at bus i
        for t in self.iter_time:
            for i in self.iter_bus:

                P_out = 0
                Q_out = 0
                P_in = 0
                Q_in = 0

                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1

                # get power flow variables flowing out from this bus
                iter_out = np.arange(0, len(self.bus_line[i]["line_from_this_bus"]))

                if iter_out.shape[0] == 0:
                    P_out = P_out
                    Q_out = Q_out
                else:
                    for k in iter_out:
                        P_out = P_out + self.P['line_{}'.format(self.bus_line[i]["line_from_this_bus"][k]), t]
                        Q_out = Q_out + self.Q['line_{}'.format(self.bus_line[i]["line_from_this_bus"][k]), t]

                # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[i]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                    Q_in = Q_in
                else:
                    for k in iter_in:
                        P_in = P_in + self.P['line_{}'.format(self.bus_line[i]["line_to_this_bus"][k]), t]
                        Q_in = Q_in + self.Q['line_{}'.format(self.bus_line[i]["line_to_this_bus"][k]), t]

                # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][id, 1] == 1:
                    self.model.addConstr(
                        P_in + self.p['gen_{}'.format(self.bus_gen[i][0]), t] == P_out + ppc['bus'][id, 4] * self.rho[i, t],
                        name='Power_balance_P_' + i)
                    self.model.addConstr(
                        Q_in + self.q['gen_{}'.format(self.bus_gen[i][0]), t] == Q_out + ppc['bus'][id, 5] * self.rho[i, t],
                        name='Power_balance_Q_' + i)

                # if this is a bus with only load
                else:
                    self.model.addConstr(P_in == P_out + ppc['bus'][id, 4] * self.rho[i, t], name='Power_balance_P_' + i)
                    self.model.addConstr(Q_in == Q_out + ppc['bus'][id, 5] * self.rho[i, t], name='Power_balance_Q_' + i)

        # # Voltage drop along line k
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1

                # for this line, get the bus index
                f_bus = int(ppc['line'][id, 1])
                t_bus = int(ppc['line'][id, 2])

                self.model.addConstr(self.V['bus_{}'.format(t_bus), t] - self.V['bus_{}'.format(f_bus), t]
                                     + (ppc['line'][id, 3] * self.P[i, t] + ppc['line'][id, 4] * self.Q[i, t]) / self.Voltage_Substation >= -(1 - self.ul[i, t]) * self.BigM)
                self.model.addConstr(self.V['bus_{}'.format(t_bus), t] - self.V['bus_{}'.format(f_bus), t]
                                     + (ppc['line'][id, 3] * self.P[i, t] + ppc['line'][id, 4] * self.Q[i, t]) / self.Voltage_Substation <= (1 - self.ul[i, t]) * self.BigM)

        # # substation bus does not have parent bus
        for t in self.iter_time:
            self.model.addConstr(self.beta['bus_2', 'bus_1', t] == 0)  # hard coded

        # # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in self.iter_bus:
                # model_flow.addConstr(beta.sum('*', j, t) <= 1)
                self.model.addConstr(sum(self.beta[i, j, t] for i in self.iter_bus) <= 1)

        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1

                # for this line, get the bus index
                f_bus = int(ppc['line'][id, 1])
                t_bus = int(ppc['line'][id, 2])

                self.model.addConstr(self.beta['bus_{}'.format(f_bus), 'bus_{}'.format(t_bus), t]
                                     + self.beta['bus_{}'.format(t_bus), 'bus_{}'.format(f_bus), t] == self.ul[i, t])

        # # Static lines cannot change
        for t in self.iter_time:
            for i in self.line_static:
                self.model.addConstr(self.ul[i, t] == 1)

        # # The damaged line will remain closed once it is closed
        for t in range(1, self.Total_Time):
            for i in self.line_damaged:
                self.model.addConstr(self.ul[i, t] >= self.ul[i, t - 1])

        # # Load that has been picked up cannot be shed again
        for t in range(1, self.Total_Time):  # use set type as iterators are easy to implement N\{m} type of constraints
            for i in self.iter_bus:
                self.model.addConstr(self.rho[i, t] >= self.rho[i, t - 1])



    def form_network_operation_int(self):
        """
        Define a simplified network optimization using pure integer program
        :param ppc:
        :return:
        """
        ppc = self.ppc

        self.p = self.model.addVars(self.iter_gen, self.iter_time, lb=-500, ub=500, vtype=gb.GRB.INTEGER, name='p')
        self.P = self.model.addVars(self.iter_line, self.iter_time, lb=-500, ub=500, vtype=gb.GRB.INTEGER, name='P')
        self.rho = self.model.addVars(self.iter_bus, self.iter_time, vtype=gb.GRB.BINARY, name='rho')
        self.ul = self.model.addVars(self.iter_line, self.iter_time, vtype=gb.GRB.BINARY, name='ul')
        self.beta = self.model.addVars(self.iter_bus, self.iter_bus, self.iter_time, vtype=gb.GRB.BINARY, name='beta')

        # ------------------ Notes -----------------
        # It is important to define the bound in gurobi-py
        # It seems the default lower bound of gurobi is 0, that is, it assumes positive values
        # So for negative values, it is important to define the lower bound, as in this case, the line flow.
        # Bound for variables in "addVars" together with vtype define types like NonNegativeReals

        # # Line flow limits
        for t in self.iter_time:
            for i in self.iter_line:
                id = int(i[i.find('_') + 1:]) - 1 # get the matrix index from the component name
                self.model.addConstr(self.P[i, t] <= self.ul[i, t] * 1000)
                self.model.addConstr(self.P[i, t] >= -self.ul[i, t] * 1000)

        # # Power balance at bus i
        for t in self.iter_time:
            for i in self.iter_bus:
                P_out = 0
                P_in = 0

                # # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1

                # # get power flow variables flowing out from this bus
                iter_out = np.arange(0, len(self.bus_line[i]["line_from_this_bus"]))

                if iter_out.shape[0] == 0:
                    P_out = P_out
                else:
                    for k in iter_out:
                        P_out = P_out + self.P['line_{}'.format(self.bus_line[i]["line_from_this_bus"][k]), t]

                # # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[i]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                else:
                    for k in iter_in:
                        P_in = P_in + self.P['line_{}'.format(self.bus_line[i]["line_to_this_bus"][k]), t]

                # # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][id, 1] == 1:
                    self.model.addConstr(
                        P_in + self.p['gen_{}'.format(self.bus_gen[i][0]), t] == P_out + self.rho[i, t],)

                # # if this is a bus with only load
                else:
                    self.model.addConstr(P_in == P_out + self.rho[i, t])


        # # substation bus does not have parent bus
        for t in self.iter_time:
            self.model.addConstr(self.beta['bus_2', 'bus_1', t] == 0)  # hard coded

        # # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in self.iter_bus:
                # model_flow.addConstr(beta.sum('*', j, t) <= 1)
                self.model.addConstr(sum(self.beta[i, j, t] for i in self.iter_bus) <= 1)

        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1

                # for this line, get the bus index
                f_bus = int(ppc['line'][id, 1])
                t_bus = int(ppc['line'][id, 2])

                self.model.addConstr(self.beta['bus_{}'.format(f_bus), 'bus_{}'.format(t_bus), t]
                                     + self.beta['bus_{}'.format(t_bus), 'bus_{}'.format(f_bus), t] == self.ul[i, t])

        # # Static lines cannot change
        for t in self.iter_time:
            for i in self.line_static:
                self.model.addConstr(self.ul[i, t] == 1)

        # # The damaged line will remain closed once it is closed
        for t in range(1, self.Total_Time):
            for i in self.line_damaged:
                self.model.addConstr(self.ul[i, t] >= self.ul[i, t - 1])

        # # Load that has been picked up cannot be shed again
        for t in range(1, self.Total_Time):  # use set type as iterators are easy to implement N\{m} type of constraints
            for i in self.iter_bus:
                self.model.addConstr(self.rho[i, t] >= self.rho[i, t - 1])



    def define_objective(self):
        """
        maximize the load pickup
        :return:
        """
        self.ObjVar = self.model.addVar(1, vtype=gb.GRB.CONTINUOUS)

        obj = 0
        for t in self.iter_time:
            for i in self.iter_bus:
                id = int(i[i.find('_') + 1:]) - 1 # get the matrix index from the component name
                obj = obj + self.rho[i, t] * self.ppc['bus'][id, 4] * self.BasePower

        self.model.addConstr(self.ObjVar == obj)

        self.model.setObjective(self.ObjVar, gb.GRB.MAXIMIZE)



    def add_variable_list(self, VariableName, IterList):
        """
        add the variable list to the model object for callback use
        :param VariableName: variable name in string format
        :param IterList: a nested list with all iterator that defined the variables
        """

        # #  get variable dimension
        VarDim = len(IterList)

        # # define an empty variable list
        VarList = []

        # # get string type attribute name
        variable = operator.attrgetter(VariableName)(self)

        for k in range(VarDim):
            for i in IterList[k]:
                for j in IterList[k]:
                    VarList.append(variable[i, j])


        # # add the variable list as a model attribute
        # # setattr(object, attribute name, value), here the value is an empty list
        setattr(self.model, '_' + VariableName, VarList)



    def get_solution_2d(self, VariableName, NameKey, ListIndex, SolDict=None):
        """
        get solution and store into a one name key structured dictionary
        :param VariableName: variable name in string format
        :param NameKey: desired key set in list or range format that you would like to retrieve
        :param ListIndex: desired index range in list format that you would like to retrieve
        :param SolDict: dictionary object with plot methods
        :return: SolDict
        """
        if SolDict == None:
            SolDict = SolutionDict()
        else:
            pass

        # # get string type attribute name
        variable = operator.attrgetter(VariableName)(self)

        for i in NameKey:
            SolDict[i] = []
            for j in ListIndex:
                SolDict[i].append(variable[i, j].x)

        return SolDict



    def get_solution_3d(self, VariableName, NameKey, ListIndex1, ListIndex2, SolDict=None):
        """
        get solution and store into a one name key structured dictionary
        :param VariableName: variable name in string format
        :param NameKey: desired key set in list or range format that you would like to retrieve
        :param ListIndex: desired index range in list format that you would like to retrieve
        :param SolDict: dictionary object with plot methods
        :return: SolDict
        """
        if SolDict == None:
            SolDict = SolutionDict()
        else:
            pass

        # # get string type attribute name
        variable = operator.attrgetter(VariableName)(self)

        for i in NameKey:
            SolDict[i] = []
            for j in ListIndex1:
                SolDict[j] = []
                for k in ListIndex2:
                    SolDict[i][j].append(variable[i, j, k].x)

        return SolDict



    def get_solution_route(self):
        """
        get crew dispatch route and plot
        """
        vrp = self.vrp

        SolDict = OrderedDict()
        for c in self.iter_crew:
            SolDict[c] = OrderedDict()
            SolDict[c]['x'] = OrderedDict()
            SolDict[c]['route'] = []
            for i in self.ordered_vertex:
                for j in self.ordered_vertex:
                    SolDict[c]['x'][i, j] = self.x[c, i, j].x
                    if self.x[c, i, j].x == 1:
                        SolDict[c]['route'].append((i, j))

        # # hard coded for single crew plot
        plt.figure(figsize=(7, 5))
        for i in SolDict[0]['route']:
            vrp['graph'].add_edge(i[0], i[1])
        nx.draw(vrp['graph'], vrp['fault']['location'], with_labels=True)
        plt.show()

        return SolDict



    def form_cop(self):
        """
        formulate the co-optimization problem in MIP
        :return:
        """
        self.define_problem_object()
        self.form_crew_dispatch()
        self.form_network_operation()
        self.form_coupling()
        self.define_objective()



    def form_mp(self):
        """
        formulate the master problem in MIP
        :return:
        """
        self.define_problem_object()
        self.form_crew_dispatch()
        self.form_network_operation_int()
        self.form_coupling()
        self.define_objective()



    def form_sp(self):
        """
        formulate the subproblem in MIP
        :return:
        """
        self.define_problem_object()
        self.form_network_operation()
        self.define_objective()



    def test_crew_dispatch(self):
        """
        test the crew dispatch and repairing
        :return:
        """
        # # define model
        self.define_problem_object()
        self.form_crew_dispatch()

        # # define objective
        self.obj = self.AT[0, 'd']
        self.obj = self.obj - sum(sum(self.z[m, t] for m in self.iter_vertex) for t in self.iter_time)

        # # set objective to be "minimize"
        self.model.setObjective(self.obj, gb.GRB.MINIMIZE)



    def test_network_operation(self):
        """
        fix damaged line to test the reconfiguration
        """
        # # define model
        self.define_problem_object()
        self.form_network_operation()
        self.define_objective()

        for t in self.iter_time:
            for m in self.line_damaged:
                self.model.addConstr(self.ul[m, t] == 0)

        # for t in self.iter_time:
        #     for m in {'line_33','line_36','line_37'}:
        #         self.model.addConstr(self.ul[m, t] == 0)
        #
        # for t in self.iter_time:
        #     self.model.addConstr(self.ul['line_34', t] == 0)
        #
        # for t in self.iter_time:
        #     self.model.addConstr(self.ul['line_35', t] == 0)




    def test_network_operation_int(self):
        """
        fix damaged line to test the reconfiguration
        """
        # # define model
        self.define_problem_object()
        self.form_network_operation_int()
        self.define_objective()

        for t in self.iter_time:
            for m in self.line_damaged:
                self.model.addConstr(self.ul[m, t] == 0)

        # for t in self.iter_time:
        #     for m in {'line_33','line_36','line_37'}:
        #         self.model.addConstr(self.ul[m, t] == 0)
        #
        # for t in self.iter_time:
        #     self.model.addConstr(self.ul['line_34', t] == 0)
        #
        # for t in self.iter_time:
        #     self.model.addConstr(self.ul['line_35', t] == 0)





if __name__ == "__main__":

    # import testcase data
    import Fun_IEEETestCase as case
    import Fun_Crew_Dispatch as c_d
    ppc = case.case33_noDG_tieline()
    vrp = c_d.crew_dispatch_determ()

    # # # ----------------------------------------
    # # #           co-optimization model
    # # # ----------------------------------------
    # cop = fg.OutageManageGurobi()
    # cop.data_preparation(ppc, vrp)
    # cop.form_cop()
    #
    # # # set solver parameters
    # cop.model.Params.MIPGap = 0.02
    #
    # # # solve problem
    # cop.model.update()
    # cop.model.optimize()
    #
    # # # # get objective function value of MIP
    # ObjVal_cop = cop.model.objVal
    # print('objective is {}'.format(ObjVal_cop))
    # cop.get_solution_route()
    # load_status = cop.get_solution_2d('rho', cop.iter_bus, cop.iter_time)
    # load_status.plot_bin_2d()
    # line_status = cop.get_solution_2d('ul', cop.iter_line, cop.iter_time)
    # line_status.plot_bin_2d()
    # line_flow = cop.get_solution_2d('P', ['line_1'], cop.iter_time)
    # line_flow.plot_step_2d()
    # repair_status = cop.get_solution_2d('z', cop.ordered_vertex, cop.iter_time)
    # repair_status.plot_bin_2d()

    # # # # ----------------------------------------
    # # # #           test master problem
    # # # # ----------------------------------------
    mp = OutageManageGurobi()
    mp.data_preparation(ppc, vrp)
    mp.form_mp()

    # # set solver parameters
    # mp.model.Params.MIPGap = 0.02

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
    # # cop.model.Params.MIPGap = 0.02
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
    # # line_flow = cop.get_solution_2d('P', ['line_1'], cop.iter_time)
    # # line_flow.plot_step_2d()

    # # # ----------------------------------------
    # # #           test network operation
    # # # ----------------------------------------
    # mp = fg.OutageManageGurobi()
    # mp.data_preparation(ppc, vrp)
    # mp.test_network_operation()
    #
    # # # set solver parameters
    # # cop.model.Params.MIPGap = 0.02
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
    # cop.model.Params.MIPGap = 0.02

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