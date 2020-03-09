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


import pyomo.environ as pm
from formulation_general import *



class OutageManagePyomo(OutageManage):
    """
    distribution system outage management formulation using Pyomo
    """

    def define_problem_object(self):
        self.model = pm.ConcreteModel()



    def form_crew_dispatch(self, Route_Example=None):

        vrp = self.vrp

        ## repairing constraint
        self.model.AT = pm.Var(self.iter_crew, self.iter_vertex, within=pm.NonNegativeReals)
        self.model.f = pm.Var(self.iter_vertex, self.iter_time, within=pm.Binary)
        self.model.x = pm.Var(self.iter_crew, self.iter_vertex, self.iter_vertex, within=pm.Binary)
        self.model.y = pm.Var(self.iter_crew, self.iter_vertex, within=pm.Binary)
        self.model.z = pm.Var(self.iter_vertex, self.iter_time, within=pm.Binary)  # line availability indicator

        ## crew visited constraint
        self.model.vehicle_routing = pm.ConstraintList()
        for c in self.iter_crew:
            # 30
            for m in set(self.iter_vertex) - {'0', 'd'}:
                self.model.vehicle_routing.add(sum(self.model.x[c, m, n] for n in set(self.iter_vertex) - {m}) - sum(
                    self.model.x[c, n, m] for n in set(self.iter_vertex) - {m}) == 0)
            # 31
            self.model.vehicle_routing.add(sum(self.model.x[c, '0', n] for n in set(self.iter_vertex) - {'0'}) - sum(
                self.model.x[c, n, '0'] for n in set(self.iter_vertex) - {'0'}) == 1)

        # 32
        self.model.vehicle_routing.add(sum(sum(self.model.x[c, n, 'd'] for n in set(self.iter_vertex) - {'d'}) for c in self.iter_crew) == 1)

        # 33
        for m in set(self.iter_vertex) - {'0', 'd'}:
            self.model.vehicle_routing.add(sum(self.model.y[c, m] for c in self.iter_crew) == 1)

        # 34
        for c in self.iter_crew:
            for m in set(self.iter_vertex) - {'d'}:
                self.model.vehicle_routing.add(self.model.y[c, m] == sum(self.model.x[c, m, n] for n in set(self.iter_vertex) - {'0', m}))

        ## enforcement
        for c in self.iter_crew:
            for i in self.iter_vertex:
                for j in self.iter_vertex:
                    if i == j:
                        self.model.vehicle_routing.add(self.model.x[c, i, j] == 0)
        # model.vehicle_routing.add(model.y[0, 'd'] == 1)

        ## define constraint of repairing damaged components
        self.model.repair = pm.ConstraintList()

        # 39
        for c in self.iter_crew:
            for n in self.iter_vertex:
                for m in set(self.iter_vertex) - {'d'}:
                    self.model.repair.add(self.model.AT[c, m] + vrp['repair'][c][m] + vrp['travel'][m][n] <= self.model.AT[c, n] + (
                                1 - self.model.x[c, m, n]) * self.BigM)

        # # 40-42: the time a damaged component is repaired
        for m in self.iter_vertex:
            self.model.repair.add(sum(self.model.f[m, t] for t in self.iter_time) == 1)
            self.model.repair.add(sum(t * self.model.f[m, t] for t in self.iter_time) >= sum(
                self.model.AT[c, m] + vrp['repair'][c][m] * self.model.y[c, m] for c in self.iter_crew))
            self.model.repair.add(sum(t * self.model.f[m, t] for t in self.iter_time) <= sum(
                self.model.AT[c, m] + vrp['repair'][c][m] * self.model.y[c, m] for c in self.iter_crew) + 1 - self.epsilon)

        # # 43
        for c in self.iter_crew:
            for m in self.iter_vertex:
                self.model.repair.add(self.model.AT[c, m] <= self.model.y[c, m] * self.BigM)

        # # 44
        for t in self.iter_time:
            for m in self.iter_vertex:
                self.model.repair.add(self.model.z[m, t] <= sum(self.model.f[m, tau] for tau in np.arange(0, t)))



    def form_network_operation(self):

        ppc = self.ppc

        ## network operation variables
        self.model.p = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.q = pm.Var(self.iter_gen, self.iter_time, within=pm.Reals)
        self.model.P = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.Q = pm.Var(self.iter_line, self.iter_time, within=pm.Reals)
        self.model.V = pm.Var(self.iter_bus, self.iter_time, within=pm.NonNegativeReals)
        self.model.rho = pm.Var(self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.ug = pm.Var(self.iter_gen, self.iter_time, within=pm.Binary)
        self.model.ul = pm.Var(self.iter_line, self.iter_time, within=pm.Binary)
        self.model.beta = pm.Var(self.iter_bus, self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.nb = pm.Var(self.iter_time, within=pm.NonNegativeReals)

        ## distflow constraint: power balance at bus i
        self.model.con_distflow_bus = pm.ConstraintList()  # In pyomo formulation, use ConstraintList() is a generic way to add constraints
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
                        P_out = P_out + self.model.P['line_{}'.format(self.bus_line[i]["line_from_this_bus"][k]), t]
                        Q_out = Q_out + self.model.Q['line_{}'.format(self.bus_line[i]["line_from_this_bus"][k]), t]

                # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[i]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                    Q_in = Q_in
                else:
                    for k in iter_in:
                        P_in = P_in + self.model.P['line_{}'.format(self.bus_line[i]["line_to_this_bus"][k]), t]
                        Q_in = Q_in + self.model.Q['line_{}'.format(self.bus_line[i]["line_to_this_bus"][k]), t]

                # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][id, 1] == 1:
                    self.model.con_distflow_bus.add(
                        P_in + self.model.p['gen_{}'.format(self.bus_gen[i][0]), t] == P_out + ppc['bus'][id, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(
                        Q_in + self.model.q['gen_{}'.format(self.bus_gen[i][0]), t] == Q_out + ppc['bus'][id, 5] * self.model.rho[i, t])

                # if this is a bus with only load
                else:
                    self.model.con_distflow_bus.add(P_in == P_out + ppc['bus'][id, 4] * self.model.rho[i, t])
                    self.model.con_distflow_bus.add(Q_in == Q_out + ppc['bus'][id, 5] * self.model.rho[i, t])

        ## distflow constraint: voltage drop along line k
        self.model.con_distflow_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1

                # for this line, get the bus index
                f_bus = int(ppc['line'][id, 1])
                t_bus = int(ppc['line'][id, 2])

                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][id, 3] * self.model.P[i, t] + ppc['line'][id, 4] * self.model.Q[i, t]) / self.Voltage_Substation <= (1 - self.model.ul[i, t]) * self.BigM)
                self.model.con_distflow_line.add(
                    self.model.V['bus_{}'.format(t_bus), t] - self.model.V['bus_{}'.format(f_bus), t]
                    + (ppc['line'][id, 3] * self.model.P[i, t] + ppc['line'][id, 4] * self.model.Q[i, t]) / self.Voltage_Substation >= -(1 - self.model.ul[i, t]) * self.BigM)

        ## operation limits
        # self.model.con_lim_gen = pm.ConstraintList()
        # for t in self.iter_time:
        #     for i in self.iter_gen:
        #         # get the matrix index from the component name
        #         id = int(i[i.find('_') + 1:]) - 1
        #         self.model.con_lim_gen.add(self.model.p[i, t] >= ppc['gen'][id, 2] * self.model.ug[i, t])
        #         self.model.con_lim_gen.add(self.model.p[i, t] <= ppc['gen'][id, 3] * self.model.ug[i, t])
        #         self.model.con_lim_gen.add(self.model.q[i, t] >= ppc['gen'][id, 4] * self.model.ug[i, t])
        #         self.model.con_lim_gen.add(self.model.q[i, t] <= ppc['gen'][id, 5] * self.model.ug[i, t])

        self.model.con_lim_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1
                self.model.con_lim_line.add(self.model.P[i, t] <= self.model.ul[i, t] * ppc['line'][id, 5])  #ppc['line'][id, 5]
                self.model.con_lim_line.add(self.model.P[i, t] >= -self.model.ul[i, t] * ppc['line'][id, 5])
                self.model.con_lim_line.add(self.model.Q[i, t] <= self.model.ul[i, t] * ppc['line'][id, 6]) # ppc['line'][id, 6]
                self.model.con_lim_line.add(self.model.Q[i, t] >= -self.model.ul[i, t] * ppc['line'][id, 6])

        self.model.con_lim_voltage = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_bus:
                if i == 'bus_1':
                    self.model.con_lim_voltage.add(self.model.V[i, t] == self.Voltage_Substation)
                else:
                    self.model.con_lim_voltage.add(self.model.V[i, t] >= 1 - self.Voltage_Variation)
                    self.model.con_lim_voltage.add(self.model.V[i, t] <= 1 + self.Voltage_Variation)


        ## tree topology constraints
        self.model.con_radiality = pm.ConstraintList()

        # substation bus does not have parent bus
        for t in self.iter_time:
            self.model.con_radiality.add(self.model.beta['bus_2', 'bus_1', t] == 0)  # hard coded

        # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in self.iter_bus:
                self.model.con_radiality.add(sum(self.model.beta[i, j, t] for i in self.iter_bus) <= 1)

        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1
                # for this line, get the bus index
                f_bus = int(ppc['line'][id, 1])
                t_bus = int(ppc['line'][id, 2])
                self.model.con_radiality.add(self.model.beta['bus_{}'.format(f_bus), 'bus_{}'.format(t_bus), t]
                                        + self.model.beta['bus_{}'.format(t_bus), 'bus_{}'.format(f_bus), t] == self.model.ul[i, t])

        ## line status constraints
        self.model.status_line = pm.ConstraintList()

        ## static line cannot change
        for t in self.iter_time:
            for i in self.line_static:
                self.model.status_line.add(self.model.ul[i, t] == 1)

        ## Repaired and closed damaged line cannot open again
        for t in range(1, self.Total_Time):
            for i in self.line_damaged:
                self.model.status_line.add(self.model.ul[i, t] >= self.model.ul[i, t - 1])

        ## load status constraints
        self.model.status_load = pm.ConstraintList()
        for t in range(1, self.Total_Time):
            for i in self.iter_bus:
                self.model.status_load.add(self.model.rho[i, t] >= self.model.rho[i, t - 1])



    def form_network_operation_int(self):
        ppc = self.ppc

        ## network operation variables
        self.model.p = pm.Var(self.iter_gen, self.iter_time, within=pm.Integers)
        self.model.P = pm.Var(self.iter_line, self.iter_time, within=pm.Integers)
        self.model.rho = pm.Var(self.iter_bus, self.iter_time, within=pm.Binary)
        self.model.ul = pm.Var(self.iter_line, self.iter_time, within=pm.Binary)
        self.model.beta = pm.Var(self.iter_bus, self.iter_bus, self.iter_time, within=pm.Binary)

        ## distflow constraint: power balance at bus i
        self.model.con_distflow_bus = pm.ConstraintList()  # In pyomo formulation, use ConstraintList() is a generic way to add constraints
        for t in self.iter_time:
            for i in self.iter_bus:
                P_out = 0
                P_in = 0
                id = int(i[i.find('_') + 1:]) - 1  # get the matrix index from the component name
                # get power flow variables flowing out from this bus
                iter_out = np.arange(0, len(self.bus_line[i]["line_from_this_bus"]))
                if iter_out.shape[0] == 0:
                    P_out = P_out
                else:
                    for k in iter_out:
                        P_out = P_out + self.model.P['line_{}'.format(self.bus_line[i]["line_from_this_bus"][k]), t]
                # get power flow variables flowing into from this bus
                iter_in = np.arange(0, len(self.bus_line[i]["line_to_this_bus"]))

                if iter_in.shape[0] == 0:
                    P_in = P_in
                else:
                    for k in iter_in:
                        P_in = P_in + self.model.P['line_{}'.format(self.bus_line[i]["line_to_this_bus"][k]), t]

                # if this is a bus with generation capability and load, the power flow balance is
                if ppc['bus'][id, 1] == 1:
                    self.model.con_distflow_bus.add(
                        P_in + self.model.p['gen_{}'.format(self.bus_gen[i][0]), t] == P_out + self.model.rho[i, t])
                # if this is a bus with only load
                else:
                    self.model.con_distflow_bus.add(P_in == P_out + self.model.rho[i, t])

        self.model.con_lim_line = pm.ConstraintList()
        for t in self.iter_time:
            for i in self.iter_line:
                self.model.con_lim_line.add(self.model.P[i, t] <= self.model.ul[i, t] * 100)  # ppc['line'][id, 5]
                self.model.con_lim_line.add(self.model.P[i, t] >= -self.model.ul[i, t] * 100)

        ## tree topology constraints
        self.model.con_radiality = pm.ConstraintList()

        # substation bus does not have parent bus
        for t in self.iter_time:
            self.model.con_radiality.add(self.model.beta['bus_2', 'bus_1', t] == 0)  # hard coded

        # each bus will only at least have one parent bus
        for t in self.iter_time:
            for j in self.iter_bus:
                self.model.con_radiality.add(sum(self.model.beta[i, j, t] for i in self.iter_bus) <= 1)

        for t in self.iter_time:
            for i in self.iter_line:
                # get the matrix index from the component name
                id = int(i[i.find('_') + 1:]) - 1
                # for this line, get the bus index
                f_bus = int(ppc['line'][id, 1])
                t_bus = int(ppc['line'][id, 2])
                self.model.con_radiality.add(self.model.beta['bus_{}'.format(f_bus), 'bus_{}'.format(t_bus), t]
                            + self.model.beta['bus_{}'.format(t_bus), 'bus_{}'.format(f_bus), t] == self.model.ul[i, t])

        ## line status constraints
        self.model.status_line = pm.ConstraintList()

        ## static line cannot change
        for t in self.iter_time:
            for i in self.line_static:
                self.model.status_line.add(self.model.ul[i, t] == 1)

        ## Repaired and closed damaged line cannot open again
        for t in range(1, self.Total_Time):
            for i in self.line_damaged:
                self.model.status_line.add(self.model.ul[i, t] >= self.model.ul[i, t - 1])

        ## load status constraints
        self.model.status_load = pm.ConstraintList()
        for t in range(1, self.Total_Time):
            for i in self.iter_bus:
                self.model.status_load.add(self.model.rho[i, t] >= self.model.rho[i, t - 1])



    def form_coupling(self):
        """
        Coupling: line availability and repairing indicator
        :return:
        """
        self.model.coupling = pm.ConstraintList()
        for t in self.iter_time:
            for m in self.line_damaged:
                self.model.coupling.add(self.model.ul[m, t] <= self.model.z[m, t])



    def define_objective(self):

        ppc = self.ppc

        def obj_restoration(model):
            obj = 0 # initialize the objective function
            ## load pickups
            for t in self.iter_time:
                for i in self.iter_bus:
                    id = int(i[i.find('_') + 1:]) - 1 # get the matrix index from the component name
                    obj = obj - model.rho[i, t] * ppc['bus'][id, 4]
            return obj

        self.model.obj = pm.Objective(rule=obj_restoration)



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

        ## get string type attribute name
        variable = operator.attrgetter(VariableName)(self.model)

        for i in NameKey:
            SolDict[i] = []
            for j in ListIndex:
                SolDict[i].append(variable[i, j].value)

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
                    SolDict[c]['x'][i, j] = self.model.x[c, i, j].value
                    if self.model.x[c, i, j].value == 1:
                        SolDict[c]['route'].append((i, j))

        ## hard coded for single crew plot
        plt.figure(figsize=(7, 5))
        for i in SolDict[0]['route']:
            vrp['graph'].add_edge(i[0], i[1])
        nx.draw(vrp['graph'], vrp['fault']['location'], with_labels=True)
        plt.show()



    def form_cop(self):
        """
        formulate co-optimization problem
        :return:
        """
        self.define_problem_object()
        self.form_crew_dispatch()
        self.form_network_operation()
        self.form_coupling()
        self.define_objective()

        # for t in self.iter_time:
        #     for m in {'line_33','line_34','line_35','line_36','line_37'}:
        #         self.model.coupling.add(self.model.ul[m, t] == 0)



    def form_mp(self):
        """
        define the master problem in MIP
        :return:
        """
        self.define_problem_object()
        self.form_crew_dispatch()
        self.form_network_operation_int()
        self.form_coupling()
        self.define_objective()



    def form_sp(self):
        """
        define the subproblem in MIP
        :return:
        """
        self.define_problem_object()
        self.form_network_operation()
        self.form_coupling()
        self.define_objective()



    def test_crew_dispatch(self):
        self.define_problem_object()
        self.form_crew_dispatch()

        def obj_restoration(model):
            # initialize the objective function
            obj = 0
            ## crew traveling time
            # obj = obj + sum(sum(sum(self.ordered_total_time[i, j] * model.x[c, i, j] for j in self.iter_vertex) for i in self.iter_vertex) for c in self.iter_crew)
            ## minimize the terminal time
            obj = obj + model.AT[0, 'd']
            ## repairing component
            obj = obj - sum(sum(model.z[m, t] for m in self.iter_vertex) for t in self.iter_time)

            return obj

        self.model.obj = pm.Objective(rule=obj_restoration)



    def test_network_operation(self):
        """
        fix damaged line to test the reconfiguration
        """
        self.define_problem_object()
        self.form_network_operation()
        self.define_objective()

        self.model.coupling = pm.ConstraintList()
        for t in self.iter_time:
            for m in self.line_damaged:
                self.model.coupling.add(self.model.ul[m, t] == 0)

        for t in self.iter_time:
            for m in {'line_33','line_36','line_37'}:
                self.model.coupling.add(self.model.ul[m, t] == 1)

        for t in self.iter_time:
            self.model.coupling.add(self.model.ul['line_34', t] == 0)

        for t in self.iter_time:
            self.model.coupling.add(self.model.ul['line_35', t] == 1)





if __name__=="__main__":

    # import testcase data
    import Fun_IEEETestCase as case
    import Fun_Crew_Dispatch as c_d
    ppc = case.case33_noDG_tieline()
    vrp = c_d.crew_dispatch_determ()

    # # test co-optimization
    # test_1 = fm.OutageManagePyomo()
    # test_1.data_preparation(ppc, vrp)
    # test_1.form_cop()
    # opt = pm.SolverFactory("cplex", executable = '/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
    # opt.options['mipgap'] = 0.02  # if gap=b, then it is (b*100) %
    # results = opt.solve(test_1.model, tee = True)
    # print(results['solver'][0]['Termination condition'])
    # test_1.get_solution_route()
    # load_status = test_1.get_solution_2d('rho', test_1.iter_bus, test_1.iter_time)
    # load_status.plot_bin_2d()
    # line_status = test_1.get_solution_2d('ul', test_1.iter_line, test_1.iter_time)
    # line_status.plot_bin_2d()
    # line_flow = test_1.get_solution_2d('P', ['line_1'], test_1.iter_time)
    # line_flow.plot_step_2d()

    # test master problem
    cop = OutageManagePyomo()
    cop.data_preparation(ppc, vrp)
    cop.form_mp()
    # opt = pm.SolverFactory("cplex", executable = '/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex') # solver_io="python" or "nl"
    # opt.options['mipgap'] = 0.02  # if gap=b, then it is (b*100) %
    opt = pm.SolverFactory("gurobi", solver_io="python")
    results = opt.solve(cop.model, tee = True)
    print(results['solver'][0]['Termination condition'])
    print('objective is {}'.format(-cop.model.obj.value()))
    cop.get_solution_route()
    # load_status = test_1.get_solution_2d('rho', test_1.iter_bus, test_1.iter_time)
    # load_status.plot_bin_2d()
    # line_status = test_1.get_solution_2d('ul', test_1.iter_line, test_1.iter_time)
    # line_status.plot_bin_2d()
    # line_flow = test_1.get_solution_2d('P', ['line_1'], test_1.iter_time)
    # line_flow.plot_step_2d()
    # repair_status = test_1.get_solution_2d('z', test_1.ordered_vertex, test_1.iter_time)
    # repair_status.plot_bin_2d()

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