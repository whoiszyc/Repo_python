# Import python package
import numpy as np
import sympy as sy
from pyomo.environ import *
from collections import OrderedDict
import operator
import sys


# Import from IASF itself
from .utils import SolutionDict
from. general import PowerSystemProblem


class SchedulingProblem(PowerSystemProblem):
    """
    Define power system scheduling problem class, including unit commitment and economic dispatch
    Define scheduling problem time series and related iterators
    Define necessary dependent input, that is, if ED is defined, UC result is necessary.
    Scheduling problem is coordinated in terms of minutes.
    """

    def __init__(self, time_start, time_end, time_step_minute, problem_type, dependency=None):

        # pass problem type
        self.problem_type = problem_type
        self.dependency = dependency

        # convert time string into minute data
        start_hour, start_minute = time_start.split(':')
        time_start_min = int(start_hour) * 60 + int(start_minute)
        end_hour, end_minute = time_end.split(':')
        end_time_min = int(end_hour) * 60 + int(end_minute)

        # generate time series data in minutes
        if problem_type == 'UC':
            if time_step_minute < 60:
                print('For unit commitment, the minimum time step in minute should be 60 s. Now minimum time step is enforced.')
                time_step_minute = 60
            self.time_series = list(range(time_start_min, end_time_min, time_step_minute))
        elif problem_type == 'ED':
            if time_step_minute < 10:
                print('For economic dispatch, the minimum time step in minute should be 10 s. Now minimum time step is enforced.')
                time_step_minute = 10
            self.time_series = list(range(time_start_min, end_time_min, time_step_minute))
        else:
            print('Function under construction')

        # generate time related iterator
        self.number_time = len(self.time_series)
        self.iter_time = range(0, self.number_time)
        self.iter_time_1 = range(1, self.number_time)

        ## Generator iterators for minimum up and down time for different generators
        ## Assume the hour index is: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ... N
        ## Minimum up time iterator: The index t of the formulation will always start at 1, end at N-1
        self.iter_hour_updown = range(1, self.number_time - 1)


    def get_system_reserve_data(self, RV):
        self.RV = RV


    def get_system_load_data(self, PL, QL=None):
        self.PL = PL

        if QL == None:
            """use data from ppc as the Q load"""
            total_Q = sum(self.ppc['bus'][i, self.idx.QD] for i in range(self.number_bus))
            self.QL = [total_Q] * self.number_time
        else:
            self.QL = QL


    def set_bus_load(self, type, ratio=None):
        """
        dispatch the load data based on:
        type=1: percentage in static network data
        type=2: a given desired percentage
        """

        self.PL_bus = OrderedDict()
        self.QL_bus = OrderedDict()

        if type == 1:
            total_P = sum(self.ppc['bus'][i, self.idx.PD] for i in range(self.number_bus))
            total_Q = sum(self.ppc['bus'][i, self.idx.QD] for i in range(self.number_bus))
            for i in self.iter_bus:
                self.PL_bus[i] = []
                self.QL_bus[i] = []
                PL_percentage = self.ppc['bus'][i, self.idx.PD] / total_P
                QL_percentage = self.ppc['bus'][i, self.idx.QD] / total_Q
                for t in self.iter_time:
                    self.PL_bus[i].append(PL_percentage * self.PL[t])
                    self.QL_bus[i].append(QL_percentage * self.QL[t])

        elif type == 2:
            print('This function has not been added')
        else:
            pass


    def convert_cost(self):
        """Convert the generator cost to a simplified version C=A*u + mc*delta_p"""
        p = sy.Symbol('p')
        cost_curve = OrderedDict()
        cost_curve_diff = OrderedDict()
        self.cost_fixed = OrderedDict()
        self.cost_marginal = OrderedDict()
        self.cost_curve_simple = OrderedDict()

        ## It is important to convert the data to float after using the symbolic module
        for i in self.iter_gen:
            ## get generator name
            cost_curve[i] = self.ppc["gencost"][i, 1]*p**2 + self.ppc["gencost"][i, 2]*p + self.ppc["gencost"][i, 3]
            cost_curve_diff[i] = sy.diff(cost_curve[i], p)
            self.cost_fixed[i] = float(cost_curve[i].subs(p, self.ppc["gen"][i, 1]))
            self.cost_marginal[i] = float(cost_curve_diff[i].subs(p,(self.ppc["gen"][i, 2] - self.ppc["gen"][i, 1])*0.5))
            self.cost_curve_simple[i] = self.cost_marginal[i]*(p - self.ppc["gen"][i, 1]) + self.cost_fixed[i]


    def define_pyomo_model(self, PyomoModelType):
        if PyomoModelType == 'Concrete':
            self.model = ConcreteModel()
        elif PyomoModelType == 'Abstract':
            self.model = ConcreteModel()
        else:
            print('No such type, please choose: Concrete or Abstract')


    def define_problem(self, PowerFlowModel = None):
        """common variables and constraints"""
        self.add_variable_generator()
        self.add_constraint_generator()
        self.add_constraint_system_reserve()

        if self.problem_type == 'UC' and PowerFlowModel == None:
            print('No power flow model is given. No power flow constraint is considered.')
            self.add_constraint_power_balance()

        elif self.problem_type == 'UC' and PowerFlowModel == 'DC':
            self.add_variable_network()
            self.make_ybus()
            self.add_constraint_dc_power_flow()

        elif self.problem_type == 'ED' and PowerFlowModel == 'DC':
            # ## Check dependency if economic dispatch is defined
            if self.dependency == None:
                print('ED is defined without UC results. Please input generator commitment results.')
                sys.exit(1)
            else:
                pass
            self.add_constraint_commit_enforce()
            self.add_variable_network()
            self.make_ybus()
            self.add_constraint_dc_power_flow()

        else:
            print("Functions are under construction")

        ## add objective funtion
        self.add_objective()


    def solve_problem(self, SolverName, verbose=None, SolverExecutablePath=None):
        if SolverExecutablePath==None:
            opt = SolverFactory(SolverName)
        else:
            opt = SolverFactory(SolverName, executable=SolverExecutablePath)

        if verbose == None or verbose == 0:
            self.solution = opt.solve(self.model)
        elif verbose == 1:
            self.solution = opt.solve(self.model, tee=True)
        else:
            self.solution = opt.solve(self.model)

        print(self.solution['solver'][0]['Termination condition'])


    def add_variable_generator(self):
        """
        pl: Incremental output of generator at Segment 1 (MW)
        Pg: Generator total power output (MW)
        R: System reserve (MW)
        SC: Variables associated with startup cost (associated with MW)
        status: On-off status
        """
        self.model.p1 = Var(self.iter_gen, self.iter_time, within=NonNegativeReals)
        self.model.Pg = Var(self.iter_gen, self.iter_time, within=NonNegativeReals)
        self.model.R = Var(self.iter_gen, self.iter_time, within=NonNegativeReals)
        self.model.SC = Var(self.iter_gen, self.iter_time, within=NonNegativeReals)
        self.model.status = Var(self.iter_gen, self.iter_time, within=Binary)


    def add_variable_network(self):
        """
        Note that the variables for generators are in MW, while the ones for power flow are in per unit.
        V: bus voltage (per unit)
        ang: bus angle (radian)
        Pf: active power flow (per unit)
        Qf: reactive power flow (per unit)
        """
        self.model.V = Var(self.iter_bus, self.iter_time, within=NonNegativeReals)
        self.model.A = Var(self.iter_bus, self.iter_time, within=Reals)
        self.model.Pf = Var(self.iter_bus, self.iter_bus, self.iter_time, within=Reals)
        self.model.Qf = Var(self.iter_bus, self.iter_bus, self.iter_time, within=Reals)
        self.model.Qg = Var(self.iter_gen, self.iter_time, within=Reals)


    def add_constraint_generator(self):
        """
        generator constraint
        """
        ## Constraints: generator output lower and upper bounds
        def con_gen_lower(m, i, t):
            return 0 <= m.p1[i, t]
        self.model.con_gen_lower = Constraint(self.iter_gen, self.iter_time, rule=con_gen_lower)

        def con_gen_upper(model, i, t):
            return model.p1[i,t] <= (self.ppc["gen"][i, 2] - self.ppc["gen"][i, 1]) * model.status[i, t]
        self.model.con_gen_upper = Constraint(self.iter_gen, self.iter_time, rule=con_gen_upper)

        ## generator total power output
        def con_gen_total_power(model, i, t):
            return model.Pg[i, t] == model.p1[i,t] + self.ppc["gen"][i, 1] * model.status[i, t]
        self.model.con_gen_total_power=Constraint(self.iter_gen, self.iter_time, rule=con_gen_total_power)

        ## generator reserve limit
        def con_gen_reserve_1(model, i, t):
            return model.R[i, t] <= self.ppc["gen"][i, 3] * model.status[i, t]
        self.model.con_gen_reserve_1 = Constraint(self.iter_gen, self.iter_time, rule=con_gen_reserve_1)

        def con_gen_reserve_2(model, i, t):
            return model.Pg[i, t] + model.R[i, t] <= self.ppc["gen"][i, 2] * model.status[i, t]
        self.model.con_gen_reserve_2 = Constraint(self.iter_gen, self.iter_time, rule=con_gen_reserve_2)

        ## generator ramp up (down) limit
        def con_gen_ramp_up(model, i, t):
            return model.Pg[i,t] - model.Pg[i, t-1] <= self.ppc["gen"][i, 4] * model.status[i,t-1] + self.ppc["gen"][i, 1]\
                   * (model.status[i,t] - model.status[i, t-1])
        self.model.con_gen_ramp_up = Constraint(self.iter_gen, self.iter_time_1, rule=con_gen_ramp_up)

        def con_gen_ramp_down(model, i, t):
            return model.Pg[i, t]-model.Pg[i, t-1] >= -self.ppc["gen"][i, 5]*model.status[i,t] - self.ppc["gen"][i,1]\
                   *(model.status[i,t-1] - model.status[i,t])
        self.model.con_gen_ramp_down = Constraint(self.iter_gen, self.iter_time_1, rule=con_gen_ramp_down)

        ## constraints associated with startup cost
        def con_startup(model,i,t):
            return model.SC[i, t] >= model.status[i, t] - model.status[i, t-1]
        self.model.con_startup=Constraint(self.iter_gen, self.iter_time_1, rule=con_startup)

        ## minimum up time constraint in general time series
        self.model.con_min_up = ConstraintList()
        for i in self.iter_gen:
            for t in self.iter_hour_updown:
                for k in range(1, self.ppc["gen"][i, 8]-1):
                    self.model.con_min_up.add(self.model.status[i,t] - self.model.status[i, t-1] <=
                                              self.model.status[i, min(t + k, self.number_time-1)])

        ## minimum up time constraint in initial condition when status[i,0]=1
        self.model.con_min_up_initial = ConstraintList()
        for i in self.iter_gen:
            for k in range(1, self.ppc["gen"][i,8]-1):
                self.model.con_min_up_initial.add(self.model.status[i, 0]<= self.model.status[i, min(0 + k, self.number_time-1)])

        ## minimum down time constraint in general time series
        self.model.con_min_down=ConstraintList()
        for i in self.iter_gen:
            for t in self.iter_hour_updown:
                for k in range(1, self.ppc["gen"][i, 9]-1):
                    self.model.con_min_up.add(self.model.status[i,t-1]-self.model.status[i,t] <= 1 - self.model.status[i,min(t+k, self.number_time-1)] )

        ## minimum up time constraint in initial condition when status[i,0]=1
        self.model.con_min_down_initial = ConstraintList()
        for i in self.iter_gen:
            for k in range(1, self.ppc["gen"][i,9]-1):
                self.model.con_min_down_initial.add(self.model.status[i, 0] <= 1 - self.model.status[i, min(0+k, self.number_time-1)])


    def add_constraint_commit_enforce(self):
        ## If the Fix_Status is given, then the problem becomes an economic dispatch.
        ## Then the status variable is enforced to be equal to the given commitment results
        self.model.con_commit_enforce = ConstraintList()
        for i in self.iter_gen:
             for t in self.iter_time:
                self.model.con_commit_enforce.add(self.model.status[i, t] == self.dependency[i][t])


    def add_constraint_power_balance(self):
        """
        simple power balance equation
        """
        def con_balance(model, t):
            return sum(model.Pg[i, t] for i in self.iter_gen) == self.PL[t]
        self.model.con_balance=Constraint(self.iter_time, rule=con_balance)


    def add_constraint_system_reserve(self):
        """
        system reserve requirement
        """
        def con_system_reserve(model,t):
            return sum(model.R[i,t] for i in self.iter_gen) == self.RV[t]
        self.model.con_system_reserve=Constraint(self.iter_time, rule=con_system_reserve)


    def add_constraint_dc_power_flow(self):
        """
        add power flow constraint
        DC power flow model with reactive power retained is used.
        References: P. A. Trodden, W. A. Bukhsh, A. Grothey, and K. I. M. McKinnon, “Optimization-based Islanding
        of power networks using piecewise linear AC power flow,” IEEE Trans. Power Syst., vol. 29, no. 3, pp. 1212–1220, 2014.
        DOI: 10.1109/TPWRS.2013.2291660
        """
        ## pass name-index
        idx = self.idx

        ## add nodal balance constraints
        ## looping is based on bus type for convenience
        ## Right hand side (RHS) = summation of power flows, reference direction is from bus
        ## Left hand side (LHS) = summation of generation and load in generation convention

        ## add nodal balance constraints
        self.model.con_nodal_balance = ConstraintList()
        for t in self.iter_time:
            for i in self.bus_type_load:
                RHS_P = 0
                RHS_Q = 0
                for j in self.iter_bus:  # loop all buses
                    if j != i:
                        RHS_P = RHS_P + self.model.Pf[i, j, t]
                        RHS_Q = RHS_Q + self.model.Qf[i, j, t]
                    else:
                        pass
                self.model.con_nodal_balance.add(-float(self.PL_bus[i][t]/self.base_mva) == RHS_P)
                self.model.con_nodal_balance.add(-float(self.QL_bus[i][t]/self.base_mva) == RHS_Q)

        for t in self.iter_time:
            for k in self.bus_type_gen['gen']:
                ## we use generator index for looping
                ## get bus variable index
                i = self.bus_type_gen['bus'][k]
                RHS_P = 0
                RHS_Q = 0
                for j in self.iter_bus:  # loop all buses
                    if j != i:
                        RHS_P = RHS_P + self.model.Pf[i, j, t]
                        RHS_Q = RHS_Q + self.model.Qf[i, j, t]
                    else:
                        pass
                self.model.con_nodal_balance.add(self.model.Pg[k, t]/self.base_mva - float(self.PL_bus[i][t]/self.base_mva) == RHS_P)
                self.model.con_nodal_balance.add(self.model.Qg[k, t]/self.base_mva - float(self.QL_bus[i][t]/self.base_mva) == RHS_Q)

        ## add line flow constraints
        self.model.con_line_flow = ConstraintList()
        for t in self.iter_time:
            for i in self.iter_bus:
                for j in self.iter_bus:
                    if i == j:
                        self.model.con_line_flow.add(self.model.Pf[i, j, t] == 0)
                        self.model.con_line_flow.add(self.model.Qf[i, j, t] == 0)
                    else:
                        self.model.con_line_flow.add(self.model.Pf[i, j, t] ==
                                                     self.G[i, j] * (2 * self.model.V[i, t] - 1)
                                                     - self.G[i, j] * (self.model.V[i, t] + self.model.V[j, t] - 1)
                                                     - self.B[i, j] * (self.model.A[i, t] - self.model.A[j, t]))
                        self.model.con_line_flow.add(self.model.Qf[i, j, t] ==
                                                      self.B[i, j] * (1 - 2 * self.model.V[i, t])
                                                     + self.B[i, j] * (self.model.V[i, t] + self.model.V[j, t] - 1)
                                                     - self.G[i, j] * (self.model.A[i, t] - self.model.A[j, t]))

        # slack bus angle should be always zero
        self.model.con_slack_bus = ConstraintList()
        for t in self.iter_time:
            for i in self.bus_type_ref:
                self.model.con_slack_bus.add(self.model.A[i,t] == 0)

        # add voltage constraints
        self.model.con_voltage = ConstraintList()
        for t in self.iter_time:
            for i in self.iter_bus:
                self.model.con_voltage.add(self.ppc['bus'][i, self.idx.VMIN] <= self.model.V[i, t] <= self.ppc['bus'][i, self.idx.VMAX])


    def add_objective(self):
        def obj_cost_min(model):
            marginal_cost = sum(sum(model.p1[i, t] * self.cost_marginal[i] for t in self.iter_time) for i in self.iter_gen)
            fixed_cost = sum(sum(model.status[i, t] * self.cost_fixed[i] for t in self.iter_time) for i in self.iter_gen)
            reserve_cost = sum(sum(model.R[i, t] * self.ppc["gen"][i, 6] for t in self.iter_time) for i in self.iter_gen)
            startup_cost = sum(sum(model.SC[i, t] * self.ppc["gen"][i, 7] for t in self.iter_time) for i in self.iter_gen)
            total_cost = marginal_cost + fixed_cost + reserve_cost + startup_cost
            return total_cost
        self.model.obj_cost = Objective(rule=obj_cost_min)


    def get_solution_2d(self, VariableName, iter, iter_time, SolDict=None):
        """
        Get solution from pyomo-formulated optimization problem
        """
        if SolDict == None:
            print('No designated dictionary is given. One will be created automatically. Remeber to input attraibute values.')
            SolDict = SolutionDict()
        else:
            pass

        ## get string type attribute name
        variable = operator.attrgetter(VariableName)(self.model)

        for i in iter:
            SolDict[i] = []
            for t in iter_time:
                SolDict[i].append(variable[i, t].value)

        return SolDict


    def get_slotuion_flow(self, iter_time, SolDict_P=None, SolDict_Q=None):
        """
        Get solution from pyomo-formulated optimization problem
        Since power flow is defined using bus iterator. There will be many virtual lines that does not exit.
        In optimization, we force them to be zero.
        For plotting, it is better to read the power flow by line.
        """
        if SolDict_P == None:
            print('No designated dictionary is given. One will be created automatically. Remeber to input attraibute values.')
            SolDict_P = SolutionDict()
        else:
            pass

        if SolDict_Q == None:
            print('No designated dictionary is given. One will be created automatically. Remeber to input attraibute values.')
            SolDict_Q = SolutionDict()
        else:
            pass

        for k in range(self.number_line):
            idx_from_bus = int(self.ppc['branch'][k, self.idx.F_BUS] - 1)
            idx_to_bus = int(self.ppc['branch'][k, self.idx.T_BUS] - 1)
            SolDict_P[k] = []
            SolDict_Q[k] = []
            for t in iter_time:
                SolDict_P[k].append(self.model.Pf[idx_from_bus, idx_to_bus, t].value)
                SolDict_Q[k].append(self.model.Qf[idx_from_bus, idx_to_bus, t].value)

        return SolDict_P, SolDict_Q




