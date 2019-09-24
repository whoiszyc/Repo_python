import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *
from collections import OrderedDict
import operator




class idx:
    """
    Define indices with name for convenience
    Value corresponds to the matpower data format
    """
    F_BUS = 0  # f, from bus number
    T_BUS = 1  # t, to bus number
    BR_R = 2  # r, resistance (p.u.)
    BR_X = 3  # x, reactance (p.u.)
    BR_B = 4  # b, total line charging susceptance (p.u.)
    RATE_A = 5  # rateA, MVA rating A (long term rating)
    RATE_B = 6  # rateB, MVA rating B (short term rating)
    RATE_C = 7  # rateC, MVA rating C (emergency rating)
    TAP = 8  # ratio, transformer off nominal turns ratio
    SHIFT = 9  # angle, transformer phase shift angle (degrees)
    BR_STATUS = 10  # initial branch status, 1 - in service, 0 - out of service
    ANGMIN = 11  # minimum angle difference, angle(Vf) - angle(Vt) (degrees)
    ANGMAX = 12  # maximum angle difference, angle(Vf) - angle(Vt) (degrees)

    ## included in power flow solution, not necessarily in input
    PF = 13  # real power injected at "from" bus end (MW)
    QF = 14  # reactive power injected at "from" bus end (MVAr)
    PT = 15  # real power injected at "to" bus end (MW)
    QT = 16  # reactive power injected at "to" bus end (MVAr)

    ## included in opf solution, not necessarily in input
    ## assume objective function has units, u
    MU_SF = 17  # Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA)
    MU_ST = 18  # Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA)
    MU_ANGMIN = 19  # Kuhn-Tucker multiplier lower angle difference limit
    MU_ANGMAX = 20  # Kuhn-Tucker multiplier upper angle difference limit


    ## define bus types
    PQ      = 1
    PV      = 2
    REF     = 3
    NONE    = 4

    ## define the indices
    BUS_I       = 0    # bus number (1 to 29997)
    BUS_TYPE    = 1    # bus type
    PD          = 2    # Pd, real power demand (MW)
    QD          = 3    # Qd, reactive power demand (MVAr)
    GS          = 4    # Gs, shunt conductance (MW at V = 1.0 p.u.)
    BS          = 5    # Bs, shunt susceptance (MVAr at V = 1.0 p.u.)
    BUS_AREA    = 6    # area number, 1-100
    VM          = 7    # Vm, voltage magnitude (p.u.)
    VA          = 8    # Va, voltage angle (degrees)
    BASE_KV     = 9    # baseKV, base voltage (kV)
    ZONE        = 10   # zone, loss zone (1-999)
    VMAX        = 11   # maxVm, maximum voltage magnitude (p.u.)
    VMIN        = 12   # minVm, minimum voltage magnitude (p.u.)

    ## included in opf solution, not necessarily in input
    ## assume objective function has units, u
    LAM_P       = 13   # Lagrange multiplier on real power mismatch (u/MW)
    LAM_Q       = 14   # Lagrange multiplier on reactive power mismatch (u/MVAr)
    MU_VMAX     = 15   # Kuhn-Tucker multiplier on upper voltage limit (u/p.u.)
    MU_VMIN     = 16   # Kuhn-Tucker multiplier on lower voltage limit (u/p.u.)



class UnitCommitment:

    def get_network_data(self, ppc, idx):
        self.ppc = ppc
        self.idx = idx

        ## calculate base
        self.base_mva = ppc['baseMVA']     # MVA
        self.base_KV = ppc['baseKV']       # KV
        self.base_KA = self.base_mva/self.base_KV    # KA
        self.base_Ohm = self.base_KV/self.base_KA    # Ohm
        self.base_Siemens = 1/self.base_Ohm     # Siemens

        ## get size of bus, line, generator and time
        self.number_bus = ppc['bus'].shape[0] # shape returns (row,column)
        self.number_gen = ppc['gen'].shape[0]
        self.number_line = ppc['branch'].shape[0]


    def create_iterator(self, number_time):
        self.number_time = number_time
        self.iter_time = range(0,self.number_time)
        self.iter_time_1 = range(1,self.number_time)

        ## Index number (for list and array starting from 0) as iterator
        self.iter_bus = range(0,self.number_bus)
        self.iter_line = range(0, self.number_line)
        self.iter_gen = range(0, self.number_gen)


        ## Generator iterators for minimum up and down time for different generators
        ## Assume the hour index is: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ... N
        ## Minimum up time iterator: The index t of the formulation will always start at 1, end at N-1
        self.iter_hour_updown = range(1, self.number_time - 1)


        ## create name as iterator for indexing and searching
        self.iter_bus_name = []
        for i in self.iter_bus:
            self.iter_bus_name.append('bus_{}'.format(int(self.ppc['bus'][i, 0])))

        self.iter_line_name = []
        for i in self.iter_line:
            self.iter_line_name.append('line_{}_{}'.format(int(self.ppc['branch'][i, 0]),int(self.ppc['branch'][i, 1])))

        self.iter_gen_name = []
        for i in self.iter_gen:
            self.iter_gen_name.append('gen_{}'.format(int(self.ppc['gen'][i, 0])))


    def get_system_reserve_data(self, RV):
        self.RV = RV


    def get_system_load_data(self, PL, QL=None):
        self.PL = PL

        if QL == None:
            """use data from ppc as the Q load"""
            total_Q = sum(self.ppc['bus'][i, 3] for i in range(self.number_bus))
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
            total_P = sum(self.ppc['bus'][i, 2] for i in range(self.number_bus))
            total_Q = sum(self.ppc['bus'][i, 3] for i in range(self.number_bus))
            for i in self.iter_bus:
                self.PL_bus[i] = []
                self.QL_bus[i] = []
                PL_percentage = self.ppc['bus'][i, 2] / total_P
                QL_percentage = self.ppc['bus'][i, 3] / total_Q
                for t in self.iter_time:
                    self.PL_bus[i].append(PL_percentage * self.PL[t])
                    self.QL_bus[i].append(QL_percentage * self.QL[t])

        elif type == 2:
            print('This function has not been added')
        else:
            pass


    def get_cost(self):
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
            cost_curve[i] = self.ppc["gencost"][i, 1]*p**2 + self.ppc["gencost"][i,2]*p + self.ppc["gencost"][i, 3]
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


    def define_problem(self, SchedulingType, PowerFlowModel = None):
        """common variables and constraints"""
        self.add_variable_generator()
        self.add_constraint_generator()
        self.add_constraint_system_reserve()

        if SchedulingType == 'UC' and PowerFlowModel == None:
            self.add_constraint_power_balance()

        elif SchedulingType == 'UC' and PowerFlowModel == 'dc':
            self.add_variable_network()
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
        pl: Incremental output of generator at Segment 1
        Pg: Generator total power output
        R: System reserve
        SC: Variables associated with startup cost
        status: On-off status
        """
        self.model.p1 = Var(self.iter_gen, self.iter_time, within=NonNegativeReals)
        self.model.Pg = Var(self.iter_gen, self.iter_time, within=NonNegativeReals)
        self.model.R = Var(self.iter_gen, self.iter_time, within=NonNegativeReals)
        self.model.SC = Var(self.iter_gen, self.iter_time, within=NonNegativeReals)
        self.model.status = Var(self.iter_gen, self.iter_time, within=Binary)


    def add_variable_network(self):
        """
        V: bus voltage
        ang: bus angle
        Pf: active power flow
        Qf: reactive power flow
        """
        self.model.V = Var(self.iter_bus, self.iter_time, within=NonNegativeReals)
        self.model.ang = Var(self.iter_bus, self.iter_time, within=Reals)
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
                self.model.con_min_up_initial.add(self.model.status[i, 0]<= self.model.status[i, min(0+k, self.number_time-1)])

        ## minimum down time constraint in general time series
        self.model.con_min_down=ConstraintList()
        for i in self.iter_gen:
            for t in self.iter_hour_updown:
                for k in range(1, self.ppc["gen"][i,9]-1):
                    self.model.con_min_up.add(self.model.status[i,t-1]-self.model.status[i,t] <= 1 - self.model.status[i,min(t+k, self.number_time-1)] )

        ## minimum up time constraint in initial condition when status[i,0]=1
        self.model.con_min_down_initial = ConstraintList()
        for i in self.iter_gen:
            for k in range(1, self.ppc["gen"][i,9]-1):
                self.model.con_min_down_initial.add(self.model.status[i, 0] <= 1 - self.model.status[i, min(0+k, self.number_time-1)])


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
        piece-wise linearized power flow may be added later
        references: (1)10.1109/TPWRS.2013.2291660, (2)10.1109/TPWRS.2017.2733224
        """
        ## pass name-index
        idx = self.idx

        ## create bus type list and store bus index
        ## for a generator bus, ['bus'] stores bus
        self.bus_type_gen = OrderedDict()
        self.bus_type_gen['bus'] = []
        self.bus_type_gen['gen'] = []
        self.bus_type_ref = []
        self.bus_type_load = []
        for i in self.iter_bus:
            if self.ppc['bus'][i, idx.BUS_TYPE] == 1:
                self.bus_type_load.append(i)

            elif self.ppc['bus'][i, idx.BUS_TYPE] == 2 or 3:
                self.bus_type_gen['bus'].append(i)
                k = list(self.ppc['gen'][:,0]).index(i + 1)
                self.bus_type_gen['gen'].append(k)

            elif self.ppc['bus'][i, idx.BUS_TYPE] == 3:
                self.bus_type_ref.append(i)

            else:
                print('No such type of bus')

        ## add nodal balance constraints
        ## looping is based on bus type for convenience
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
                self.model.con_nodal_balance.add(-float(self.PL_bus[i][t]) == RHS_P)
                self.model.con_nodal_balance.add(-float(self.QL_bus[i][t]) == RHS_Q)

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
                self.model.con_nodal_balance.add(self.model.Pg[k, t] - float(self.PL_bus[i][t]) == RHS_P)
                self.model.con_nodal_balance.add(self.model.Qg[k, t] - float(self.QL_bus[i][t]) == RHS_Q)

        ## add nodal balance constraints
        self.model.con_line_flow = ConstraintList()
        for t in self.iter_time:
            for i in self.iter_bus:
                for j in self.iter_bus:
                    if i == j:
                        self.model.con_line_flow.add(self.model.Pf[i, j, t] == 0)
                        self.model.con_line_flow.add(self.model.Qf[i, j, t] == 0)
                    else:
                        self.model.con_line_flow.add(self.model.Pf[i, j, t] - self.B[i, j]*(self.model.ang[i, t]-self.model.ang[j, t])==0)
                        self.model.con_line_flow.add(
                            self.model.Qf[i,j,t]-self.B[i,j]*(1-2*self.model.V[i,t])+self.B[i,j]*(self.model.V[i,t]-self.model.V[j,t]-1)==0)

        # slack bus angle should be always zero
        self.model.con_slack_bus = ConstraintList()
        for t in self.iter_time:
            for i in self.bus_type_ref:
                self.model.con_slack_bus.add(self.model.ang[i,t]==0)


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
        if SolDict == None:
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


    # def plot_generator_solution_dic_2d(self, SolDict, x_str='Time', y_str='Value', title_str='Generator Scheduling Results'):
    #     plt.figure(figsize=(12, 5))
    #     for i in u.iter_gen:
    #         plt.step(self.iter_time, SolDict[i], label='Gen {}'.format(int(u.ppc["gen"][i, 0])), linewidth=3)
    #     plt.title(title_str)
    #     plt.xlabel(x_str)
    #     plt.ylabel(y_str)
    #     plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    #     plt.show()


    def make_ybus(self):
        """
        make Y bus from matpower data format
        """

        ## pass parameter
        idx = self.idx
        baseMVA = self.ppc["baseMVA"]
        bus = self.ppc["bus"]
        branch = self.ppc["branch"]

        ## below from pypower with indexing change
        nb = bus.shape[0]          # number of buses
        nl = branch.shape[0]       # number of lines

        ## for each branch, compute the elements of the branch admittance matrix where
        ##      | If |   | Yff  Yft |   | Vf |
        ##      |    | = |          | * |    |
        ##      | It |   | Ytf  Ytt |   | Vt |
        stat = branch[:, idx.BR_STATUS]              ## ones at in-service branches
        Ys = stat / (branch[:, idx.BR_R] + 1j * branch[:, idx.BR_X])  ## series admittance
        Bc = stat * branch[:, idx.BR_B]              ## line charging susceptance
        tap = np.ones(nl)                           ## default tap ratio = 1
        i = np.nonzero(branch[:, idx.TAP])              ## indices of non-zero tap ratios
        tap[i] = branch[i, idx.TAP]                  ## assign non-zero tap ratios
        tap = tap * np.exp(1j * np.pi / 180 * branch[:, idx.SHIFT])  ## add phase shifters
        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * np.conj(tap))
        Yft = - Ys / np.conj(tap)
        Ytf = - Ys / tap

        ## compute shunt admittance
        ## if Psh is the real power consumed by the shunt at V = 1.0 p.u.
        ## and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
        ## then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
        ## i.e. Ysh = Psh + j Qsh, so ...
        ## vector of shunt admittances
        Ysh = (bus[:, idx.GS] + 1j * bus[:, idx.BS]) / baseMVA

        ## build connection matrices
        f = branch[:, idx.F_BUS] - 1                           ## list of "from" buses
        t = branch[:, idx.T_BUS] - 1                           ## list of "to" buses

        ## connection matrix for line & from buses
        Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))

        ## connection matrix for line & to buses
        Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))

        ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
        ## at each branch's "from" bus, and Yt is the same for the "to" bus end
        i = np.r_[range(nl), range(nl)]  # double set of row indices

        Yf = csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])), (nl, nb))
        Yt = csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])), (nl, nb))

        ## build Ybus
        Ybus = Cf.T * Yf + Ct.T * Yt + csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))

        ## Add Y bus into attribute
        self.Ybus = Ybus

        ## split Y bus into real and imaginary parts
        for i in range(self.number_bus):
            for j in range(self.number_bus):
                if i != j:
                    Ybus[i, j] = Ybus[i, j]*(-1)
                else:
                    pass
        self.G = np.real(Ybus)
        self.B = np.imag(Ybus)




class SolutionDict(OrderedDict):
    """
    solution dictionary is an ordered dictionary that stores the optimization results
    First level key is variable category like generator, load
    Second level key is variable index like generator 30, 31
    At last, time series data is stored in a list
    """

    def plot_step_2d(self, legendlist=None, x_str='Time', y_str='Value', title_str='Generator Scheduling Results', figsize=(12,5)):
        """step plot"""
        plt.figure(figsize=figsize)

        key_list = []
        for i in self.keys():
            key_list.append(i)

        total_time = []
        for i in range(len(key_list)):
            total_time.append(len(self[i]))

        if legendlist==None:
            legendlist = ['variable {}'.format(i) for i in key_list]

        for i in key_list:
            plt.step(range(0, total_time[i]), self[i], label=legendlist[i], linewidth=3)
        plt.title(title_str)
        plt.xlabel(x_str)
        plt.ylabel(y_str)
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
        plt.show()



## Import the studied system data using matpower/pypower format
## Read IEEE 39 bus system data
import Func_IEEETestCase
pppc = Func_IEEETestCase.case39()


## Get the ISO level time-series data
## This data set should be generated by modules developed by BNL
## read ISO level load data
## index for reading load data from pandas data frame
LoadData=pd.read_csv('P_Load.csv')  # pandas dataframe
PL = list(LoadData.iloc[:, 1])  # normal, type: series


## ISO level spinning reserve requirement
ReserveData=pd.read_csv('P_Reserve.csv')  # pandas dataframe
RV = list(ReserveData.iloc[:, 1])  # normal, type: series


## define an instance of a UC problem class
u = UnitCommitment()

## prepare data
u.get_network_data(pppc, idx)
u.create_iterator(24)
u.get_system_reserve_data(RV)
u.get_system_load_data(PL)
u.set_bus_load(1)
u.get_cost()

## define UC without power flow
u.define_pyomo_model('Concrete')
u.define_problem('UC')
u.solve_problem('glpk', 1)
gen_power_normal = u.get_solution_2d('Pg', u.iter_gen, u.iter_time)
gen_power_status = u.get_solution_2d('status', u.iter_gen, u.iter_time)
gen_power_normal.plot_step_2d(u.iter_gen_name, 'Time (h)', 'Power (MW)', 'Generator Output', (12, 5))

## solve a UC with power flow constraints
# u.make_ybus()
# u.define_pyomo_model('Concrete')
# u.define_problem('UC','dc')
# u.solve_problem('glpk', 1)





