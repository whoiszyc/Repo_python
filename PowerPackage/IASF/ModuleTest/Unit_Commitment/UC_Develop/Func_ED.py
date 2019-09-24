import numpy as np
import pandas as pd
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *


"""
Economic dispatch problem considering fixed generator on-off
"""
# To call a function within a script, the function should be put in front
# Define UC as a function (dictionary of system data, vector of load forecasting, vector of reserve requirement, solver_name)
def ED_FlowFree_StatusFix (ppc, PL, RV, GenStatus, cost_LoadShedding, solver_name):

    # store necessary problem formulation data for result processing
    problem_data={}

    # calculate base
    base_mva=ppc['baseMVA']     # MVA
    base_KV=ppc['baseKV']       # KV
    base_KA=base_mva/base_KV    # KA
    base_Ohm=base_KV/base_KA    # Ohm
    base_Siemens=1/base_Ohm     # Siemens

    # get size of bus, line and generator
    number_bus=ppc['bus'].shape[0] # shape returns (row,column)
    number_gen=ppc['gen'].shape[0]
    number_hour = len(RV)

    # create iterator for bus, generator and horizon
    iter_bus=np.arange(0,number_bus)
    iter_gen=np.arange(0,number_gen)
    iter_hour=np.arange(0,number_hour)
    iter_hour_1=np.arange(1,number_hour)

    # Generator iterators for minimum up and down time for different generators
    # Assume the hour index is: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ... N
    # Minimum up time iterator: The index t of the formulation will always start at 1, end at N-1
    iter_hour_updown=np.arange(1, number_hour-1)

    # store the problem data for processing
    problem_data["number_bus"] = number_bus
    problem_data["number_gen"] = number_gen
    problem_data["number_hour"] = number_hour
    problem_data["iter_gen"] = iter_gen
    problem_data["iter_hour"] = iter_hour


    """
    Load data distribution
    The load data (active power) is distributed according the static load data of standard IEEE 39-bus.
    For now we are not sure this is the way that we want to perform our attack.
    """
    # # get static load data from standard IEEE 39-bus system (unit: MW/Mvar)
    # load_p=ppc['bus'][:,2]
    # load_q=ppc['bus'][:,3]


    """
    Convert the generator cost to a simplified version C=A*u + mc*delta_p
    """
    p = sy.Symbol('p')
    cost_curve={}
    cost_curve_diff={}
    cost_curve_simple={}
    cost_fixed={}
    cost_marginal={}
    for i in iter_gen:
        cost_curve[i]=ppc["gencost"][i,1]*p**2 + ppc["gencost"][i,2]*p + ppc["gencost"][i,3]
        cost_curve_diff[i]=sy.diff(cost_curve[i],p)
        cost_fixed[i]=cost_curve[i].subs(p, ppc["gen"][i, 1])
        cost_marginal[i]=cost_curve_diff[i].subs(p,(ppc["gen"][i, 2]-ppc["gen"][i, 1])*0.5)
        cost_curve_simple[i]=cost_marginal[i]*(p-ppc["gen"][i, 1])+cost_fixed[i]



    """
    Define problem in pyomo
    """
    ####################### Define a concrete pyomo model ####################
    model = ConcreteModel()



    ########################## Define variables ###########################
    # ! Note: Variables are defined using either iterators or sets, not total numbers
    model.p1=Var(iter_gen, iter_hour, within=NonNegativeReals) # Incremental output of generator at Segment 1
    model.Pg=Var(iter_gen, iter_hour, within=NonNegativeReals) # Generator total power output
    model.R=Var(iter_gen, iter_hour, within=NonNegativeReals)
    model.SC=Var(iter_gen, iter_hour, within=NonNegativeReals)  # Variables associated with startup cost
    # model.status=Var(iter_gen, iter_hour, within=Binary) # On-off status (considering fixed during the economic dispatch)
    model.LS=Var(iter_hour, within=NonNegativeReals)  # Load shedding


    ################ Define constraints ##########################
    # Constraints: generator output lower and upper bounds
    def con_gen_lower(model,i,t):
        return 0<= model.p1[i,t]
    model.con_gen_lower=Constraint(iter_gen, iter_hour, rule=con_gen_lower)

    def con_gen_upper(model,i,t):
        return model.p1[i,t] <= (ppc["gen"][i,2]-ppc["gen"][i,1])*GenStatus[i][t]
    model.con_gen_upper=Constraint(iter_gen, iter_hour, rule=con_gen_upper)

    # generator total power output
    def con_gen_total_power(model,i,t):
        return model.Pg[i,t] == model.p1[i,t] + ppc["gen"][i,1]*GenStatus[i][t]
    model.con_gen_total_power=Constraint(iter_gen, iter_hour, rule=con_gen_total_power)

    # generator reserve limit
    def con_gen_reserve_1(model,i,t):
        return model.R[i,t] <= ppc["gen"][i,3]*GenStatus[i][t]
    model.con_gen_reserve_1=Constraint(iter_gen, iter_hour, rule=con_gen_reserve_1)

    def con_gen_reserve_2(model,i,t):
        return model.Pg[i,t] + model.R[i,t] <= ppc["gen"][i,2]*GenStatus[i][t]
    model.con_gen_reserve_2=Constraint(iter_gen, iter_hour, rule=con_gen_reserve_2)

    # generator ramp up (down) limit
    def con_gen_ramp_up(model,i,t):
        return model.Pg[i,t]-model.Pg[i,t-1] <= ppc["gen"][i,4]*GenStatus[i][t-1] + ppc["gen"][i,1]*(GenStatus[i][t]-GenStatus[i][t-1])
    model.con_gen_ramp_up=Constraint(iter_gen, iter_hour_1, rule=con_gen_ramp_up)

    def con_gen_ramp_down(model,i,t):
        return model.Pg[i,t]-model.Pg[i,t-1] >= -ppc["gen"][i,5]*GenStatus[i][t] - ppc["gen"][i,1]*(GenStatus[i][t-1]-GenStatus[i][t])
    model.con_gen_ramp_down=Constraint(iter_gen, iter_hour_1, rule=con_gen_ramp_down)

    # network constraints: power balance
    def con_balance(model,t):
        return sum(model.Pg[i,t] for i in iter_gen) + model.LS[t] == PL[t]
    model.con_balance=Constraint(iter_hour, rule=con_balance)

    # system reserve requirement
    def con_system_reserve(model,t):
        return sum(model.R[i,t] for i in iter_gen) == RV[t]
    model.con_system_reserve=Constraint(iter_hour, rule=con_system_reserve)

    # constraints associated with startup cost
    def con_startup(model,i,t):
        return model.SC[i,t] >= GenStatus[i][t] - GenStatus[i][t-1]
    model.con_startup=Constraint(iter_gen, iter_hour_1, rule=con_startup)


    ########################### Objective and solving the problem ###############################
    def obj_cost(model):
        return sum(sum(model.p1[i,t]*cost_marginal[i]         # marginal cost
                       + GenStatus[i][t]*cost_fixed[i]     # fixed cost
                       + model.R[i,t]*ppc["gen"][i,6]         # reserve cost
                       + model.SC[i,t]*ppc["gen"][i,7]        # startup cost
                       for t in iter_hour) for i in iter_gen)+ sum(model.LS[t]*cost_LoadShedding for t in iter_hour)
    model.obj_cost=Objective(rule=obj_cost)



    opt = SolverFactory(solver_name)
    results = opt.solve(model)

    return problem_data, model

