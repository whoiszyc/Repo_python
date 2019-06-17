
import pandas as pd
import numpy as np
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *


# Setting for plot
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)



"""
Define problem in pyomo
"""
#############################  Define a abstract pyomo model  ###############
model = AbstractModel()



#############################   Define sets  ###########################
model.T=Set()
model.Tshift=Set()
model.G=Set()  # The generator set
model.L=Set()  # The load set



###################   Define parmeters  #########################
# generator parameters (time-invariant)
# If the parameters are time-variant, then it can be defined as: model.p=(model.G, model.T)
model.Pmin=Param(model.G, within=NonNegativeReals, default=0)
model.Pmax=Param(model.G, within=NonNegativeReals, default=50000)
model.Rmax=Param(model.G, within=NonNegativeReals) # maximum reserve limit (minimum reserve limit is zero)
model.RU=Param(model.G, within=NonNegativeReals) # ramp up limit
model.RD=Param(model.G, within=NonNegativeReals) # ramp down limit
model.cost_fixed=Param(model.G, within=NonNegativeReals, default=0)
model.cost_marginal=Param(model.G, within=NonNegativeReals, default=20)
model.cost_reserve=Param(model.G, default=0)
model.cost_startup=Param(model.G, default=0)

# system reserve requirement
model.Rmin_sys=Param(within=NonNegativeReals)

# system load forecasting
model.PL=Param(model.L, model.T, within=NonNegativeReals)



# ################   Define variables #########################
# ! Note: Variables are defined using either iterators or sets, not total numbers
model.p1=Var(model.G, model.T, within=NonNegativeReals) # Incremental output of generator at Segment 1
model.Pg=Var(model.G, model.T, within=NonNegativeReals) # Incremental output of generator at Segment 1
model.R=Var(model.G, model.T, within=NonNegativeReals)
model.SC=Var(model.G, model.T, within=NonNegativeReals)  # Variables associated with startup cost
model.status=Var(model.G, model.T, within=Binary) # On-off status



################ Define constraints ##########################
# Constraints: generator output lower and upper bounds
def con_gen_lower(model,i,t):
    return 0<= model.p1[i,t]
model.con_gen_lower=Constraint(model.G, model.T, rule=con_gen_lower)

def con_gen_upper(model,i,t):
    return model.p1[i,t] <= (model.Pmax[i]-model.Pmin[i])*model.status[i, t]
model.con_gen_upper=Constraint(model.G, model.T, rule=con_gen_upper)

# generator total power output
def con_gen_total_power(model,i,t):
    return model.Pg[i,t] == model.p1[i,t] + model.Pmin[i]*model.status[i, t]
model.con_gen_total_power=Constraint(model.G, model.T, rule=con_gen_total_power)

# generator reserve limit
def con_gen_reserve_1(model,i,t):
    return model.R[i,t] <= model.Rmax[i]*model.status[i, t]
model.con_gen_reserve_1=Constraint(model.G, model.T, rule=con_gen_reserve_1)

def con_gen_reserve_2(model,i,t):
    return model.Pg[i,t]+model.R[i,t] <= model.Pmax[i]*model.status[i, t]
model.con_gen_reserve_2=Constraint(model.G, model.T, rule=con_gen_reserve_2)

# generator ramp up (down) limit
def con_gen_ramp_up(model,i,t):
    return model.Pg[i,t]-model.Pg[i,t-1] <= model.RU[i]*model.status[i,t-1] + model.Pmin[i]*(model.status[i,t]-model.status[i,t-1])
model.con_gen_ramp_up=Constraint(model.G, model.Tshift, rule=con_gen_ramp_up)

def con_gen_ramp_down(model,i,t):
    return model.Pg[i,t]-model.Pg[i,t-1] >= -model.RD[i]*model.status[i,t] - model.Pmin[i]*(model.status[i,t-1]-model.status[i,t])
model.con_gen_ramp_down=Constraint(model.G, model.Tshift, rule=con_gen_ramp_down)

# network constraints: power balance
def con_balance(model,t):
    return sum(model.Pg[i,t] for i in model.G) == sum(model.PL[l,t] for l in model.L)
model.con_balance=Constraint(model.T, rule=con_balance)

# System reserve requirement
def con_system_reserve(model,t):
    return sum(model.R[i,t] for i in model.G) == model.Rmin_sys
model.con_system_reserve=Constraint(model.T, rule=con_system_reserve)

# constraints associated with startup cost
def con_startup(model,i,t):
    return model.SC[i,t]>=model.status[i,t]-model.status[i,t-1]
model.con_startup=Constraint(model.G, model.Tshift, rule=con_startup)



########################### Objective and solving the problem ###############################
def obj_gen_cost(model):
    return sum(sum(model.p1[i,t]*model.cost_marginal[i]         # marginal cost
                   + model.status[i, t]*model.cost_fixed[i]     # fixed cost
                   + model.R[i,t]*model.cost_reserve[i]         # reserve cost
                   + model.SC[i,t]*model.cost_startup[i]        # startup cost
                   for t in model.T) for i in model.G)
model.obj_gen_cost=Objective(rule=obj_gen_cost)


# initialize abstract models with data of different scenarios
instance_normal = model.create_instance('IEEE39_full.dat')
opt = SolverFactory("glpk")
results =opt.solve(instance_normal)
results.write()



"""
Visualize the results
"""
# Convert set to list for plot
scheduling_time=list(instance_normal.T.value)

# Export data (pyomo book 173)
# actual power output under normal case
gen_power_normal={}  # create an empty dictionary
for i in instance_normal.G.value:
    gen_power_normal[i]=[]   # create an empty list
    for h in instance_normal.T.value:
        gen_power_normal[i].append(value(instance_normal.p1._data[i, h]) + value(instance_normal.status._data[i, h]) * instance_normal.Pmin._data[i])


plt.figure(figsize=(12,5))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
for i in instance_normal.G.value:
    plt.step(scheduling_time, gen_power_normal[i], label='Gen {}'.format(i), linewidth=3)
plt.title('UC under Normal Case')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()


#
# # Plot: 3D
# fig = plt.figure(figsize=(12,5))
# ax = fig.add_subplot(111, projection='3d')
# for i in instance_normal.G.value:
#     xs = scheduling_time
#     ys = gen_power_normal[i]
#     ax.bar(xs, ys, zs=i, zdir='y', alpha=0.7, label='Gen {}'.format(i))
# ax.set_xlabel('Hours')
# ax.set_ylabel('Generator Index')
# ax.set_zlabel('Power')
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
# plt.show()
# plt.title('UC under Normal Case')



