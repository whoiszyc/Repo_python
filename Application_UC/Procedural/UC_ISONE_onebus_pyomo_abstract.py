
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
##############  Define a abstract pyomo model  ##############
model = AbstractModel()


##############   Define the time set  ##############
# no parameters are only associated with the time set
model.T=Set()


##############   Define the generator set and its parmeters  ##############
model.G=Set()  # The generator set

# parameters (time-invariant)
# If the parameters are time-variant, then it can be defined as: model.p=(model.G, model.T)
model.Pmin=Param(model.G, within=NonNegativeReals, default=0)
model.Pmax=Param(model.G, within=NonNegativeReals, default=50000)
model.cost_fixed=Param(model.G, within=NonNegativeReals, default=0)
model.cost_marginal=Param(model.G, within=NonNegativeReals, default=20)

##############   Define the load set and its parmeters  ##############
model.L=Set()  # The load set

model.P=Param(model.L, model.T, within=NonNegativeReals)



# Define variables
# ! Note: Variables are defined using either iterators or sets, not total numbers
model.p1=Var(model.G, model.T, within=NonNegativeReals) # Incremental output of generator at Segment 1
model.status=Var(model.G, model.T, within=Binary) # On-off status



# Constraints: generator lower and upper bounds
def con_gen_lower(model,i,t):
    return 0<= model.p1[i,t]
model.con_gen_lower=Constraint(model.G, model.T, rule=con_gen_lower)


def con_gen_upper(model,i,t):
    return model.p1[i,t] <= (model.Pmax[i]-model.Pmin[i])*model.status[i, t]
model.con_gen_upper=Constraint(model.G, model.T, rule=con_gen_upper)



# Network constraints: power balance
def con_balance_normal(model,t):
    return sum(model.p1[i,t]+model.status[i,t]*model.Pmin[i] for i in model.G) == sum(model.P[l,t] for l in model.L)
model.con_balance=Constraint(model.T, rule=con_balance_normal)




# Objective: minimum cost
def obj_gen_cost(model):
    return sum(sum(model.p1[i,t]*model.cost_marginal[i]+model.status[i, t]*model.cost_fixed[i] for t in model.T) for i in model.G)
model.obj_gen_cost=Objective(rule=obj_gen_cost)


# Initialize abstract models with data of different scenarios
instance_normal = model.create_instance('IEEE39.dat')
instance_attack_scenario1 = model.create_instance('IEEE39_attack_scenario1.dat')
instance_attack_scenario2 = model.create_instance('IEEE39_attack_scenario2.dat')
opt = SolverFactory("glpk")
opt.solve(instance_normal)
opt.solve(instance_attack_scenario1)
opt.solve(instance_attack_scenario2)



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

# # actual power output under attack 1
gen_power_attack_scenario1={}  # create an empty dictionary
for i in instance_attack_scenario1.G.value:
    gen_power_attack_scenario1[i]=[]   # create an empty list
    for h in instance_attack_scenario1.T.value:
        gen_power_attack_scenario1[i].append(value(instance_attack_scenario1.p1._data[i, h]) + value(instance_attack_scenario1.status._data[i, h]) * instance_attack_scenario1.Pmin._data[i])

# # actual power output under attack 2
gen_power_attack_scenario2={}  # create an empty dictionary
for i in instance_attack_scenario2.G.value:
    gen_power_attack_scenario2[i]=[]   # create an empty list
    for h in instance_attack_scenario2.T.value:
        gen_power_attack_scenario2[i].append(value(instance_attack_scenario2.p1._data[i, h]) + value(instance_attack_scenario2.status._data[i, h]) * instance_attack_scenario2.Pmin._data[i])


#
# # plot
# plt.figure(figsize=(12,5))
# plt.xlabel('time (hour)')
# plt.ylabel('power (MW)')
# plt.plot(PL_norm,'b',linewidth=3,label='Normal')
# plt.plot(PL_at,'r',linestyle="--",linewidth=3,label='Attack')
# plt.title('Forcasted Load')
# plt.legend()
# plt.show()

# Plot: 2D
plt.figure(figsize=(12,5))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
for i in instance_attack_scenario1.G.value:
    plt.step(scheduling_time, gen_power_attack_scenario1[i], label='Gen {}'.format(i), linewidth=3)
plt.title('UC under Attack Scenario 1')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()

plt.figure(figsize=(12,5))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
for i in instance_attack_scenario2.G.value:
    plt.step(scheduling_time, gen_power_attack_scenario2[i], label='Gen {}'.format(i), linewidth=3)
plt.title('UC under Attack Scenario 2')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()

plt.figure(figsize=(12,5))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
for i in instance_normal.G.value:
    plt.step(scheduling_time, gen_power_normal[i], label='Gen {}'.format(i), linewidth=3)
plt.title('UC under Normal Case')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()
#
# # Plot: 2D
# plt.figure(figsize=(12,5))
# plt.xlabel('time (hour)')
# plt.ylabel('power (MW)')
# for i in range(number_gen):
#     plt.step(iter_hour, gen_power_at[i], label='Gen {}'.format(int(ppc["gen"][i, 0])), linewidth=3)
# plt.title('UC under Cyber Attack')
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
# plt.show()


# # Plot: 3D
# fig = plt.figure(figsize=(12,5))
# ax = fig.add_subplot(111, projection='3d')
# for i in iter_gen:
#     xs = iter_hour
#     ys = gen_power_normal[i]
#     ax.bar(xs, ys, zs=int(ppc["gen"][i,0]), zdir='y', alpha=0.7, label='Gen {}'.format(int(ppc["gen"][i,0])))
# ax.set_xlabel('Hours')
# ax.set_ylabel('Generator Index')
# ax.set_zlabel('Power')
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
# plt.show()
# plt.title('UC under Normal Case')
#
# # Plot: 3D
# fig = plt.figure(figsize=(12,5))
# ax = fig.add_subplot(111, projection='3d')
# for i in iter_gen:
#     xs = iter_hour
#     ys = gen_power_at[i]
#     ax.bar(xs, ys, zs=int(ppc["gen"][i,0]), zdir='y', alpha=0.7, label='Gen {}'.format(int(ppc["gen"][i,0])))
# ax.set_xlabel('Hours')
# ax.set_ylabel('Generator Index')
# ax.set_zlabel('Power')
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
# plt.show()
# plt.title('UC under Cyber Attack')


