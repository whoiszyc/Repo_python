# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 21:05:37 2018
@author: Yichen Zhang

Test the UC under compromised day-ahead forecasted load
Compromised data are generated by BNL algorithm

"""

import pypsa
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)
from gurobipy import *


### Get load data from csv
LoadData=pd.read_csv('Load_normal.csv',header=None)
LoadData1=pd.read_csv('Load_attack.csv',header=None)
PL=LoadData.iloc[:,0]
PL1=LoadData1.iloc[:,0]
PL=PL
PL1=PL1
H=len(PL)  # Scheduling horizon

# Plot
plt.figure(figsize=(11,8))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.plot(PL,'b',linewidth=4,label='Normal')
plt.plot(PL1,'r--',linewidth=4,label='Attack')
plt.title('Forcasted Load')
plt.legend()
plt.show()


# Define generator data
Pn_coal=20000
Pn_gas=10000
Pc_lim=[0.1*Pn_coal,1.0*Pn_coal]
Pc_cost=20
Pg_lim=[0.1*Pn_gas, 1.0*Pn_gas]
Pg_cost=70


### Under Normal Condition
m = Model("UC")
# Create variables
Uc = m.addVars(H,vtype=GRB.BINARY, name="Uc") # On-off of coal plant
Ug = m.addVars(H,vtype=GRB.BINARY, name="Ug") # On-off of gas plant
Pc = m.addVars(H,name="Pc")
Pg = m.addVars(H,name="Pg")
# Set objective
obj=0
for x in range(H):
    obj=obj+Pc[x]*Pc_cost
    print(Pc[x])
for x in range(H):
    obj=obj+Pg[x]*Pg_cost
m.setObjective(obj)  #, GRB.MAXIMIZE
# Add constraint: operating
m.addConstrs((Pc_lim[0]*Uc[i]<=Pc[i] for i in range(H)),name='c1')
m.addConstrs((Pc_lim[1]*Uc[i]>=Pc[i] for i in range(H)),name='c2')
m.addConstrs((Pg_lim[0]*Ug[i]<=Pg[i] for i in range(H)),name='c3')
m.addConstrs((Pg_lim[1]*Ug[i]>=Pg[i] for i in range(H)),name='c4')
# Add constraint: load balancing
m.addConstrs((Pg[i]+Pc[i]==PL.iloc[i] for i in range(H)),name='c5')
# Call solver
m.optimize()
# Record value
value=[]
for v in m.getVars():
   value.append(v.x)
dd=pd.Series(value)



### Under Attack
m1 = Model("UC1")
# Create variables
Uc = m1.addVars(H,vtype=GRB.BINARY, name="Uc") # On-off of coal plant
Ug = m1.addVars(H,vtype=GRB.BINARY, name="Ug") # On-off of gas plant
Pc = m1.addVars(H,name="Pc")
Pg = m1.addVars(H,name="Pg")
# Set objective
obj=0
for x in range(H):
    obj=obj+Pc[x]*Pc_cost
    print(Pc[x])
for x in range(H):
    obj=obj+Pg[x]*Pg_cost
m1.setObjective(obj)  #, GRB.MAXIMIZE
# Add constraint: operating
m1.addConstrs((Pc_lim[0]*Uc[i]<=Pc[i] for i in range(H)),name='c1')
m1.addConstrs((Pc_lim[1]*Uc[i]>=Pc[i] for i in range(H)),name='c2')
m1.addConstrs((Pg_lim[0]*Ug[i]<=Pg[i] for i in range(H)),name='c3')
m1.addConstrs((Pg_lim[1]*Ug[i]>=Pg[i] for i in range(H)),name='c4')
# Add constraint: load balancing
m1.addConstrs((Pg[i]+Pc[i]==PL1.iloc[i] for i in range(H)),name='c5')
# Call solver
m1.optimize()
# Record value
value1=[]
for v in m1.getVars():
   value1.append(v.x)
dd1=pd.Series(value1)



plt.figure(figsize=(11,8))
plt.subplot(211)
#plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.step(range(H-1),dd.iloc[2*H+1:3*H],'b',label='Coal',linewidth=4)
plt.step(range(H-1),dd.iloc[3*H+1:4*H],'r',label='Gas',linewidth=4,linestyle='--')
plt.title('Unit Commitment Under Normal Condition (Gurobipy)')
plt.legend()
plt.show()
plt.subplot(212)
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.step(range(H-1),dd1.iloc[2*H+1:3*H],'b',label='Coal',linewidth=4)
plt.step(range(H-1),dd1.iloc[3*H+1:4*H],'r',label='Gas',linewidth=4,linestyle='--')
plt.title('Unit Commitment Under Attack (Gurobipy)')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
plt.legend()
plt.show()



plt.figure(figsize=(15,6))
plt.ylabel('power (MW)')
plt.step(range(H-1),dd.iloc[2*H+1:3*H],'b',label='Normal',linewidth=4)
plt.step(range(H-1),dd1.iloc[2*H+1:3*H],'r',label='Compromised',linewidth=4,linestyle='--')
plt.legend()
plt.title('Coal Plant Dispatch')
plt.show()