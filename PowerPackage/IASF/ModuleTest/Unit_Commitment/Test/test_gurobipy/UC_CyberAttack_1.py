# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 21:05:37 2018
@author: Yichen Zhang

Test the UC under compromised day-ahead forecasted load
Compromised data are artificially manipulated

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
LoadData=pd.read_csv('P_Load.csv')
LoadData1=pd.read_csv('P_Load1.csv')
PL=LoadData.iloc[:,1]
PL1=LoadData1.iloc[:,1]
PL=PL*0.0025
PL1=PL1*0.0025
H=len(PL)  # Scheduling horizon

plt.figure(figsize=(11,8))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.plot(PL,'b',linewidth=4,label='Normal')
plt.plot(PL1,'r--',linewidth=4,label='Attack')
plt.title('Forcasted Load')
plt.legend()
plt.show()





# Define generator data
Pc_lim=[10,650]
Pc_s1=600 # fixed cost
Pc_s2=8 # marginal cost


Pg_lim=[10, 400]
Pg_s1=80 # fixed cost
Pg_s2=20 # marginal cost



### Under Normal Condition
m = Model("UC")
# Create variables
Uc = m.addVars(H,vtype=GRB.BINARY, name="Uc") # On-off of coal plant
Ug = m.addVars(H,vtype=GRB.BINARY, name="Ug") # On-off of gas plant
Pc = m.addVars(H,name="Pc")
Pg = m.addVars(H,name="Pg")
# Set objective
obj=0
for x in range(H):  # range(3) == [0, 1, 2]
    obj=obj+Uc[x]*Pc_s1+Pc[x]*Pc_s2
for x in range(H):
    obj=obj+Ug[x]*Pg_s1+Pg[x]*Pg_s2
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
for x in range(H):  # range(3) == [0, 1, 2]
    obj=obj+Uc[x]*Pc_s1+Pc[x]*Pc_s2
for x in range(H):
    obj=obj+Ug[x]*Pg_s1+Pg[x]*Pg_s2
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
value=[]
for v in m1.getVars():
   value.append(v.x)
dd1=pd.Series(value)

# get the commitment signal under attack
Ugt=dd1.iloc[0:H]
Uct=dd1.iloc[H:2*H]




plt.figure(figsize=(11,8))
plt.subplot(211)
#plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.step(range(H),dd.iloc[2*H:3*H],'b',label='Coal',linewidth=4)
plt.step(range(H),dd.iloc[3*H:4*H],'r',label='Gas',linewidth=4,linestyle='--')
plt.title('Unit Commitment Under Normal Condition (Gurobipy)')
plt.legend()
plt.show()
plt.subplot(212)
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.step(range(H),dd1.iloc[2*H:3*H],'b',label='Coal',linewidth=4)
plt.step(range(H),dd1.iloc[3*H:4*H],'r',label='Gas',linewidth=4,linestyle='--')
plt.title('Unit Commitment Under Attack (Gurobipy)')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
plt.legend()
plt.show()




