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
PL=PL*0.25
PL1=PL1*0.25
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
Pc_lim=[0.1*10000,1.0*10000]
Pc_mc=20 # marginal cost (mc)
Pc_sc=0 #start-up cost (sc, or fixed cost)

Pg_lim=[0.1*5000, 1.0*5000]
Pg_mc=70 # marginal cost (mc)
Pg_sc=0 #start-up cost (sc, or fixed cost)



### Under Normal Condition
m = Model("UC")
# Create variables
Uc = m.addVars(H,vtype=GRB.BINARY, name="Uc") # On-off of coal plant
Sc = m.addVars(H,vtype=GRB.BINARY, name="Sc") # On-off of coal plant
Ug = m.addVars(H,vtype=GRB.BINARY, name="Ug") # On-off of gas plant
Sg = m.addVars(H,vtype=GRB.BINARY, name="Sg") # On-off of gas plant
Pc = m.addVars(H,name="Pc")
Pg = m.addVars(H,name="Pg")
# Set objective
obj=0
for x in range(H):  # range(3) == [0, 1, 2]
    obj=obj+Pc[x]*Pc_mc+Sc[x]*Pc_sc
    print(Pc[x])
for x in range(H):
    obj=obj+Pg[x]*Pg_mc+Sg[x]*Pg_sc
m.setObjective(obj)  #, GRB.MAXIMIZE
# Add constraint: operating
m.addConstrs((Pc_lim[0]*Uc[i]<=Pc[i] for i in range(H)),name='c1')
m.addConstrs((Pc_lim[1]*Uc[i]>=Pc[i] for i in range(H)),name='c2')
m.addConstrs((Pg_lim[0]*Ug[i]<=Pg[i] for i in range(H)),name='c3')
m.addConstrs((Pg_lim[1]*Ug[i]>=Pg[i] for i in range(H)),name='c4')
# Add constraint: load balancing
m.addConstrs((Pg[i]+Pc[i]==PL.iloc[i] for i in range(H)),name='c5')
# Add constraints: start-up cost
m.addConstrs((Sc[i]>=0 for i in range(H)), name='start-up1-1')
m.addConstrs((Sc[i]>=Uc[i+1]-Uc[i] for i in range(H-1)), name='start-up1-2')
m.addConstr((Sc[0]>=Uc[0]-0), name='start-up1-3')
m.addConstrs((Sg[i]>=0 for i in range(H)), name='start-up2-1')
m.addConstrs((Sg[i]>=Ug[i+1]-Ug[i] for i in range(H-1)), name='start-up2-2')
m.addConstr((Sg[0]>=Ug[0]-0), name='start-up1-3')
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
Sc = m1.addVars(H,vtype=GRB.BINARY, name="Sc") # On-off of coal plant
Ug = m1.addVars(H,vtype=GRB.BINARY, name="Ug") # On-off of gas plant
Sg = m1.addVars(H,vtype=GRB.BINARY, name="Sg") # On-off of gas plant
Pc = m1.addVars(H,name="Pc")
Pg = m1.addVars(H,name="Pg")
# Set objective
obj=0
for x in range(H):
    obj=obj+Pc[x]*Pc_mc+Sc[x]*Pc_sc
    print(Pc[x])
for x in range(H):
    obj=obj+Pg[x]*Pg_mc+Sg[x]*Pg_sc
m1.setObjective(obj)  #, GRB.MAXIMIZE
# Add constraint: operating
m1.addConstrs((Pc_lim[0]*Uc[i]<=Pc[i] for i in range(H)),name='c1')
m1.addConstrs((Pc_lim[1]*Uc[i]>=Pc[i] for i in range(H)),name='c2')
m1.addConstrs((Pg_lim[0]*Ug[i]<=Pg[i] for i in range(H)),name='c3')
m1.addConstrs((Pg_lim[1]*Ug[i]>=Pg[i] for i in range(H)),name='c4')
# Add constraint: load balancing
m1.addConstrs((Pg[i]+Pc[i]==PL1.iloc[i] for i in range(H)),name='c5')
# Add constraints: start-up cost
m1.addConstrs((Sc[i]>=0 for i in range(H)), name='start-up1-1')
m1.addConstrs((Sc[i]>=Uc[i+1]-Uc[i] for i in range(H-1)), name='start-up1-2')
m1.addConstr((Sc[0]>=Uc[0]-0), name='start-up1-3')
m1.addConstrs((Sg[i]>=0 for i in range(H)), name='start-up2-1')
m1.addConstrs((Sg[i]>=Ug[i+1]-Ug[i] for i in range(H-1)), name='start-up2-2')
m1.addConstr((Sg[0]>=Uc[0]-0), name='start-up2-3')
# Call solver
m1.optimize()
# Record value
value=[]
for v in m.getVars():
   value.append(v.x)
dd1=pd.Series(value)







plt.figure(figsize=(11,8))
plt.subplot(211)
#plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.step(range(H),dd.iloc[4*H:5*H],'b',label='Coal',linewidth=4)
plt.step(range(H),dd.iloc[5*H:6*H],'r',label='Gas',linewidth=4,linestyle='--')
plt.title('Unit Commitment Under Normal Condition (Gurobipy)')
plt.legend()
plt.show()
plt.subplot(212)
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.step(range(H),dd1.iloc[4*H:5*H],'b',label='Coal',linewidth=4)
plt.step(range(H),dd1.iloc[5*H:6*H],'r',label='Gas',linewidth=4,linestyle='--')
plt.title('Unit Commitment Under Attack (Gurobipy)')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
plt.legend()
plt.show()




