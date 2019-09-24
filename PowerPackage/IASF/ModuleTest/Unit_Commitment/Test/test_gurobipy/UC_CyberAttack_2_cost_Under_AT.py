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
PL2=LoadData2.iloc[:,1]
PL=PL*0.03
PL2=PL2*0.03
H=len(PL)  # Scheduling horizon

plt.figure(figsize=(11,8))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.plot(PL,'b',linewidth=4,label='Normal')
plt.plot(PL2,'r--',linewidth=4,label='Attack')
plt.title('Forcasted Load')
plt.legend()
plt.show()





# Define generator data
Pc_lim=[100,650]
Pc_s1=600 # fixed cost
Pc_s2=8 # marginal cost


Pg_lim=[100, 400]
Pg_s1=80 # fixed cost
Pg_s2=20 # marginal cost

cc=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
gg=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
### Under Attack
m3 = Model("UC1")
# Create variables
Pc = m3.addVars(H,name="Pc")
Pg = m3.addVars(H,name="Pg")
# Set objective
obj=0
for x in range(H):  # range(3) == [0, 1, 2]
    obj=obj+cc[x]*Pc_s1+Pc[x]*Pc_s2
for x in range(H):
    obj=obj+gg[x]*Pg_s1+Pg[x]*Pg_s2
m3.setObjective(obj)  #, GRB.MAXIMIZE
# Add constraint: operating
m3.addConstrs((Pc_lim[0]*cc[i]<=Pc[i] for i in range(H)),name='c1')
m3.addConstrs((Pc_lim[1]*cc[i]>=Pc[i] for i in range(H)),name='c2')
m3.addConstrs((Pg_lim[0]*gg[i]<=Pg[i] for i in range(H)),name='c3')
m3.addConstrs((Pg_lim[1]*gg[i]>=Pg[i] for i in range(H)),name='c4')
# Add constraint: load balancing
m3.addConstrs((Pg[i]+Pc[i]==PL.iloc[i] for i in range(H)),name='c5')
# Call solver
m3.optimize()
# Record value
value=[]
for v in m3.getVars():
   value.append(v.x)
dd2=pd.Series(value)








