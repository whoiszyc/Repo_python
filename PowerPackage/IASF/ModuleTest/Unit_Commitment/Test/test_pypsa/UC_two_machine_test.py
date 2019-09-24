# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:59:34 2018

@author: Yichen Zhang
"""

import pypsa
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)



### Get load data from csv
LoadData=pd.read_csv('P_Load.csv')
PL=LoadData.iloc[:,1]
H=len(PL)  # Scheduling horizon

# Plot
plt.figure(figsize=(10,7))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.plot(PL,'b',linewidth=4,label='Normal')
plt.title('Forcasted Load')
plt.legend()
plt.show()




### Minimum part load demonstration under normal condition
#In final hour load goes below part-load limit of coal gen (40%), forcing gas to commit.
nu = pypsa.Network()
nu.set_snapshots(range(H))

# Single bus system
nu.add("Bus","ISO_NE")

# add generators
for i in range(10):
    nu.add("Generator","Gen {}".format(i),
           bus="ISO_NE",
           committable=True,
           p_min_pu=8,
           p_max_pu=20,
           marginal_cost=20,
           p_nom=100)

nu.add("Load","load",bus="ISO_NE",p_set=PL)

nu.lopf(nu.snapshots,solver_name="glpk")
print(nu.generators_t.status)
print(nu.generators_t.p)



plt.figure(figsize=(11,8))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
for i in range(10):
    plt.plot(nu.generators_t.p.iloc[:,i],label='Gen {}'.format(i),linewidth=4)
plt.title('Generator Scheduling')
plt.legend()
plt.show()


#plt.figure(figsize=(10,7))
#plt.subplot(211)
##plt.xlabel('time (hour)')
#plt.ylabel('power (MW)')
#plt.step(nu.generators_t.p.iloc[:,0],'b',label='Coal',linewidth=4)
#plt.step(nu.generators_t.p.iloc[:,1],'r',label='Gas',linewidth=4,linestyle='--')
#plt.title('Unit Commitment Under Normal Condition')
#plt.legend()
#plt.show()
#plt.subplot(212)
#plt.xlabel('time (hour)')
#plt.ylabel('power (MW)')
#plt.step(nu1.generators_t.p.iloc[:,0],'b',label='Coal',linewidth=4)
##plt.step(nu1.generators_t.p.iloc[:,1],'r',label='Gas',linewidth=4,linestyle='--')
#plt.title('Unit Commitment Under Attack')
#plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
#plt.legend()
#plt.show()