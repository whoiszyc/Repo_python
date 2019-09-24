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
        'size'   : 20}
matplotlib.rc('font', **font)


# ### Get load data from csv
# LoadData=pd.read_csv('P_Load.csv')
# LoadData1=pd.read_csv('P_Load1.csv')
# PL=LoadData.iloc[:,1]
# PL1=LoadData1.iloc[:,1]
# H=len(PL)  # Scheduling horizon

# ### Test load data
PL=[50, 80, 90, 70]
PL1=[1, 2, 3, 2]
H=len(PL)  # Scheduling horizon

# Plot
plt.figure(figsize=(15,7))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.plot(PL,'b',linewidth=4,label='Case 1')
plt.plot(PL1,'r--',linewidth=4,label='Case 2')
plt.title('Forcasted Load')
plt.legend()
plt.show()


### Minimum part load demonstration under normal condition
#In final hour load goes below part-load limit of coal gen (40%), forcing gas to commit.
nu = pypsa.Network()
nu.set_snapshots(range(H))
# Single bus system
nu.add("Bus","bus")
# The unit of component input is MW
nu.add("Generator","coal",bus="bus",
       committable=True,
       capital_cost=50,
       marginal_cost=20,
       p_min_pu=0,
       p_max_pu=1.0,
       p_nom=100)
nu.add("Generator","gas",bus="bus",
       committable=True,
       capital_cost=50,
       marginal_cost=70,
       p_min_pu=0,
       p_max_pu=1.0,
       p_nom=100)
nu.add("Load","load",bus="bus",p_set=PL)
nu.lopf(nu.snapshots,solver_name="glpk")
print(nu.generators_t.status)
print(nu.generators_t.p)



### Minimum part load demonstration under attack
#In final hour load goes below part-load limit of coal gen (40%), forcing gas to commit.
nu1 = pypsa.Network()
nu1.set_snapshots(range(H))
# Single bus system
nu1.add("Bus","bus")
# The unit of component input is MW
nu1.add("Generator","coal",bus="bus",
       committable=True,
       marginal_cost=20,
       p_min_pu=0,
       p_max_pu=1.0,
       p_nom=100)
nu1.add("Generator","gas",bus="bus",
       committable=True,
       marginal_cost=70,
       p_min_pu=0,
       p_max_pu=1.0,
       p_nom=100)
nu1.add("Load","load",bus="bus",p_set=PL1)
nu1.lopf(nu1.snapshots,solver_name="glpk")
print(nu1.generators_t.status)
print(nu1.generators_t.p)


"""
plt.figure(figsize=(11,8))
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.plot(nu.generators_t.p.iloc[:,0],'b',label='coal',linewidth=4)
plt.plot(nu.generators_t.p.iloc[:,1],'r',label='gas',linewidth=4)
plt.title('Generator Scheduling')
plt.legend()
plt.show()
"""

plt.figure(figsize=(15,7))
plt.subplot(211)
#plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.step(nu.generators_t.p.iloc[:,0],'b',label='Coal',linewidth=4)
plt.step(nu.generators_t.p.iloc[:,1],'r',label='Gas',linewidth=4,linestyle='--')
plt.title('Unit Commitment Under Normal Condition')
plt.legend()
plt.show()
plt.subplot(212)
plt.xlabel('time (hour)')
plt.ylabel('power (MW)')
plt.step(nu1.generators_t.p.iloc[:,0],'b',label='Coal',linewidth=4)
plt.step(nu1.generators_t.p.iloc[:,1],'r',label='Gas',linewidth=4,linestyle='--')
plt.title('Unit Commitment Under Attack')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
plt.legend()
plt.show()
