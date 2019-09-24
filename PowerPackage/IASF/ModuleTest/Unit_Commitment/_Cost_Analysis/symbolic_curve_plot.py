import sympy as sy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

""" 
Import the studied system data using matpower/pypower format
"""
import case_39bus as case
ppc=case.case39() # dictionary data

# get size of bus, line and generator
number_bus=ppc['bus'].shape[0] # shape returns (row,column)
number_gen=ppc['gen'].shape[0]

# create iterator for bus, generator and horizon
iter_bus=np.arange(0,number_bus)
iter_gen=np.arange(0,number_gen)

def symbolic_curve_plot(ppc, index, p, cost_curve):
    N=len(index)
    xt={}
    yt={}
    for i in range(N):
        xt[i]=np.linspace(ppc["gen"][index[i], 9],ppc["gen"][index[i], 8])
    for i in range(N):
        yt[i]=[]
        for j in range(len(xt[i])):
            yt[i].append(cost_curve[index[i]].subs(p, xt[i][j]))
    return xt,yt,index

p = sy.Symbol('p')
cost_curve={}
cost_curve_diff={}
cost_curve_simple={}
cost_fixed={}
cost_marginal={}
for i in iter_gen:
    cost_curve[i]=ppc["gencost"][i,1]*p**2 + ppc["gencost"][i,2]*p + ppc["gencost"][i,3]
    cost_curve_diff[i]=sy.diff(cost_curve[i],p)
    cost_fixed[i]=cost_curve[i].subs(p, ppc["gen"][i, 9])
    cost_marginal[i]=cost_curve_diff[i].subs(p,(ppc["gen"][i, 8]-ppc["gen"][i, 9])*0.5)
    cost_curve_simple[i]=cost_marginal[i]*(p-ppc["gen"][i, 9])+cost_fixed[i]



aa,bb,cc=symbolic_curve_plot(ppc,[0,1,2],p,cost_curve_simple)

new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

plt.figure(figsize=(7,5))
plt.xlabel('power (MW)')
plt.ylabel('cost ($/h)')
for i in range(len(aa)):
    plt.plot(aa[i],bb[i],label='Gen {}'.format(int(cc[i])),linewidth=3,color=new_colors[cc[i]])
# plt.title('Generator Scheduling')
plt.legend()
plt.show()

