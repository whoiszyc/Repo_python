import sympy as sy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


xs = sy.Symbol('xs')
xt1=np.linspace(100,400)
xt2=np.linspace(1000,3000)
xt3=np.linspace(100,3000)

# cost coefficients
coef=[0.00660, 19.0800, 230.9689]
cost= coef[0] * xs ** 2 + coef[1] * xs + coef[2]


cost_t1=[]
cost_t2=[]
cost_t3=[]

for i in range(len(xt1)):
    cost_t1.append(cost.subs(xs,xt1[i]))

for i in range(len(xt2)):
    cost_t2.append(cost.subs(xs,xt2[i]))

for i in range(len(xt3)):
    cost_t3.append(cost.subs(xs,xt3[i]))

# plot
plt.figure(figsize=(8,5))
plt.xlabel('power (MW)')
plt.ylabel('cost ($/h)')
plt.plot(xt1,cost_t1,'b',linewidth=3,label="Orginal")
plt.plot(xt2,cost_t2,'r',linewidth=3,label="Expansion")
plt.plot(xt3,cost_t3,'b',linewidth=1,linestyle='--')
# plt.axvline(x=1000, ymin=0, ymax=0.4,color='k', linestyle='--')
plt.show()
plt.legend()



