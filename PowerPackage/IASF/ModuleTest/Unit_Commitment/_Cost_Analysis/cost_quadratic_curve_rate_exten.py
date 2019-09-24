import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

x1 = np.linspace(0,2000)

# cost coefficients
coe_1=[0.02533, 25.5472, 24.3891]   # 2.4 to 12
coe_2=[0.00660, 19.0800, 230.9689]  # 25 to 100

cost_1=coe_1[0]*x1**2 + coe_1[1]*x1 + coe_1[2]
cost_2=coe_2[0]*x1**2 + coe_2[1]*x1 + coe_2[2]

plt.figure(figsize=(10,6))
plt.plot(x1,cost_1,'b',linewidth=2,label="a={},b={},c={}".format(coe_1[0],coe_1[1],coe_1[2]))
plt.plot(x1,cost_2,'r',linewidth=2,label="a={},b={},c={}".format(coe_2[0],coe_2[1],coe_2[2]))
plt.axvline(x=100, ymin=0, ymax=0.1,color='b', linestyle='--')
plt.axvline(x=200, ymin=0, ymax=0.1, color='b', linestyle='--')
plt.axvline(x=250, ymin=0, ymax=0.3,color='r', linestyle='--')
plt.axvline(x=500, ymin=0, ymax=0.3,color='r', linestyle='--')
plt.axvline(x=600, ymin=0, ymax=0.8,color='k', linestyle='--')
plt.axvline(x=1800, ymin=0, ymax=0.8,color='k', linestyle='--')
plt.xlabel('Power (MW)')
plt.ylabel('Cost ($/h)')
plt.axis([0,2000,0,160000])
plt.show()
plt.legend()


