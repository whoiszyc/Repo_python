
# Example from "Optimizing the spinning reserve requirements"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

x1 = np.linspace(2.4,12)
x2 = np.linspace(70,200)
x3 = np.linspace(140,350)
x=np.linspace(0,350)

# cost coefficients
# All plants are Oil/Steam with different capacity
coef={}
coef['A']=[0.02533, 25.5472, 24.3891]   # Group A, capacity=12 MW
coef['F']=[0.00260, 23.0000, 259.1310]  # Group F, capacity=197 MW
coef['G']=[0.00153, 10.8616, 177.0575]  # Group F, capacity=350 MW

# cost curve
cost_1= coef['A'][0]*x1**2 + coef['A'][1]*x1 + coef['A'][2]
cost_2= coef['F'][0]*x2**2 + coef['F'][1]*x2 + coef['F'][2]
cost_3= coef['G'][0]*x3**2 + coef['G'][1]*x3 + coef['G'][2]

# extended curve
cost_1_e= coef['A'][0]*x**2 + coef['A'][1]*x + coef['A'][2]
cost_2_e= coef['F'][0]*x**2 + coef['F'][1]*x + coef['F'][2]
cost_3_e= coef['G'][0]*x**2 + coef['G'][1]*x + coef['G'][2]

plt.figure(figsize=(10,5))
plt.plot(x,cost_1_e,'b',linewidth=2,linestyle='--')
plt.plot(x,cost_2_e,'r',linewidth=2,linestyle='--')
plt.plot(x,cost_3_e,'g',linewidth=2,linestyle='--')
plt.plot(x1,cost_1,'b',linewidth=3,label="a={},b={},c={}, Cap: 2.4-12 MW".format(coef['A'][0],coef['A'][1],coef['A'][2]))
plt.plot(x2,cost_2,'r',linewidth=3,label="a={},b={},c={}, Cap: 70-200 MW".format(coef['F'][0],coef['F'][1],coef['F'][2]))
plt.plot(x3,cost_3,'g',linewidth=3,label="a={},b={},c={}, Cap: 140-350 MW".format(coef['G'][0],coef['G'][1],coef['G'][2]))
# plt.axvline(x=100, ymin=0, ymax=0.1,color='b', linestyle='--')
# plt.axvline(x=200, ymin=0, ymax=0.1, color='b', linestyle='--')
# plt.axvline(x=70, ymin=0, ymax=0.5, color='r', linestyle='--')
# plt.axvline(x=200, ymin=0, ymax=0.5,color='r', linestyle='--')
# plt.axvline(x=140, ymin=0, ymax=0.5,color='g', linestyle='--')
# plt.axvline(x=350, ymin=0, ymax=0.5,color='g', linestyle='--')
plt.xlabel('Power (MW)')
plt.ylabel('Cost ($/h)')
# plt.axis([0,2000,0,160000])
plt.show()
plt.legend()
plt.title('Oil/Steam Plant with Different Capacities')


