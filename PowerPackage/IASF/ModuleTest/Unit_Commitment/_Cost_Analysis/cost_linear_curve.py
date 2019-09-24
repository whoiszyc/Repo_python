import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

# coal plant
mc1=18; n1=210
# Gas plant
mc2=40; n2=120

x = np.linspace(0,100)

plt.figure(figsize=(7,6))
plt.plot(x,mc1*x+n1,'b',label='Coal',linewidth=2)
plt.plot(x,mc2*x+n2,'r',label='Gas',linewidth=2)
plt.xlabel('Power (MW)')
plt.ylabel('Cost ($/h)')
plt.show()
plt.legend()