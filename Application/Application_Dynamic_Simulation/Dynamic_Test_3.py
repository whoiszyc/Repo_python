import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import numpy as np
from scipy.integrate import odeint


def deriv(A, t, Ab):
    return np.dot(Ab, A)


Ab = np.array([[-0.25,    0,    0],
               [ 0.25, -0.2,    0],
               [    0,  0.2, -0.1]])

time = np.linspace(0, 25, 101)
A0 = np.array([10, 20, 30])

MA = odeint(deriv, A0, time, args=(Ab,))

# plt.figure(1)
# plt.plot(t1,y1*du,'b--',linewidth=3,label='Transfer Fcn')
# plt.plot(t2,y2*du,'g:',linewidth=2,label='State Space')
# plt.plot(t3,y3,'r-',linewidth=1,label='ODE Integrator')
# y_ss = Kp * du
# plt.plot([0,max(t1)],[y_ss,y_ss],'k:')
# plt.xlim([0,max(t1)])
# plt.xlabel('Time')
# plt.ylabel('Response (y)')
# plt.legend(loc='best')
# plt.savefig('2nd_order.png')
# plt.show()

