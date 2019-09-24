import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


from ModuleTest.backup.control_backup import dyn_LFC, m

# assume the data is obtained from the economic dispatch
number_area = 3

# center of inertia in each area
M = [12, 10, 10]

# tie-line synchronizing coefficient
Ps_0_1 = 0.7
Ps_0_2 = 0.8
Ps_1_2 = 5
Ps = OrderedDict()
Ps[0] = OrderedDict()
Ps[0][0] = 0
Ps[0][1] = Ps_0_1
Ps[0][2] = Ps_0_2
Ps[1] = OrderedDict()
Ps[1][0] = Ps[0][1]
Ps[1][1] = 0
Ps[1][2] = Ps_1_2
Ps[2] = OrderedDict()
Ps[2][0] = Ps[0][2]
Ps[2][1] = Ps[1][2]
Ps[2][2] = 0


m = m()
m.add_inertia(M)
m.add_tieline_syn(Ps)



# solve frequency dynamic

# define objective
LFC = dyn_LFC()

# define symbolic model
LFC.symbolic_model_define(0, m, 1)
LFC.symbolic_model_define(1, m, 1)
LFC.symbolic_model_define(2, m, 1)

# get state space
LFC.get_vector()
LFC.get_state_space()


# define saturation
LFC.saturation_default(-100000, 100000)
# LFC.saturation_set('Pm_0',-1, 0.1)
# LFC.saturation_set('Pm_1',-1, 0.1)
# LFC.saturation_set('Pm_2',-1, 0.1)


# give initial condition
LFC.x0 = [0] * LFC.number_state

# define time series
LFC.t = np.linspace(0, 200, 2000)

# define input
LFC.input_default()
LFC.input_set('Pl_0', 5, 200, 0.35)

# solve problem
LFC.ode_euler()



plt.figure(figsize=(8,4))
plt.plot(LFC.t, LFC.get_state_data('f_0')*60, 'b--', linewidth=2, label='Area 1')
plt.plot(LFC.t, LFC.get_state_data('f_1')*60, 'g--', linewidth=2, label='Area 2')
plt.plot(LFC.t, LFC.get_state_data('f_2')*60, 'r--', linewidth=2, label='Area 3')
plt.xlabel('Time (second)')
plt.ylabel('Frequency (Hz)')
plt.legend(loc='best')
plt.title('Frequency in Each Area')
plt.savefig('frequency.png')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(LFC.t, LFC.get_state_data('Ptl_0_1'), 'b--', linewidth=2, label='Tie-line flow between area 1 and 2')
plt.plot(LFC.t, LFC.get_state_data('Ptl_0_2'), 'g--', linewidth=2, label='Tie-line flow between area 1 and 3')
plt.plot(LFC.t, LFC.get_state_data('Ptl_1_2'), 'r--', linewidth=2, label='Tie-line flow between area 2 and 3')
plt.xlabel('Time (second)')
plt.ylabel('Tie-line Power (per unit)')
plt.legend(loc='best')
plt.title('Tie-line Power Flow')
plt.savefig('tie_line.png')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(LFC.t, LFC.get_state_data('Pm_0'), 'b--', linewidth=2, label='Area 1')
plt.plot(LFC.t, LFC.get_state_data('Pm_1'), 'g--', linewidth=2, label='Area 2')
plt.plot(LFC.t, LFC.get_state_data('Pm_2'), 'r--', linewidth=2, label='Area 3')
plt.xlabel('Time (second)')
plt.ylabel('Mechanical Power Variation (per unit)')
plt.legend(loc='best')
plt.title('Mechanical Power Variation in Each Area')
plt.savefig('power.png')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(LFC.t, LFC.get_state_data('Pv_0'), 'b--', linewidth=2, label='Area 1')
plt.plot(LFC.t, LFC.get_state_data('Pv_1'), 'g--', linewidth=2, label='Area 2')
plt.plot(LFC.t, LFC.get_state_data('Pv_2'), 'r--', linewidth=2, label='Area 3')
plt.xlabel('Time (second)')
plt.ylabel('Valve Position Variation (per unit)')
plt.legend(loc='best')
plt.title('Valve Position Variation in Each Area')
plt.savefig('valve.png')
plt.show()

