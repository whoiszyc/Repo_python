import sympy as sy
import pandas as pd
import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from collections import OrderedDict



def iter_remove_index(end, a):
    # create a iterator from 0 to end without a given a

    # prepare a list for iteration without i
    iter_j = [j for j in range(end)]
    #get the index in iter_j where the element equals to a
    index_equal_a = iter_j.index(a)
    # delete the element
    iter_j.remove(iter_j[index_equal_a])

    return iter_j


class m:
    # parameters that will not change based on operating condition
    number = 3
    Tb = [0.12, 0.10, 0.15, 0.10]
    Tg = [0.50, 0.45, 0.60, 0.50]
    D = [1.0, 1.0, 1.0, 1.0]
    R = [0.05, 0.05, 0.05, 0.05]
    K = [0.05, 0.07, 0.08, 0.15]
    f = []
    pm = []
    pv = []
    pl = []
    er = []
    for i in range(number):
        f.append(sy.symbols('f_{}'.format(i)))
        pm.append(sy.symbols('Pm_{}'.format(i)))
        pv.append(sy.symbols('Pv_{}'.format(i)))
        pl.append(sy.symbols('Pl_{}'.format(i)))
        er.append(sy.symbols('E_{}'.format(i)))

    ptl = OrderedDict()
    for i in range(number):
        ptl[i] = []
        for j in range(number):
             ptl[i].append(sy.symbols('Ptl_{}_{}'.format(i, j)))

    # center of inertia frequency
    def add_inertia(self, Mcoi):
        # Mcoi should be a list, containing the center of inertia in each area
        self.M = Mcoi

    # tie-line synchronizing coefficient
    def add_tieline_syn(self, Pss):
        self.Ps = [[] for i in range(self.number)]
        for i in range(self.number):
            for j in range(self.number):
                self.Ps[i].append(2*math.pi*Pss[i][j])



class dyn_LFC:
    # ordered dictionary for flexible model definition
    dfdt = OrderedDict()
    x = OrderedDict()
    u = OrderedDict()

    # vectorize dynamic model, state and input for state space
    dfdt_vector = []
    x_vector = []
    u_vector = []
    # define name for easy to search index by name string
    x_vector_name = []
    u_vector_name = []

    # time series data from dynamic simulation
    t = []
    u_t = OrderedDict()
    x_0 = []
    x_t = OrderedDict()

    # state saturation
    x_sat = OrderedDict()


    def symbolic_model_define(self, i, m, tie_line_flag):
        # define the symbolic dynamic model, state and control vector of area i
        # input: i - area index, m - class for parameter of LFC
        # input: tie_line_flag = 1 keep tie-line flow as scheduled, 0 otherwise
        self.dfdt[i] = []
        self.x[i] = []
        self.u[i] = []

        # ------------------dynamic equation and states--------------------
        # swing dynamics
        self.dfdt[i].append(1 / m.M[i] * (m.pm[i] - m.pl[i] - m.D[i] * m.f[i] - sum(p for p in m.ptl[i])))
        self.x[i].append(m.f[i])
        # tie-line power flow
        for j in iter_remove_index(m.number, i):
            if m.Ps[i][j] == 0:
                pass
            else:
                self.dfdt[i].append(m.Ps[i][j] * (m.f[i] - m.f[j]))
                self.x[i].append(m.ptl[i][j])
        # turbine
        self.dfdt[i].append(1 / m.Tb[i] * (m.pv[i] - m.pm[i]))
        self.x[i].append(m.pm[i])
        # governor
        self.dfdt[i].append(1 / m.Tg[i] * (-m.pv[i] - m.K[i] * m.er[i] - 1 / m.R[i] * m.f[i]))
        self.x[i].append(m.pv[i])
        # area error
        self.dfdt[i].append(tie_line_flag * sum(p for p in m.ptl[i]) + (1 / m.R[i] + m.D[i]) * m.f[i])
        self.x[i].append(m.er[i])

        # --------------- control input ------------
        self.u[i].append(m.pl[i])



    def get_vector(self):
        # put dynamic system in one list in order
        for i in range(len(self.dfdt)):
            for j in range(len(self.dfdt[i])):
                self.dfdt_vector.append(self.dfdt[i][j])

        # put state vector in one list in order
        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                self.x_vector.append(self.x[i][j])
                self.x_vector_name.append(str(self.x[i][j]))

        # put control vector in one list in order
        for i in range(len(self.u)):
            for j in range(len(self.u[i])):
                self.u_vector.append(self.u[i][j])
                self.u_vector_name.append(str(self.u[i][j]))



    def get_state_space(self):
        self.number_state = len(self.x_vector)
        self.number_input = len(self.u_vector)

        self.A = [[] for i in range(self.number_state)]
        self.B = [[] for i in range(self.number_state)]

        for i in range(self.number_state):
            for j in range(self.number_state):
                self.A[i].append(float(sy.diff(self.dfdt_vector[i], self.x_vector[j])))

        for i in range(self.number_state):
            for j in range(self.number_input):
                self.B[i].append(float(sy.diff(self.dfdt_vector[i], self.u_vector[j])))



    def saturation_default(self, default_lb, default_ub):
        for i in range(self.number_state):
            self.x_sat[i] = {}
            self.x_sat[i]['lb'] = default_lb
            self.x_sat[i]['ub'] = default_ub

    def saturation_set(self, state_name, lb, ub):
        state_index = self.x_vector_name.index(state_name)
        self.x_vector_name.index(state_name)
        self.x_sat[state_index]['lb'] = lb
        self.x_sat[state_index]['ub'] = ub


    def input_default(self):
        # given the simulation time series, default input would be a zero vector
        self.number_step = len(self.t)
        for i in range(self.number_input):
            self.u_t[i] = [0] *self.number_step

    def input_set(self, input_name, f_time, t_time, value):
        # get index from name, which is a string type
        input_index = self.u_vector_name.index(input_name)

        # change value with respect to corresponding time interval
        for i in range(self.number_step):
            if self.t[i] >= f_time and self.t[i]< t_time:
                self.u_t[input_index][i] = self.u_t[input_index][i] + value


    def ode_euler(self):

        # initialize x_t dictionary
        number_state = len(self.x_vector)
        for i in range(self.number_state):
            self.x_t[i] = []
            # pass initial condition to x_t
            self.x_t[i].append(self.x0[i])

        # begin simulation
        h = self.t[1] - self.t[0]

        # simple euler integration
        for t in range(1, self.number_step):
            for i in range(number_state):
                x_i_t = self.x_t[i][t-1] + h*sum(self.A[i][j]*self.x_t[j][t-1] for j in range(self.number_state)) \
                        + h*sum(self.B[i][j]*self.u_t[j][t-1] for j in range(self.number_input))

                # check saturation
                if (self.x_sat[i]['lb'] <= x_i_t) and (self.x_sat[i]['ub'] >= x_i_t):
                    self.x_t[i].append(x_i_t)
                elif self.x_sat[i]['lb'] >= x_i_t:
                    self.x_t[i].append(self.x_sat[i]['lb'])
                elif self.x_sat[i]['ub'] <= x_i_t:
                    self.x_t[i].append(self.x_sat[i]['ub'])


    # get simulation data
    def get_state_data(self, name):
        state_index = self.x_vector_name.index(name)
        return pd.Series(self.x_t[state_index])





# assume the data is obtained from the economic dispatch
number_area = 3

# center of inertia in each area
M = [12, 10, 10]

# tie-line synchronizing coefficient
Ps_0_1 = 0.7
Ps_0_2 = 0.8
Ps_1_2 = 1
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
LFC.saturation_set('Pm_0',-1, 0.1)
LFC.saturation_set('Pm_1',-1, 0.1)
LFC.saturation_set('Pm_2',-1, 0.1)


# give initial condition
LFC.x0 = [0] * LFC.number_state

# define time series
LFC.t = np.linspace(0, 50, 2000)

# define input
LFC.input_default()
LFC.input_set('Pl_0', 5, 50, 0.35)

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
plt.ylabel('Mechanical Power Variation (per unit)')
plt.legend(loc='best')
plt.title('Valve Position Variation in Each Area')
plt.savefig('valve.png')
plt.show()

