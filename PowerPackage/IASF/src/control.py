import sympy as sy
import pandas as pd
import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from collections import OrderedDict


from .utils import iter_remove_index
from. general import PowerSystemProblem

class FrequencyControlProblem(PowerSystemProblem):
    """
    Define load frequency control problem by areas (balancing authorities)
    The object contains model and numerical integration method.
    The frequency control problem is dependent on the economic dispatch (ED) results.
    The ED will determine three factors:
        (1) Tie-line synchronizing coefficient
        (2) Headroom of generator turbine-governor
        (3) Center of inertia for area i
    The frequency control problem is solved in terms of seconds.
    """

    def __init__(self, LFC_para, instant_time=None, time_series=None):
        """
        Define data structure
        Define symbolic states
        Input load frequency control parameters and define area indices and iterator
        Input time instant at an economic dispatch result as the starting point of the load frequency control
        The format of the time instant should be 'hh:mm' in 24 hour clock convention.
        """

        # ordered dictionary for flexible model definition
        self.dfdt = OrderedDict()
        self.x = OrderedDict()
        self.u = OrderedDict()

        # vectorize dynamic model, state and input for state space
        self.dfdt_vector = []
        self.x_vector = []
        self.u_vector = []

        # define name for easy to search index by name string
        self.x_vector_name = []
        self.u_vector_name = []

        # time series data from dynamic simulation
        self.t = []
        self.u_t = OrderedDict()
        self.x_0 = []
        self.x_t = OrderedDict()

        # state saturation
        self.x_sat = OrderedDict()

        # area number and iterator
        self.number_area = len(LFC_para.D)
        self.iter_area = range(0, self.number_area)

        # define symbolic system states
        self.f = []      # frequency
        self.pm = []     # mechanical power
        self.pv = []     # valve position
        self.pl = []     # load variation (disturbance)
        self.er = []     # area control error (ACE)
        for i in self.iter_area:
            self.f.append(sy.symbols('f_{}'.format(i)))
            self.pm.append(sy.symbols('Pm_{}'.format(i)))
            self.pv.append(sy.symbols('Pv_{}'.format(i)))
            self.pl.append(sy.symbols('Pl_{}'.format(i)))
            self.er.append(sy.symbols('E_{}'.format(i)))

        self.ptl = OrderedDict() # tie-line flow
        for i in self.iter_area:
            self.ptl[i] = []
            for j in self.iter_area:
                 self.ptl[i].append(sy.symbols('Ptl_{}_{}'.format(i, j)))

        # define time instant at an economic dispatch result as the starting point of the load frequency control
        self.instant_time = instant_time

        # define the load frequency control parameters attributes and pass the value to instance if given in initialization
        self.T_turbine = LFC_para.T_turbine
        self.T_governor = LFC_para.T_governor
        self.damp = LFC_para.D
        self.droop = LFC_para.R
        self.gain = LFC_para.K

        # define the "time series" attribute and pass the parameter to the instance if given in initialization
        self.time_series = time_series



    def locate_index(self):
        # convert instant format from string to numerical number in minutes
        start_hour, start_minute = self.instant_time.split(':')
        time_start_min = int(start_hour) * 60 + int(start_minute)

        # find the closest time in economic dispatch and its index
        time_distance = []
        for value in self.time_series.values():
            time_distance.append(abs(value - time_start_min))
        self.instant_index = np.argmin(time_distance)


    def compute_area_generation(self, economic_dispatch, multi_instant_indices=None):
        """
        split generator into areas based on network information
        calculate center of inertia for each area using given time instant and economic dispatch results
        generator is de-committed when its output is zero due to the minimum output constraint
        """

        # define a dictionary to store generator and bus in different areas
        self.area_gen = {}
        self.area_bus = {}

        # define arttribute for area inertia
        self.area_inertia = []
        self.area_base = []
        self.area_headroom = []

        # loop bus data
        for i in self.iter_area:
            self.area_gen[i] = []
            self.area_bus[i] = []

        for i in self.iter_bus:
            current_bus_area = self.ppc['bus'][i, self.idx.BUS_AREA] - 1 # convert name index to array index
            for k in self.iter_area:
                if k == current_bus_area:
                    self.area_bus[k].append(i)
                    try:
                        self.area_gen[k].append(self.bus_type_gen['bus'].index(i))
                    except ValueError:
                        # Then, the current bus is not a generator bus. We proceed.
                        pass

        # loop all area for computing center of inertia
        for i in self.iter_area:
            # loop all generators in this area
            print('Area {}'.format(i))
            base = 0
            mass = 0
            power_limit = 0
            power_gen = 0
            for k in self.area_gen[i]:
                base += self.ppc["gen"][k, 10]
                mass += self.ppc["gen"][k, 10] * self.ppc["gen"][k, 11] * (1 - int(int(economic_dispatch[k][self.instant_index])==0))
                power_limit += self.ppc['gen'][k, 2]
                power_gen += economic_dispatch[k][self.instant_index]
                print('Gen {} output {} status {}'.format(k, economic_dispatch[k][self.instant_index],
                                                          (1 - int(int(economic_dispatch[k][self.instant_index])==0))))
            # compute headroom
            self.area_headroom.append((power_limit - power_gen)/base)
            # store base for converting per unit value
            self.area_base.append(base)
            # compute center of inertia (in terms of base)
            self.area_inertia.append(mass/base)




    def compute_tieline_syn(self, angle, voltage):
        """
        Calculate tie-line synchronizing coefficient for each area using given time instant and economic dispatch results
        The principle is to approximate the term |E1||E2|/X12*cos(t1-t2) using the terminal buses.
        The 2*pi term is because the dynamic angle difference in radian is calculated from frequency, which maybe in per unit.
        The 2*pi term is used in the case where frequency is in per unit.
        """

        # compute area-averaged angle and voltage
        self.area_voltage = []
        self.area_angle = []
        for i in self.iter_area:
            voltage_sum = 0
            angle_sum = 0
            for j in self.area_bus[i]:
                voltage_sum += voltage[j][self.instant_index]
                angle_sum += angle[j][self.instant_index]
            self.area_voltage.append(voltage_sum/len(self.area_bus[i]))
            self.area_angle.append(angle_sum/len(self.area_bus[i]))

        # identify tieline from branch data
        line_bus = self.ppc['branch'][:, 0:2]
        # replace bus name index in the line-bus matrix by area index
        for i in self.iter_line:
            from_bus_idx = line_bus[i, 0] - 1
            to_bus_idx = line_bus[i, 1] - 1
            for j in self.iter_area:
                for k in self.area_bus[j]:
                    if k == from_bus_idx:
                        line_bus[i, 0] = j
                    elif k == to_bus_idx:
                        line_bus[i, 1] = j
                    else:
                        pass
        self.line_bus_area = line_bus

        # define a tieline dictionary
        tieline = OrderedDict()
        for i in self.iter_area:
            tieline[i] = OrderedDict()
            for j in self.iter_area:
                tieline[i][j] = []

        # define a tieline dictionary
        tieline_coef = OrderedDict()
        for i in self.iter_area:
            tieline_coef[i] = OrderedDict()
            for j in self.iter_area:
                tieline_coef[i][j] = 0

        # define a tieline dictionary
        tieline_adm = OrderedDict()
        for i in self.iter_area:
            tieline_adm[i] = OrderedDict()
            for j in self.iter_area:
                tieline_adm[i][j] = 0

        # identify tieline
        for i in self.iter_line:
            if self.line_bus_area[i, 0] != self.line_bus_area[i, 1]:
                index_a = int(self.line_bus_area[i, 0])
                index_b = int(self.line_bus_area[i, 1])
                # add index
                tieline[index_a][index_b].append(i)
                tieline[index_b][index_a].append(i)
        self.tieline = tieline

        # compute admittance
        for i in self.iter_area:
            for j in self.iter_area:
                if i != j:
                    tieline_adm[i][j] = sum(np.reciprocal(self.ppc['branch'][k, self.idx.BR_X]) for k in tieline[i][j])
        self.tieline_adm = tieline_adm

        # compute synchronizing coefficients using |E1||E2|/X12*cos(t1-t2)
        # remember to perform a change of base
        for i in self.iter_area:
            for j in self.iter_area:
                if i != j:
                    tieline_coef[i][j] = 2 * math.pi * self.area_voltage[i] * self.area_voltage[j] * self.tieline_adm[i][j] \
                                         * math.cos(self.area_angle[i] - self.area_angle[j]) * self.base_mva / self.area_base[i]
        self.tieline_coef = tieline_coef



    def symbolic_model_define(self, i, tie_line_flag=1):
        # define the symbolic dynamic model, state and control vector of area i
        # disturbance is given in the unit of MW
        # input: i - area index, m - class for parameter of LFC
        # input: tie_line_flag = 1 keep tie-line flow as scheduled, 0 otherwise
        self.dfdt[i] = []
        self.x[i] = []
        self.u[i] = []

        # dynamic equation and states
        # swing dynamics
        self.dfdt[i].append(1 / (self.area_inertia[i] * 2) * (self.pm[i] - self.pl[i]/self.area_base[i] - self.damp[i] * self.f[i] - sum(p for p in self.ptl[i])))
        self.x[i].append(self.f[i])
        # tie-line power flow
        for j in iter_remove_index(self.number_area, i):
            if self.tieline_coef[i][j] == 0:
                pass
            else:
                self.dfdt[i].append(self.tieline_coef[i][j] * (self.f[i] - self.f[j]))
                self.x[i].append(self.ptl[i][j])
        # turbine
        self.dfdt[i].append(1 / self.T_turbine[i] * (self.pv[i] - self.pm[i]))
        self.x[i].append(self.pm[i])
        # governor
        self.dfdt[i].append(1 / self.T_governor[i] * (-self.pv[i] - self.gain[i] * self.er[i] - 1 / self.droop[i] * self.f[i]))
        self.x[i].append(self.pv[i])
        # area error
        self.dfdt[i].append(tie_line_flag * sum(p for p in self.ptl[i]) + (1 / self.droop[i] + self.damp[i]) * self.f[i])
        self.x[i].append(self.er[i])

        # control input
        self.u[i].append(self.pl[i])



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

