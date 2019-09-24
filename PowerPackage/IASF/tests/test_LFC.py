import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

# IASF object and function
from IASF.src.control import FrequencyControlProblem
from IASF.src.utils import SolutionDict

# test case data
from IASF.cases.ieee_39bus import *
from IASF.cases.index import idx_matpower
from IASF.cases.dyn import LFC_39bus
network_data = case39()


# read dependent data in excel format
time_series_df = pd.read_excel('time_series.xlsx')
ed_df = pd.read_excel('economic_dispatch.xlsx')
pf_df = pd.read_excel('power_flow.xlsx')
voltage_df = pd.read_excel('voltage.xlsx')
angle_df = pd.read_excel('angle.xlsx')
# convert to dictionary
# ! ! Note that here the data structure is master dictionary includes sub-dictionary as time-series data
# ! ! Our traditional method is master disctionary includes list as time-series data
time_series = time_series_df.to_dict()
ed = ed_df.to_dict()
pf = pf_df.to_dict()
voltage = voltage_df.to_dict()
angle = angle_df.to_dict()

# # read dependent data in csv format
# # !! The problem using csv format to read and write data is that once the data is converted back to the dictionary,
# # !! the key becomes a string type rather than int type.
# ed_df0 = pd.read_csv('economic_dispatch.csv')
# pf_df0 = pd.read_csv('power_flow.csv')
# # convert to dictionary
# ed0 = ed_df0.to_dict()
# pf0 = pf_df0.to_dict()


# Define load frequency control instance and pass data to its attributes
lfc = FrequencyControlProblem(LFC_39bus, '13:45')
lfc.time_series = time_series[0]


lfc.locate_index()
# print(lfc.instant_index)

lfc.get_network_data(network_data, idx_matpower)
lfc.compute_area_generation(ed)

lfc.compute_tieline_syn(angle, voltage)
# print(lfc.line_bus_area)
# print(lfc.tieline)
# print(lfc.area_voltage)
# print(lfc.area_angle)
# print(lfc.tieline_adm)
# print(lfc.tieline_coef)


# define symbolic model
lfc.symbolic_model_define(0, 1)
lfc.symbolic_model_define(1, 1)
lfc.symbolic_model_define(2, 1)

# get state space
lfc.get_vector()
lfc.get_state_space()


# define saturation
lfc.saturation_default(-100000, 100000)
# set saturation from computation results of economic dispatch
lfc.saturation_set('Pm_0', -1, lfc.area_headroom[0])
lfc.saturation_set('Pm_1', -1, lfc.area_headroom[1])
lfc.saturation_set('Pm_2', -1, lfc.area_headroom[2])


# give initial condition
lfc.x0 = [0] * lfc.number_state

# define time series
lfc.t = np.linspace(0, 50, 2000)

# define input
lfc.input_default()
lfc.input_set('Pl_0', 5, 50, 500)

# solve problem
lfc.ode_euler()



plt.figure(figsize=(8,4))
plt.plot(lfc.t, lfc.get_state_data('f_0')*60, 'b--', linewidth=2, label='Area 1')
plt.plot(lfc.t, lfc.get_state_data('f_1')*60, 'g--', linewidth=2, label='Area 2')
plt.plot(lfc.t, lfc.get_state_data('f_2')*60, 'r--', linewidth=2, label='Area 3')
plt.xlabel('Time (second)')
plt.ylabel('Frequency (Hz)')
plt.legend(loc='best')
plt.title('Frequency in Each Area')
# plt.savefig('frequency.png')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(lfc.t, lfc.get_state_data('Ptl_0_1'), 'b--', linewidth=2, label='Tie-line flow between area 1 and 2')
plt.plot(lfc.t, lfc.get_state_data('Ptl_0_2'), 'g--', linewidth=2, label='Tie-line flow between area 1 and 3')
plt.plot(lfc.t, lfc.get_state_data('Ptl_1_2'), 'r--', linewidth=2, label='Tie-line flow between area 2 and 3')
plt.xlabel('Time (second)')
plt.ylabel('Tie-line Power (per unit)')
plt.legend(loc='best')
plt.title('Tie-line Power Flow')
# plt.savefig('tie_line.png')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(lfc.t, lfc.get_state_data('Pm_0'), 'b--', linewidth=2, label='Area 1')
plt.plot(lfc.t, lfc.get_state_data('Pm_1'), 'g--', linewidth=2, label='Area 2')
plt.plot(lfc.t, lfc.get_state_data('Pm_2'), 'r--', linewidth=2, label='Area 3')
plt.xlabel('Time (second)')
plt.ylabel('Mechanical Power Variation (per unit)')
plt.legend(loc='best')
plt.title('Mechanical Power Variation in Each Area')
# plt.savefig('power.png')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(lfc.t, lfc.get_state_data('Pv_0'), 'b--', linewidth=2, label='Area 1')
plt.plot(lfc.t, lfc.get_state_data('Pv_1'), 'g--', linewidth=2, label='Area 2')
plt.plot(lfc.t, lfc.get_state_data('Pv_2'), 'r--', linewidth=2, label='Area 3')
plt.xlabel('Time (second)')
plt.ylabel('Valve Position Variation (per unit)')
plt.legend(loc='best')
plt.title('Valve Position Variation in Each Area')
# plt.savefig('valve.png')
plt.show()
