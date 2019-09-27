import sys
import os
import time
import pandas as pd
import numpy as np
import math
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import ast
import networkx as nx
import operator
from collections import OrderedDict
# import gurobipy as gb
import pyomo.environ as pm



class OutageManage:
    """
    A distribution outage management and decision support formulation
    The parent class will read data and specification
    """

    ## Optimization specification
    Total_Time = 100
    iter_time = np.arange(0, Total_Time)
    BigM = 1000
    FicFlowBigM = 100
    Voltage_Variation = 0.1
    Voltage_Substation = 1.05
    epsilon = 0.1
    BasePower = 1000

    ## prepare optimization data
    def data_preparation(self,ppc,vrp):
        """
        read data of distribution network and vehicle routing problem
        Data is in the format of dictionary
        """
        self.ppc = ppc
        self.vrp = vrp

        ## get distribution network data
        self.number_bus = ppc['number_bus']
        self.number_line = ppc['number_line']
        self.number_gen = ppc['number_gen']
        self.index_bus = ppc['index_bus']
        self.index_line = ppc['index_line']
        self.index_gen = ppc['index_gen']
        self.iter_bus = ppc['iter_bus']
        self.iter_gen = ppc['iter_gen']
        self.iter_line = ppc['iter_line']
        self.bus_line = ppc['bus_line']
        self.bus_gen = ppc['bus_gen']

        ## get vehicle routing data
        self.iter_crew = vrp['iter_crew']
        self.iter_vertex = vrp['iter_vertex']
        self.index_vertex = vrp['index_vertex']
        self.number_vertex = vrp['number_vertex']
        self.ordered_vertex = vrp['ordered_vertex']


        ## sort line type
        # (1)fixed line, (2)damaged line, (3)tie line
        self.line_damaged = set(self.iter_vertex) - {'0', 'd'}
        self.line_switch = set(ppc['tieline'])
        self.line_static = set(self.iter_line) - set(ppc['tieline']) - self.line_damaged


        ## Store travel time with the given order in a matrix for Z3
        ordered_travel_time = np.zeros((self.number_vertex, self.number_vertex))
        index_i = 0
        for i in self.ordered_vertex:
            index_j = 0
            for j in self.ordered_vertex:
                if i == j:
                    ordered_travel_time[index_i, index_j] = 0
                else:
                    ordered_travel_time[index_i,index_j] = vrp['travel'][i][j]
                index_j = index_j + 1
            index_i = index_i + 1

        ## store repair time with the given order in a array
        ordered_repair_time = np.zeros((self.number_vertex, 1))
        index_i = 0
        for i in self.ordered_vertex:
            ordered_repair_time[index_i] = vrp['repair'][0][i]
            index_i = index_i + 1

        ## get the total time by add the travel and repair time together
        ordered_total_time = np.zeros((self.number_vertex,self.number_vertex))
        for i in range(self.number_vertex):
            for j in range(self.number_vertex):
                ordered_total_time[i,j] = ordered_travel_time[i,j] + ordered_repair_time[j]

        ## store ordered travel time
        self.ordered_total_time = ordered_total_time


        ## Get bus and line relation index using index not string as name
        bus_line_index = OrderedDict()
        for i in range(self.number_bus):
            bus_name = 'bus_{}'.format(i+1)
            bus_line_index[i] = []

            # get line index where power flow variables flowing out from this bus
            for j in self.bus_line[bus_name]["line_from_this_bus"]:
                bus_line_index[i].append(j-1)
            # get line index where power flow variables flowing into this bus
            for j in self.bus_line[bus_name]["line_to_this_bus"]:
                bus_line_index[i].append(j-1)

        self.bus_line_index = bus_line_index




class SolutionDict(OrderedDict):
    """
    Solution dictionary is an ordered dictionary that stores the optimization results
    Solution dictionary struture: D[name key]=time series data
    """

    def plot_2d(self, x_str='Time', y_str='Value', title_str='Results', figsize=(15,7), legendlist=None):
        """step plot"""
        plt.figure(figsize=figsize)

        key_list = []
        for i in self.keys():
            key_list.append(i)

        ## assume that all variables have the same time vector
        total_time = len(self[key_list[0]])

        if legendlist==None:
            legendlist = ['variable {}'.format(i) for i in key_list]

        k = 0
        for i in key_list:
            plt.plot(range(0, total_time), self[i], label=legendlist[k], linewidth=3)
            k = k + 1
        plt.title(title_str)
        plt.xlabel(x_str)
        plt.ylabel(y_str)
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
        plt.show()


    def plot_step_2d(self, x_str='Time', y_str='Value', title_str='Results', figsize=(15,7), legendlist=None):
        """step plot"""
        plt.figure(figsize=figsize)

        key_list = []
        for i in self.keys():
            key_list.append(i)

        ## assume that all variables have the same time vector
        total_time = len(self[key_list[0]])

        if legendlist==None:
            legendlist = ['variable {}'.format(i) for i in key_list]

        k = 0
        for i in key_list:
            plt.step(range(0, total_time), self[i], label=legendlist[k], linewidth=3)
            k = k + 1
        plt.title(title_str)
        plt.xlabel(x_str)
        plt.ylabel(y_str)
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
        plt.show()


    def plot_bin_2d(self, x_str='Time', y_str='Value', title_str='Results', figsize=(15,7)):
        """step plot"""
        plt.figure(figsize=figsize)

        ## get key list
        key_list = []
        for i in self.keys():
            key_list.append(i)

        ## assume that all variables have the same time vector
        total_time = len(self[key_list[0]])

        y_axis = np.arange(0, len(key_list))
        k = 0
        for i in key_list:
            for t in range(0, total_time):
                if abs(self[i][t]) <= 0.01:
                    plt.scatter(t, y_axis[k], c='red', s=50, alpha=0.5, edgecolors='none')
                else:
                    plt.scatter(t, y_axis[k], c='green', s=50, alpha=0.5, edgecolors='none')
            k = k + 1
        plt.yticks(y_axis, key_list)
        plt.show()
        plt.title(title_str)
        plt.xlabel(x_str)
        plt.ylabel(y_str)





