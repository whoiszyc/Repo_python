
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
from pyomo.environ import *
import os
import networkx as nx
import matplotlib.pyplot as plt



def crew_dispatch_random():
    # define a dictionary for vehicle routing problem
    vrp = {}

    # --------- crew --------------
    vrp['number_crew'] = 1
    vrp['iter_crew'] = np.arange(0, vrp['number_crew'])

    # fault components and its location
    vrp['fault'] = {}
    vrp['fault']['location'] = {'0': (0, 0.2), 'd': (0, -0.2), 'line_3': (-1, 1), 'line_5': (1, 1), 'line_9': (5, 2),
                                'line_11': (4, 4), 'line_23': (-2, 3), 'line_28': (3, -1)}

    # get iterator
    vrp['ordered_vertex'] = ['0', 'line_3', 'line_5', 'line_9', 'line_11', 'line_23', 'line_28', 'd']
    vrp['iter_vertex'] = vrp['fault']['location'].keys()
    vrp['number_vertex'] = len(vrp['iter_vertex'])
    vrp['index_vertex'] = np.arange(vrp['number_vertex'])

    # create a directed graph for visualization of vehicle routing problem
    vrp['graph'] = nx.DiGraph()
    for i in vrp['ordered_vertex']:
        vrp['graph'].add_node(i)

    plt.figure(figsize=(7, 5))
    nx.draw(vrp['graph'], vrp['fault']['location'], with_labels=True)
    plt.show()


    # -----------random data -----------
    # random repairing time of fault components by each crew
    vrp['repair'] = {}
    for c in vrp['iter_crew']:
        vrp['repair'][c] = {}
        for i in vrp['ordered_vertex']:
            if i == '0' or i == 'd':
                vrp['repair'][c][i] = 0
                vrp['repair'][c][i] = 0
            else:
                vrp['repair'][c][i] = np.random.randint(1, 10)

    # random traveling time to check robustness of the formulation
    vrp['travel'] = {}
    for i in vrp['ordered_vertex']:
        vrp['travel'][i] = {}
        if i == 'd':
            for j in vrp['ordered_vertex']:
                vrp['travel'][i][j] = 0
        else:
            for j in vrp['ordered_vertex']:
                vrp['travel'][i][j] = np.random.randint(1, 10)

    return vrp





def crew_dispatch_determ():
    # define a dictionary for vehicle routing problem
    vrp = {}

    # --------- crew --------------
    vrp['number_crew'] = 1
    vrp['iter_crew'] = np.arange(0, vrp['number_crew'])

    # fault components and its location
    vrp['fault'] = {}
    vrp['fault']['location'] = {'0': (0, 0.2), 'd': (0, -0.2), 'line_3': (-1, 1), 'line_5': (1, 1), 'line_9': (5, 2),
                                'line_11': (4, 4), 'line_23': (-2, 3), 'line_28': (3, -1)}

    # get iterator
    vrp['ordered_vertex'] = ['0', 'line_3', 'line_5', 'line_9', 'line_11', 'line_23', 'line_28', 'd']
    vrp['iter_vertex'] = vrp['fault']['location'].keys()
    vrp['number_vertex'] = len(vrp['iter_vertex'])
    vrp['index_vertex'] = np.arange(vrp['number_vertex'])

    # create a directed graph for visualization of vehicle routing problem
    vrp['graph'] = nx.DiGraph()
    for i in vrp['ordered_vertex']:
        vrp['graph'].add_node(i)

    # plt.figure(figsize=(7, 5))
    # nx.draw(vrp['graph'], vrp['fault']['location'], with_labels=True)
    # plt.show()


    # ---------- deterministic time data ----------
    # repairing time of fault components by each crew
    vrp['repair'] = {}
    for c in vrp['iter_crew']:
        vrp['repair'][c] = {}
        for i in vrp['ordered_vertex']:
            if i == '0' or i == 'd':
                vrp['repair'][c][i] = 0
                vrp['repair'][c][i] = 0
            else:
                vrp['repair'][c][i] = int(5)

    # traveling time based on coordinates
    vrp['travel'] = {}
    for i in vrp['ordered_vertex']:
        vrp['travel'][i] = {}
        for j in vrp['ordered_vertex']:
            vrp['travel'][i][j] = int(
                round(math.sqrt((vrp['fault']['location'][i][0] - vrp['fault']['location'][j][0]) ** 2 + \
                                (vrp['fault']['location'][i][1] - vrp['fault']['location'][j][1]) ** 2)))


    return vrp

