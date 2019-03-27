
import numpy as np
import pandas as pd
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *


class uc:
    def get_data(self, x, y):
        self.left=x
        self.right=y


A_UC_Instance=uc()
A_UC_Instance.get_data(5,8)
print(A_UC_Instance.left)
print(A_UC_Instance.right)

