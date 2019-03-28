
import numpy as np
import pandas as pd
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *


class uc:
    def get_data(self, x, y):
        self.a=x
        self.b=y


    def operation(self, OperationType):
        if OperationType == '+':
            self.result = self.summation()
        elif OperationType == '*':
            self.result = self.multiple()
        else:
            print('Such operation is not defined')
            self.result=None

    # Functions that are called do not need to be in the front
    def summation(self):
        return self.a + self.b

    def multiple(self):
        return self.a * self.b




u=uc()
u.get_data(5, 8)
print(u.a)
print(u.b)

u.operation('kk')
print(u.result)


