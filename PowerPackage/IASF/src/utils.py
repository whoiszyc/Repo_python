
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict


class SolutionDict(OrderedDict):
    """
    solution dictionary is an ordered dictionary that stores the optimization results
    First level key is variable category like generator, load
    Second level key is variable index like generator 30, 31
    At last, time series data is stored in a list
    """

    def __init__(self, time_series=None, legend_list=None, x_str='Time',  y_str='Value', title_str=None, fig_size=(12, 5)):
        self.time_series = time_series
        self.legend_list = legend_list
        self.x_str = x_str
        self.y_str = y_str
        self.title_str = title_str
        self.fig_size = fig_size
        self.x_ticks = None
        self.y_ticks = None


    def plot_step_2d(self):
        """step plot"""
        plt.figure(figsize=self.fig_size)

        # get different components like generator 1, generator 2
        key_list = []
        for i in self.keys():
            key_list.append(i)

        # get x-axis values for different components
        total_time = []
        for i in range(len(key_list)):
            total_time.append(len(self[i]))

        # get the maximum time range index
        time_range = range(0, max(total_time))

        if self.legend_list==None:
            self.legend_list = ['variable {}'.format(i) for i in key_list]

        for i in key_list:
            plt.step(range(0, total_time[i]), self[i], label=self.legend_list[i], linewidth=3)

        # set title and label
        plt.title(self.title_str)
        plt.xlabel(self.x_str)
        plt.ylabel(self.y_str)

        # set legends
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)

        #set xticks and yticks
        if self.x_ticks == None:
            pass
        else:
            plt.xticks(time_range, self.x_ticks)

        plt.show()



def iter_remove_index(end, a):
    # create a iterator from 0 to end without a given a

    # prepare a list for iteration without i
    iter_j = [j for j in range(end)]
    #get the index in iter_j where the element equals to a
    index_equal_a = iter_j.index(a)
    # delete the element
    iter_j.remove(iter_j[index_equal_a])

    return iter_j