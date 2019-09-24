from gurobipy import *
import numpy as np

# Optimization model for tree clearing routing
def treeclear_solve (Set_N, Set_R, Cap_c, Res_i, Tr_ijc, Et_ic, Tmax, start_time, objFlag):
    """Optimization model for tree clearing routing"""
    m = Model('tree_clear')


    Tday = start_time + 2 * Tmax
    # Define big M method
    M = start_time + 2 * Tmax


    # Create decision variables for x_ijc
    x_ijc = m.addVars(Set_N, Set_N, Set_R, vtype=GRB.BINARY, name = 'x')

    # Create decision variables for y_ic
    y_ic = m.addVars(Set_N, Set_R, vtype = GRB.BINARY, name = 'y')

    # Create decision variables for at_ic
    at_ic = m.addVars(Set_N, Set_R, lb = 0.0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'at')

    # Create decision variables for rt_c
    #rt_c = m.addVars(Set_R, lb = 0.0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'rt')


    # Add constraints that x[i,i,c] = 0 to avoid self-loop
    m.addConstrs((x_ijc[i,i,c] == 0 for i in Set_N for c in Set_R), name = 'redundant')

    # Add constraints \sum_j x[i,j,c] = \sum_j x[j,i,c]: flow conservation constraints
    m.addConstrs((x_ijc.sum(i,'*',c) == x_ijc.sum('*',i,c) for i in Set_N for c in Set_R), name = 'flow_conservation')

    # Add constraints y[i,c] = \sum_j x[i,j,c]: couling constraint for flow and vist
    m.addConstrs((y_ic[i,c] == x_ijc.sum(i,'*',c) for i in Set_N for c in Set_R), name = 'vist_coupling')

    # Add constraints \sum_c y[i,c] <= 1: each component is visited once, except for the depot
    m.addConstrs((y_ic.sum(i,'*') <= 1 for i in Set_N if i != Set_N[0]), name = 'vist_cover')

    # Add constraints \sum_c y[0,c] = len (Set_R)
    m.addConstr((quicksum(y_ic[Set_N[0],c] for c in Set_R) == len(Set_R)), name = 'vist_depot')

    # Add resource constraints
    m.addConstrs((quicksum(Res_i[i] * y_ic[i,c] for i in Set_N if i != Set_N[0]) <= Cap_c[c] for c in Set_R),
                 name = 'resource_capacity')


     # Add constraints for repair time
    m.addConstrs((at_ic[j,c] >= at_ic[i,c] + Tr_ijc[i,j,c] + Et_ic[i,c] - (1 - x_ijc[i,j,c]) * M
                   for i in Set_N for j in Set_N if j != Set_N[0] for c in Set_R),
                  name = 'arrival_node')
    #
    # #Add constraints for return time, Set_N[0] represent depot index
    # m.addConstrs((rt_c[c] <= at_ic[i,c] + Tr_ijc[i,Set_N[0],c] + Et_ic[i,c] + (1 - x_ijc[i,Set_N[0],c]) * M
    #               for i in Set_N if i != Set_N[0] for c in Set_R),
    #              name = 'arrival_depot')
    #
    # m.addConstrs((rt_c[c] >= at_ic[i, c] + Tr_ijc[i, Set_N[0], c] + Et_ic[i, c] - (1 - x_ijc[i, Set_N[0], c]) * M
    #               for i in Set_N if i != Set_N[0] for c in Set_R),
    #              name='arrival_depot')
    #
    # m.addConstrs((rt_c[c] <= Tmax for c in Set_R), name = 'arrival_bound')

    # Constraints that time to return to depot should be within its timeshift
    m.addConstrs((at_ic[i,c] + Tr_ijc[i,Set_N[0],c] + Et_ic[i,c] - (1 - x_ijc[i,Set_N[0],c]) * M <= Tmax
                 for i in Set_N if i != Set_N[0] for c in Set_R), name = 'arrival_bound')


    # Set the starting time
    m.addConstrs((at_ic[Set_N[0],c] == start_time for c in Set_R), name = 'arrival_initial')

    # when objFlag = 2, it is for minimal complete time
    if objFlag == 2:
        obj = quicksum(quicksum(at_ic[i,c] + y_ic[i,c] * Et_ic[i,c] for c in Set_R)
                       + (1 - quicksum(y_ic[i,c] for c in Set_R)) * Tday for i in Set_N if i != Set_N[0]) #+ quicksum(rt_c[c] for c in Set_R)

    # When objFlag = 1, it is for minimal arrival time (i.e., waiting time)
    if objFlag == 1:
        obj = quicksum(quicksum(at_ic[i,c] for c in Set_R)
                       + (1 - quicksum(y_ic[i,c] for c in Set_R)) * Tday for i in Set_N if i != Set_N[0])

    m.setObjective(obj, GRB.MINIMIZE)

    m.update()
    m.optimize()

    return m, x_ijc, y_ic, at_ic



def poledeliver_solve (Set_N, Set_R, Cap_c, Res_i, Tr_ijc, Et_ic, Tmax, start_time, Acc_i, objFlag):
    m = Model('pole_delivery')


    Tday = start_time + 2 * Tmax
    # Define big M method
    M = start_time + 2 * Tmax


    # Create decision variables for x_ijc
    x_ijc = m.addVars(Set_N, Set_N, Set_R, vtype=GRB.BINARY, name = 'x')

    # Create decision variables for y_ic
    y_ic = m.addVars(Set_N, Set_R, vtype = GRB.BINARY, name = 'y')

    # Create decision variables for at_ic
    at_ic = m.addVars(Set_N, Set_R, lb = 0.0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'at')

    # Create decision variables for rt_c
    rt_c = m.addVars(Set_R, lb = 0.0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'rt')


    # Add constraints that x[i,i,c] = 0 to avoid self-loop
    m.addConstrs((x_ijc[i,i,c] == 0 for i in Set_N for c in Set_R), name = 'redundant')

    # Add constraints \sum_j x[i,j,c] = \sum_j x[j,i,c]: flow conservation constraints
    m.addConstrs((x_ijc.sum(i,'*',c) == x_ijc.sum('*',i,c) for i in Set_N for c in Set_R), name = 'flow_conservation')

    # Add constraints y[i,c] = \sum_j x[i,j,c]: couling constraint for flow and vist
    m.addConstrs((y_ic[i,c] == x_ijc.sum(i,'*',c) for i in Set_N for c in Set_R), name = 'vist_coupling')

    # Add constraints \sum_c y[i,c] <= 1: each component is visited once, except for the depot
    m.addConstrs((y_ic.sum(i,'*') <= 1 for i in Set_N if i != Set_N[0]), name = 'vist_cover')

    # Add constraints \sum_c y[0,c] = len (Set_R)
    m.addConstr((quicksum(y_ic[0,c] for c in Set_R) == len(Set_R)), name = 'vist_depot')

    # Add resource constraints
    m.addConstrs((quicksum(Res_i[i] * y_ic[i,c] for i in Set_N if i != Set_N[0]) <= Cap_c[c] for c in Set_R),
                 name = 'resource_capacity')


    # Add constraints for repair time
    m.addConstrs((at_ic[j,c] >= at_ic[i,c] + Tr_ijc[i,j,c] + Et_ic[i,c] - (1 - x_ijc[i,j,c]) * M
                  for i in Set_N for j in Set_N if j != Set_N[0] for c in Set_R),
                 name = 'arrival_node')

    #Add constraints for return time, Set_N[0] represent depot index
    # m.addConstrs((rt_c[c] <= at_ic[i,c] + Tr_ijc[i,Set_N[0],c] + Et_ic[i,c] + (1 - x_ijc[i,Set_N[0],c]) * M
    #               for i in Set_N if i != Set_N[0] for c in Set_R),
    #              name = 'arrival_depot')
    #
    # m.addConstrs((rt_c[c] >= at_ic[i, c] + Tr_ijc[i, Set_N[0], c] + Et_ic[i, c] - (1 - x_ijc[i, Set_N[0], c]) * M
    #               for i in Set_N if i != Set_N[0] for c in Set_R),
    #              name='arrival_depot')
    #
    # m.addConstrs((rt_c[c] <= Tmax for c in Set_R), name = 'arrival_bound')

    # Arrival back to depot constraints
    m.addConstrs((at_ic[i, c] + Tr_ijc[i, Set_N[0], c] + Et_ic[i, c] - (1 - x_ijc[i, Set_N[0], c]) * M <= Tmax
                  for i in Set_N if i != Set_N[0] for c in Set_R), name='arrival_bound')

    m.addConstrs((at_ic[Set_N[0],c] == 0 for c in Set_R), name = 'arrival_initial')


    # Add time window constraints: at_ic >= d_i * y_ic
    m.addConstrs(at_ic[i,c] >= Acc_i[i] * y_ic[i,c] for i in Set_N if i!=Set_N[0] for c in Set_R)

    if objFlag == 2:
        obj = quicksum(quicksum(at_ic[i,c] + y_ic[i,c] * Et_ic[i,c] for c in Set_R)
                       + (1 - quicksum(y_ic[i,c] for c in Set_R)) * Tday for i in Set_N if i != Set_N[0]) #+ quicksum(rt_c[c] for c in Set_R)

    if objFlag == 1:
        obj = quicksum(quicksum(at_ic[i,c] for c in Set_R)
                       + (1 - quicksum(y_ic[i,c] for c in Set_R)) * Tday for i in Set_N if i != Set_N[0])

    m.setObjective(obj, GRB.MINIMIZE)

    m.update()
    m.optimize()

    return m, x_ijc, y_ic, at_ic

# Optimization model for the clustering problem
def cluster_solve(damage_components, depots, resources, Dcr, distance, Res_c, Res_d, damage_allo):
    m = Model('clustering')

    # Create decision variables s_{sigma,i}
    allocation = m.addVars(depots,damage_components,vtype=GRB.BINARY, name='alloc')

    # Set objective to minimize distances
    #m.setObjective(allocation.prod(distance), GRB.MINIMIZE)
    m.setObjective(quicksum(allocation[sigma, i] * distance[sigma, i] for sigma in depots for i in damage_components),
                   GRB.MINIMIZE)

    # Constraints summation over
    m.addConstrs((allocation.sum('*',i) == 1 for i in damage_components),"cluster")

    # Set constraints
    m.addConstrs(
        (quicksum(Res_c[i,k] * allocation[sigma, i] for i in damage_components)
            <= Res_d[sigma,k] for sigma in depots for k in resources), "cap")

    m.addConstrs(
        (allocation[sigma,i] <= Dcr[sigma,i] for sigma in depots for i in damage_components), "DCR")


    m.addConstrs(allocation[sigma,i] == 1 for sigma,i in damage_allo)

    m.update()
    m.optimize()

    return m, allocation


# find a routing from a matrix
# x is a two-dimensional matrix (numpy ndarray)
# return the sequence of the route
def find_route(x):
    i = 0
    route = [0]
    # store the index of the row, which is a tuple, where each element is an array
    index_tuple = np.where(x[i,:] == 1)
    back_flag = 0
    # As long as there is 1 in a row
    while len(index_tuple) > 0 and back_flag == 0:
        if len(index_tuple) > 1:
            print('Error for the matrix')
            break
        else:
            # Update the row index,
            i = index_tuple[0].item()
            route.append(i)
            # Update the index tuple
            index_tuple = np.where(x[i, :] == 1)
            # When finding the start node, return flag = 1
            if i == 0:
                back_flag = 1
    return route


