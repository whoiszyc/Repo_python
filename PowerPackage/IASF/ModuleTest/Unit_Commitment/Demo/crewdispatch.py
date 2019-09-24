


import scipy.io as sio
import numpy as np
import crewoptmodel as copt
from gurobipy import *
import os
import xlrd
#import pandas as pd


flag_cluster = False


# Get the name of the current directory, __file__ represents the current file
currentpath = os.path.dirname(__file__)


# # Get the full path of the data file
# datapath = os.path.join(currentpath,"data","clustering_data.mat")
# # Load the .mat data
# input_data = sio.loadmat(datapath)
# N = int(input_data['faultN'])  # number of components needed to be repaired
# S = int(input_data['depotN'])   # number of depots
#
# damage_components = np.arange(N)
# depots = np.arange(S)
#
# # Number of resources
# K = 1
# resources = np.arange(K)
#
# N_M = [(1,17)]
#
#
# #Dcr
# Dcr = input_data['Dcr']
#
# #distance
# distance_array = np.reshape(input_data['setDist'],(S,N))
# distance = {(sigma,i) : distance_array[sigma][i] for sigma in depots for i in damage_components}
#
# Res_d_array = np.reshape(input_data['setResource_depot'],(S,K))
# Res_d = {(sigma,k) : Res_d_array[sigma][k] for sigma in depots for k in resources}
#
# Res_c_array = np.reshape(input_data['setResource_node'],(N,K))
# Res_c = {(i,k):Res_c_array[i][k] for i in damage_components for k in resources}



fault_file = os.path.join(currentpath,"fault_line.xls")  # "data","feeder123",

wb = xlrd.open_workbook(fault_file)
wbsheet = wb.sheet_by_index(0)

faultlines = {}
k = 1
for rowidx in range(wbsheet.nrows):
    if rowidx != 0:
        nodepair = [int(wbsheet.cell(rowidx,0).value), int(wbsheet.cell(rowidx,1).value)]
        faultlines[k] = nodepair
        k += 1

print(faultlines)

#faultNode = pd.read_excel(fault_file)
#print(faultNode.head())


# Indicate number of crew in each depot
crewNumdepot = [3, 2, 3]

# Indicate the number of depot
depotN = len(crewNumdepot)







# if the number of depot > 1, the cluster will be done
if depotN > 1:
    flag_cluster = True

faultN = 18

# type of resources
K = 1


seedVar = 5
np.random.seed(seed = seedVar)


# Generate the resource capacity for each crew in each depot
# sourceCrewdepot is a dict, where key is the index of depot, content is the sources for each crew
depotidx = 0
sourceCrewdepot = {}
source_min = 2
source_max = 6
for crewNum in crewNumdepot:
    np.random.seed(seed = seedVar)
    seedVar += 1
    sourceCrewdepot[depotidx] = 8 * np.round(source_min + (source_max - source_min) * np.random.rand(crewNum))
    depotidx += 1



# Generate the distance between depot to damage component
# Need to call map API to get this information
setNd1_near = np.array([1, 2, 3, 4, 5, 6, 18])
setNd1_mid =np.array([17, 7, 10, 8, 9, 11, 14])
setNd1_far = np.array([16, 15, 12, 13])

setNd2_near = np.array([17, 16, 15, 7, 8, 9, 10, 11])
setNd2_mid = np.array([12, 13, 14, 4, 3])
setNd2_far = np.array([6, 5, 18, 1, 2])

setNd3_near = np.array([14, 12, 13, 4, 8, 9, 10, 11])
setNd3_mid = [3, 7, 15, 16]
setNd3_far = [17, 6, 5, 18, 1, 2]

dmin_near = 10
dmax_near = 40

dmin_mid = 25
dmax_mid = 55

dmin_far = 35
dmax_far = 70

setDist = np.zeros((depotN,faultN), dtype='uint8')


# Generate the random distance from depot to each damaged component
seedVar += 1
np.random.seed(seed = seedVar)

temprand = np.random.rand(depotN,faultN)

setDist[0,setNd1_near-1] = dmin_near + (dmax_near - dmin_near)*temprand[0,setNd1_near-1]
setDist[0,setNd1_mid-1] = dmin_mid + (dmax_mid - dmin_mid)*temprand[0,setNd1_mid-1]
setDist[0,setNd1_far-1] = dmin_far + (dmax_far - dmin_far)*temprand[0,setNd1_far-1]

setDist[1,setNd1_near-1] = dmin_near + (dmax_near - dmin_near)*temprand[1,setNd1_near-1]
setDist[1,setNd1_mid-1] = dmin_mid + (dmax_mid - dmin_mid)*temprand[1,setNd1_mid-1]
setDist[1,setNd1_far-1] = dmin_far + (dmax_far - dmin_far)*temprand[1,setNd1_far-1]

setDist[2,setNd1_near-1] = dmin_near + (dmax_near - dmin_near)*temprand[2,setNd1_near-1]
setDist[2,setNd1_mid-1] = dmin_mid + (dmax_mid - dmin_mid)*temprand[2,setNd1_mid-1]
setDist[2,setNd1_far-1] = dmin_far + (dmax_far - dmin_far)*temprand[2,setNd1_far-1]

# Update seed for random
seedVar = 8
np.random.seed(seed = seedVar)


# Generate the source required for each component
source_min = 2
source_max = 6
setResource_node = np.round(source_min + (source_max - source_min)*np.random.rand(faultN,K))

# Generate the source available at each depot
seedVar = 14
np.random.seed(seed=seedVar)
setResource_depot = 6 * np.round(source_min + (source_max - source_min)*np.random.rand(depotN,K))

# Set Dcr = 1 for all depots and components: mean all depot can repair all components
Dcr = np.ones((depotN, faultN), dtype='uint8')

damage_components = np.arange(faultN)
depots = np.arange(depotN)
resources = np.arange(K)


# tuple list, where tuple (sigma, i) indicates component i must be allocated to depot sigma
N_M = [(1, 17)]



m_cluster,allocation = copt.cluster_solve(damage_components, depots, resources,
                            Dcr, setDist, setResource_node, setResource_depot, N_M)

# Handle the infeasible
if m_cluster.status == GRB.Status.INF_OR_UNBD:
    m_cluster.setParam(GRB.Param.Presolve, 0)
    m_cluster.optimize()

if m_cluster.status == GRB.Status.OPTIMAL:
    print('Optimal objective: %g' % m_cluster.objVal)
    # Convert the result to a matrix
    x_matrix = np.zeros((depotN, faultN), dtype='uint8')
    for key, x_value in allocation.items():
        x_matrix[key] = x_value.x

    # find the cluster node for each depot, dict with value to be a list
    # Get the result
    cluster_result = {}
    for d in depots:
        cluster_result[d] = np.where(x_matrix[d,:] == 1)[0].tolist()

    print(cluster_result)
    if m_cluster.status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % m_cluster.status)

    elif m_cluster.status == GRB.Status.INFEASIBLE:
        # Model is infeasible - compute IIS
        print('!!!!!!!!!!!')
        print('Model is infeasible, computing IIS')
        m_cluster.computeIIS()
        # m.write("clustering.ilp")
        # print('IIS')
        # print("IIS written to file 'clustering.ilp' ")
        for c in m_cluster.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)

# Partition the data set
#if flag_cluster == True:



workshift = 10
Tmax = workshift * 60
start_time = 0
objFlag = 2
print (objFlag)

for sigma in depots:

    crewNum = crewNumdepot[sigma]

    damageNum = len(cluster_result[sigma])

    nodeNum = damageNum + 1


    # extract information

    setResource_node_sub = setResource_node[(cluster_result[sigma]),0]
    # Insert the source for depot, which is equal to 0
    setResource_node_sub = np.insert(setResource_node_sub,0,0)

    # Get the resource for each crew
    setResource_crew = sourceCrewdepot[sigma]

    # travel time
    Tr = np.zeros((nodeNum, nodeNum, crewNum))
    dmin_mid = 35
    dmax_mid = 65

    for c in np.arange(crewNum):
        np.random.seed(seed = seedVar)
        seedVar += 1

        tempdistance = np.round(dmin_mid + (dmax_mid - dmin_mid) * np.random.rand(nodeNum, nodeNum))
        #tempdistance_tri = triu(tempdistance, 1);
        distance_G = tempdistance + np.transpose(tempdistance)
        Tr[:,:, c] = distance_G

    # Generate the repair time vector
    Rt_min = 50
    Rt_max = 90
    Rt = np.zeros((crewNum, nodeNum))
    for c in np.arange(crewNum):
        np.random.seed(seed = seedVar)
        seedVar += 1

        Rt[c, 1:] = np.round(Rt_min + (Rt_max - Rt_min) * np.random.rand(1, nodeNum - 1))

    Rt[:,0] = np.zeros(crewNum)
    Rt = np.transpose(Rt)

    # Define a big M value for the arrival time
    bigM = 12 * 60

    Set_N = np.arange(nodeNum)


    Set_R = np.arange(crewNum)



    Cap_c = {c: setResource_crew[c] for c in Set_R}
    Res_i = {i: setResource_node_sub[i] for i in Set_N if i != Set_N[0]}


    Tr_ijc = {(i,j,c): Tr[i,j,c] for i in Set_N for j in Set_N for c in Set_R}
    Et_ic = {(i,c): Rt[i,c] for i in Set_N for c in Set_R}

    Acc_i = np.zeros(nodeNum)
    #notaccess_idx = 3
    #Acc_i[notaccess_idx] = Tmax
    m, x, y, at = copt.poledeliver_solve(Set_N, Set_R, Cap_c, Res_i, Tr_ijc, Et_ic, Tmax, start_time, Acc_i, objFlag)


