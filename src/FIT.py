import os
import sys
import pandas as pd
import numpy as np

import gurobipy as gp
from gurobipy import GRB

import config

# Budget Configuration
if len(sys.argv) > 1:
    B = float(sys.argv[1])
else:
    # Script input or user input
    B = float(input()) 

# Importing configurations from config module
obj = config.OBJ
priority = config.PRIORITY
result_folder = config.RESULT_FOLDER
city_edge_list = config.CITY_EDGE_LIST
city_od = config.CITY_OD
distance_type = config.DISTANCE_TYPE
alpha = config.ALPHA
gamma = config.GAMMA
run = config.RUN

'''=========================================================
Load Underlying Network
========================================================='''
data = pd.read_csv("../../networks/{city_edge_list}.csv".format(city_edge_list=city_edge_list))

# identifying starting and ending nodes, and c
start_node = data.iloc[:,0].tolist()
end_node = data.iloc[:,1].tolist()
if (distance_type == "center"):
    c = data.iloc[:,2]
elif (distance_type == "osm"):
    c = data.iloc[:,3]

#count num of unique nodes
nodes = set(start_node + end_node)
arcs = []
for i in range(len(start_node)):
    if start_node[i] != end_node[i]: #eliminates self loops
        arcs.append((start_node[i], end_node[i]))
    else:
        #remove any self-loops costs
        del c[i]

# n = num nodes, m = num arcs
n = len(set(start_node + end_node))
m = len(arcs)

#get all permuations/(o,d) pairs
data = pd.read_csv("../../networks/l_star/{city_od}.csv".format(city_od=city_od))

start_node = data.iloc[:,0]
end_node =  data.iloc[:,1]
if (distance_type == "center"):
    l_star = data.iloc[:,2] #shortest path
elif (distance_type == "osm"):
    l_star = data.iloc[:,3] #shortest pat:

d = data.iloc[:,4] #demand 
if (priority == "P"):
    p = data.iloc[:,5] #priority

#store all od pairs in list 
od = []
for i in range(len(start_node)):
    od.append((start_node[i], end_node[i]))

# list of list per node (in & out)
inc = []
outgo = []
for i in nodes: #for every node
    incoming = []
    outgoing = []
    for j in range(len(arcs)): #for every arc pair
        if(arcs[j][1]==i): #if destination is equal to node
            incoming.append(arcs[j])
        if(arcs[j][0]==i): #if origin is equal to node
            outgoing.append(arcs[j])
    inc.append(incoming)
    outgo.append(outgoing)

in_dict = dict(zip(nodes, inc))
out_dict = dict(zip(nodes, outgo))

#cost dict
c_dict = dict(zip(arcs,c))

# demand dict
b_dict = dict(zip(od,d))

#priority dicts
if (obj == "MaxMin"):
    if (priority == "P"):
        p_dict = dict(zip(od,p))
        p_dict_mm = dict(zip(od, -(p-1)))
    elif (priority == "EP"):
        p_dict_mm = dict.fromkeys(od,1)
        p_dict = dict.fromkeys(od,1)  
elif (obj == "Utilitarian" or obj == "Gini"): 
    if (priority == "P"):
        p_dict = dict(zip(od, p))
    elif (priority == "EP"):
        p_dict = dict.fromkeys(od,1)
elif (obj == "LinearCombo"):
    if (priority == "P"):
        p_dict = dict(zip(od,p))
        p_dict_mm = dict(zip(od, -(p-1)))
    elif (priority == "EP"):
        p_dict = dict.fromkeys(od,1)
        p_dict_mm = dict.fromkeys(od,1)

#length of arcs dict
la_dict = dict(zip(arcs,c))

#pre computed shortest path
l_star_dict = dict(zip(od, l_star))

'''
=========================================================
Write to files
=========================================================
'''
def save_results(od, arcs):
    path = '../../{run}_{result_folder}/{obj}/alpha{alpha}/gamma{gamma}/{priority}/{B}'.format(run=run,result_folder=result_folder,obj=obj, alpha=alpha, gamma=gamma, priority=priority, B=B)
    if not os.path.isdir(path):
        os.makedirs(path)
    f = open("../../{run}_{result_folder}/{obj}/alpha{alpha}/gamma{gamma}/{priority}/{B}/objective.csv".
             format(run=run,result_folder=result_folder,obj=obj, alpha=alpha, gamma=gamma, priority=priority, B=B), "w")
    f.write(str(model.objVal))
    
    df = pd.read_csv("../../networks/{city_edge_list}.csv".format(city_edge_list=city_edge_list))

    x_list = []
    y_list = []
    l_list = []
    u_list = []

    utilities_dict = {}
    y_dict = {}
    l_dict = {}
    x_dict = {}

    for r in od:
        s = str(r)
        s = s.replace('(','').replace(')','').replace(' ','')
        utilities_dict[r] = model.getVarByName(f"u_var[{s}]").X
        y_dict[r] = model.getVarByName(f"y[{s}]").X
        l_dict[r] = model.getVarByName(f"l[{s}]").X

    for a in arcs:
        s = str(a)
        s = s.replace('(','').replace(')','').replace(' ','')
        x_dict[a] = model.getVarByName(f"x[{s}]").X

    for a in arcs:
        x_list.append(x_dict[a])
    df["x"] = x_list

    df.to_csv("../../{run}_{result_folder}/{obj}/alpha{alpha}/gamma{gamma}/{priority}/{B}/plot_data.csv"
              .format(run=run,result_folder=result_folder,obj=obj,alpha=alpha,gamma=gamma,priority=priority,B=B))

    df2 = pd.read_csv("../../networks/l_star/{city_od}.csv".format(city_od=city_od))
    for r in od:
        u_list.append(utilities_dict[r])
        y_list.append(y_dict[r])
        l_list.append(l_dict[r])

    df2["y"] = y_list
    df2["l"] = l_list
    df2["u"] = u_list
    df2['od'] = od

    df2.to_csv("../../{run}_{result_folder}/{obj}/alpha{alpha}/gamma{gamma}/{priority}/{B}/all_data.csv"
               .format(run=run,result_folder=result_folder, obj=obj,alpha=alpha,gamma=gamma,priority=priority,B=B))

'''
=========================================================
Choose solvers and options
=========================================================
'''

model = gp.Model()
model.setParam("Method", 2)
model.setParam("MIPGap", 1e-2)
model.setParam("MIPFocus", 1)

'''
#=========================================================
Problem Space
=========================================================#
'''
# Decision variables
x = model.addVars(arcs, vtype=gp.GRB.BINARY, name="x")
y = model.addVars(od,vtype=gp.GRB.BINARY, name="y") 
f = model.addVars(arcs,od,vtype=gp.GRB.BINARY, name="f")
l = model.addVars(od,vtype=GRB.CONTINUOUS, lb = 0, name="l")
u_var = model.addVars(od,vtype=GRB.CONTINUOUS, lb = 0, name="u_var")
w = model.addVars(od,od, vtype=GRB.CONTINUOUS, lb=0, name="w")
w_hold = model.addVars(od,od, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="w_h")
model.update()

# ## If previous solution exists, load and use 
if os.path.isfile("solution_{run}_{result_folder}_{priority}_{obj}.sol".format(run=run, result_folder=result_folder, priority=priority, obj=obj)):
    model.read("solution_{run}_{result_folder}_{priority}_{obj}.sol".format(run=run,result_folder=result_folder, priority=priority, obj=obj))
    model.update()

# budget constraint
model.addConstr((sum(c_dict[a]*x[a] for a in arcs) <= B), name="budget")

# mass conservation constraint
for u in nodes:
    model.addConstr((sum(x[a] for a in out_dict[u]) - sum(x[a] for a in in_dict[u])) == 0, name=f"mass_conservation[{u}]")

# mass conservation everywhere but at sink or source nodes 
for u in nodes:
    for r in od:
        if u == r[0]:        #origin
            indicator = 1
        elif u == r[1]:    #destintation
            indicator = -1
        else:               #neither
            indicator = 0
        model.addConstr((sum(f[a + r] for a in out_dict[u]) - sum(f[a+ r] for a in in_dict[u])) == (y[r] * indicator), name=f"mass_conservation_ss[{u}]")

# installation constraint
for a in arcs:
    for request in od:
        model.addConstr(f[a + request] <= x[a], name=f"install[{a}]")

# shortest path constraint
for request in od:
    model.addConstr(l[request] == sum(la_dict[a] * f[a + request] for a in arcs), name=f"short_path[{request}]")

# unserved request constraint
for request in od:
    model.addConstr(l[request] <= ((alpha * l_star_dict[request]) * y[request]), name=f"unserved[{request}]")

# new utility constraint
for r in od:
    model.addConstr(u_var[r] == 1/(l_star_dict[r]*(alpha-1))* (alpha*l_star_dict[r]-l[r])-alpha/(alpha-1)*(1-y[r]), name=f"utility[{r}]")

'''
#=========================================================
Objectives 
=========================================================#
'''
if (obj == "Utilitarian"):
    model.setObjective(sum(b_dict[r] * (p_dict[r] * u_var[r]) for r in od), GRB.MAXIMIZE)

elif (obj == "MaxMin") or (obj == "LinearCombo"):

    objs = model.addVars([1,2])
    norm_factor = 0
    for r in od:
        norm_factor += b_dict[r] 
    model.addConstr(objs[1] <= sum(b_dict[r] * (p_dict[r] * u_var[r]) for r in od)/norm_factor)
    for r in od:
        model.addConstr(objs[2] <=  p_dict_mm[r] * u_var[r])
    model.setObjective(gamma * objs[1] + (1 - gamma) * objs[2], GRB.MAXIMIZE)

'''
#=========================================================
Solve models
=========================================================#
'''
model.optimize()
save_results(od, arcs)
model.write("solution_{run}_{result_folder}_{priority}_{obj}.sol".format(run=run,result_folder=result_folder, priority=priority, obj=obj))
