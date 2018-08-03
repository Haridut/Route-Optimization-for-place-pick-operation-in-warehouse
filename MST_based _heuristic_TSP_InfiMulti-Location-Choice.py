# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:00:03 2017

@author: haridut_Athis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' A TSP Problem for a warehouse for pick/place operation in case of multiple location each with infinite quantity'''
''' Given: Order without Quantity, Distances, Location (Might be multiple) of each material in warehouse
    Output: Shortest Route to satisfy the order ''' 

import networkx as nx, pandas as pd, numpy as np, datetime, itertools




np.random.seed(0)
t1 = datetime.datetime.now() #Start Time
order = 50 # Number of items to pick (Number on order)
# Creating a 2d array of distances between each bin location 
distances = pd.read_csv('output_file.csv', header = None) 
distances= np.asarray(distances)

# Adding all the bin location names to the list 'iopoints'
iopoints = pd.read_excel('Distance_info_Updated.xlsx',sheetname=1)
iopoints = list(iopoints['Location'])

# Each bin location name is mapped to an index (index in distances matrix)
dic={}
for i in range(len(iopoints)):
    dic[iopoints[i]] = i

#Rack name mapping to Material (SKUs)
skumap = pd.read_csv('Location-SKU.csv')
#Remove all duplicates of Material and Bin Location (where both entries are the same)
skumap_1 = skumap.drop_duplicates(subset=['Material','BIN_LOCATION'], keep='first')

# Final map is a dictionary of the format {'Material no': {'Bin Location': [Index]}}
final_map={}
for i in set(skumap_1['Material']):
    temp_dic={}
    bins = np.array(skumap_1[skumap_1['Material']==i]['BIN_LOCATION'])
    for j in bins: 
        temp_dic[j] = dic[j]
    final_map[i] = temp_dic
    
# We randomly pick materials from the existing materials (Ensuring Origin is not in Materials)
material = np.random.choice(skumap_1['Material'], order-1)
#Insert '111' as a dummy for Origin in materials at the start
material = np.insert(material, 0, 1111, axis=0)

#Create a list with bin_location name for each material in the order:
location=[]
for i in material:
    location.append(list(final_map[i].keys()))

#The following block of code is a heuristic that eliminates all locations for a material except for the corner locations if corner exists
# If corner does not exist, keep all the locations
# Hence, we reduce the candidate points for enumeration

corner_dic = {'D': ['A', 'D', 'F', 'I', 'J', 'L'], 'W': ['A', 'D', 'F', 'I', 'K', 'M']} #Corner points 

#Keeping only corner elements if present, else retain all locations
for i in range(len(location)):
    if len(location[i])>1:
        tmp = [list(g) for _, g in itertools.groupby(location[i], lambda x: x[:3])]
        to_keep=[]
        for j in tmp:
            flag = True
            for k in j:
                if k[-1] in corner_dic[k[0]]:
                    flag=False
                    to_keep.append([k])
            if flag:
                flag1=True
                to_keep.append(j)
              
        to_keep = list(itertools.chain.from_iterable(to_keep))
        location[i] = to_keep

# If a location is shared by 2 or more materials, it is best that we go to the same location to pick all the materials possible
# Hence, if intersection of locations exists, retain only those locations and remove the rest
for i in range(len(location)):
    if len(location[i])>1:
        temp =[]
        for j in range(len(location)):
            if location[i]!=location[j]:
                if (len(set(location[i]).intersection(location[j]))!=0) and ((list(set(location[i]).intersection(location[j]))) not in temp): 
                    temp.append(list(set(location[i]).intersection(location[j])))
        temp = list(itertools.chain.from_iterable(temp)) 
        if len(temp)!=0:
            location[i] = temp

# Build a list of corresponding indices to map it to distance matrix (from location list)
indices=[]
for i in location:
    temp=[]
    for j in i:
        temp.append(dic[j])
    indices.append(temp)     
 
'''Procedure: Calculate shortest route assuming we pick the first location among the multiple-location choices
   followed by the pertubation of this solution to get a better route by evaluating all multi-choices '''
#Constructing a Graph with all nodes (0 to order quantity):
# 'Bin list keeps track of location of each material in the  order
bin_list=[]    
G = nx.Graph()
G.add_nodes_from(list(range(order)))
#Adding edge for all nodes except to the same node weighted by distance (We always pick the first index)
for i in range(order):
    for j in range(order):
        if i!=j:
            G.add_edge(i,j,weight=distances[indices[i][0],indices[j][0]])
    bin_list.append(location[i][0])

#Fininding the Minimum Spanning Tree
mst = nx.minimum_spanning_edges(G, data=False)
edge_list = list(mst)

#Graph out of the minimum spanning tree
G2 = nx.Graph()

#Creaing edges as per the MST
G2.add_edges_from(edge_list)

'''Set_A: Intially empty, nodes get added when they are visted
   Set_B: It has all the first choice of location indices for each material'''
list_tour=[]
list_cost =[]
for iteration in range(1000): #1000 runs through the heuristic
    set_A = [0] # The first bin to visit will always be origin (0)
    set_B = list(range(0,order)) # List of all candidates
    set_B.remove(set_A[0]) # Remove the orign from set_B as it is in set_A already
    while set_B: #If B is not empty keep doing
        last=set_A[-1] 
        flag=True
        neighbors = G2.neighbors(last)#Search for the neighbors of the last location of set_A in our graph
        for i in neighbors: #Search all neigbors
            if i not in set_A: # If the neighbor is not in tour then
                flag=False
                set_A.append(i) #Add it to tour
                set_B.remove(i) #Remove from the all-nodes list
                break    # Exit the search upon finding 'the first' possible neighbor
        #The algorithm keeps looping through until set_B is empty(All locations visited)
        if flag:
            ran = (np.random.choice(set_B)) #If no neighbor is found, randomly jump to one node in all-node list
            set_A.append(ran)#Add it to tour
            set_B.remove(ran) #Remove it from the candidate list
            
    tour = set_A
    tour.append(0) # Add origin to the end so that tour is complete
    
    #Calculate distance of the tour for each iteration
    total =0
    for i in range(len(tour)-1):
        total=total+distances[indices[tour[i]][0],indices[tour[i+1]][0]]
    list_tour.append(tour) # a list of all tours across iterations
    list_cost.append(total) # Cost of all tours across iterations

# Calculate how a person would satisfy the order without any sort of routing (Pick material one by one as per the order)
bad_cost = 0
for i in range(order-1):
    bad_cost += distances[indices[i][0],indices[i+1][0]]
bad_cost+=distances[indices[0][0],indices[-1][0]]

#Best tour and cost among the 1000 tours found
best_tour =list_tour[np.argmin(list_cost)] 
best_cost = min(list_cost)

#Printing out the final route, Distance of that tour.
final_route=[]
for i in best_tour:
    final_route.append(bin_list[i])
print('---------------------------------------------------------------------')
print('The shortest route for Heuristic_1')
for i in range(len(final_route)):
    if i!=len(final_route)-1:
        print(final_route[i]+ ' -----> ', end='')
    else:
        print(final_route[i])

#Keeping the best tour fixed and finding best location in a multi-location choice scenario:
for i in range(len(best_tour)):
    if len(indices[best_tour[i]])>1:
        temp_d = 1000000
        for j in range(len(indices[best_tour[i]])):
            for k in range(len(indices[best_tour[i+1]])):
                dist = distances[indices[best_tour[i-1]][0],indices[best_tour[i]][j]] + distances[indices[best_tour[i]][j], indices[best_tour[i+1]][k]]
                if dist<temp_d:
                    temp_d = dist
                    index_k = k
                    index_j = j
        bin_list[best_tour[i]] = location[best_tour[i]][index_j]
        bin_list[best_tour[i+1]] = location[best_tour[i+1]][index_k]
        indices[best_tour[i]] = [indices[best_tour[i]][index_j]]
        indices[best_tour[i+1]] = [indices[best_tour[i+1]][index_k]]

#Calculate distance of the modifided tour for each iteration
total_new=0
for i in range(len(best_tour)-1):
        total_new=total_new+distances[indices[best_tour[i]][0],indices[best_tour[i+1]][0]]

#Printing out the final new route, Distance of that tour and Distance of walking without routing
final_route_new=[]
for i in best_tour:
    final_route_new.append(bin_list[i]) 

print('---------------------------------------------------------------------')
print('The shortest route for Heuristic_2')
for i in range(len(final_route_new)):
    if i!=len(final_route_new)-1:
        print(final_route_new[i]+ ' -----> ', end='')
    else:
        print(final_route_new[i])
t2 = datetime.datetime.now()
print('---------------------------------------------------------------------')
print('Time: '+str(t2-t1))
print('Heuristic-1 cost: '+str(best_cost))
print('Heuristic-2 cost: '+str(total_new))
print('Bad_Cost: ' +str(bad_cost))

    
