# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:26:31 2017

@author: haridut_athi
"""
''' A TSP Problem for a warehouse for pick/place operation'''
''' Given: Order with Quantity, Distances, Location of each material in warehouse
    Output: Shortest Route to satisfy the order but with a constrain that emptying the bin is prioritized''' 

import networkx as nx, pandas as pd, numpy as np, datetime, itertools

t1 = datetime.datetime.now() #Start Time
order = 50 # Number of items to pick (Number on order)
# Creating a 2d array of distances between each bin location 
distances = pd.read_csv('output_file.csv', header = None) 
distances= np.asarray(distances)

# Adding all the bin location names to the list 'iopoints'
iopoints = pd.read_excel('Distance_info_Updated.xlsx',sheet_name=1)
iopoints = list(iopoints['Location'])

# Each bin location name is mapped to an index (index in distances matrix)
dic={}
for i in range(len(iopoints)):
    dic[iopoints[i]] = i
# Rack name mapping to Material (SKUs)
skumap = pd.read_csv('Location-SKU.csv')
#Remove all duplicates of Material and Bin Location (where both entries are the same)
skumap_1 = skumap.drop_duplicates(subset=['Material','BIN_LOCATION'], keep='first')

# Final map is a dictionary of the format {'Material no': {'Bin Location': [Index,Quantity]}}
final_map={}
for i in set(skumap_1['Material']):
    temp_dic={}
    bins = np.array(skumap_1[skumap_1['Material']==i]['BIN_LOCATION'])
    for j in bins: 
        temp_dic[j] = [dic[j],skumap_1[(skumap_1['Material']==i) & (skumap_1['BIN_LOCATION']==j)]['Quantity'].iloc[0]]
        final_map[i] = temp_dic

# We randomly pick materials from the existing materials (Ensuring Origin is not in Materials)
material = np.random.choice(skumap_1['Material'][1:], order-1)
material = np.insert(material, 0, 1111, axis=0) # First material is 'Origin' (material no:'1111') 
quant_req = np.random.randint(1,order, size=order-1) #Random generation of quantity required -each material in order
quant_req = np.insert(quant_req, 0, 0, axis=0) # Insert Origin's quantity as '0' (first material)


# Create 3 lists each with bin_location name, index and quantity available for each material in the order:
# A material may be available at many locations and all of these locations, quantity and indices are added    
location=[]
indices =[]
quantity=[]
for i in material:
    location.append(list(final_map[i].keys())) # 
    indices.append(list(np.array(list(final_map[i].values()))[:,0]))
    quantity.append(list(np.array(list(final_map[i].values()))[:,1]))

# Sorting multiple locations based on available quantity 
# The one with the lowest quantity for a specific material comes first
# All the 3 lists are sorted simultaneously
for i in range(order):
    if len(location[i])>1:
        quantity[i],location[i],indices[i] = (list(t) for t in zip(*sorted(zip(quantity[i],location[i],indices[i]))))

# Retain only the first 'n' locations that cumulatively add up to the demand for each material
# All the list are updates correspondingly
for i in range(order):
    if len(location[i])>1:
        met=0
        for j in range(len(location[i])):
            met+=quantity[i][j]
            if met > quant_req[i]:
                location[i] = location[i][:j+1]
                quantity[i] = quantity[i][:j+1]
                indices[i] = indices[i][:j+1]
                break

# Creates a single list of all indices from 'indices' list for creating the graph nodes 
single_list=[]
for i in range(len(indices)):
    single_list.append([indices[i][0]])
    if len(indices[i])>1:
        for j in range(len(indices[i])-1):
            single_list.append([indices[i][j+1]])
single_list = list(set(list(itertools.chain.from_iterable(single_list)))) # Unpack list of lists  
 

G = nx.Graph() #Graph instance
G.add_nodes_from(single_list) # Add nodes from single list (nodes are labeled with the indices itself)
#Adding edge for all nodes except to the same node (weighted by distances)
for i in range(len(single_list)):
    for j in range(len(single_list)):
        if i!=j:
            G.add_edge(single_list[i],single_list[j],weight=distances[single_list[i],single_list[j]])

# Minimum Spanning tree by Prim's algorithm
mst = nx.minimum_spanning_edges(G, data=False) 
# All the edges from the MST are added to 'edge_list'
edge_list = list(mst)

# Create a Graph out of the MST
G2 = nx.Graph()

#Creaing edges as per the MST
G2.add_edges_from(edge_list)


# MST based heuristic for solving the Traveling Salesman Problem'
'''Set_A: Intially empty, nodes get added when they are visted
   Set_B: It has all the first choice of location indices for each material
   Set_C: It has both the first choice and other locations for materials that are available at multi-locations'''
list_tour=[] 
list_cost =[]
for iteration in range(20): #Repeat the process many times since there is certain randomness in the algorithm
    set_A = [455] # The first bin to visit will always be origin (455 index)
    set_B = list(set([i[0] for i in indices])) # Set_B created using the first index of each material from 'indices' list
    set_C = [i  for i in indices if len(i)>1] # Set_C created only for materials with multi-locations
    set_C = (list(itertools.chain.from_iterable(set_C))) #Unpack
    # The block removes from set_C all duplicates of indices by retaining only the last occurence
    # This is to maintain the order of pick-up (smallest to largest quantity) #Emptying bins is the priority
    set_C2 =[]
    for i in range(1,len(set_C)+1):
        if set_C[-i] not in set_C2:    
            set_C2.append(set_C[-i])
    set_C = set_C2[::-1]
    
    #The algorithm keeps looping through until set_B is empty(All locations visited)
    while set_B:
        last=set_A[-1] 
        flag=True
        neighbors = G2.neighbors(last) #Search for the neighbors of the last location of set_A in our graph
        for i in neighbors: #Search all neigbors
            if (i not in set_A) and (i in set_B): # If the neighbor is not in tour then
                flag=False
                set_A.append(i) #Add it to tour (set_A - visited)
                set_B.remove(i) #Remove from the all-nodes (set_B) list
                break # Exit the search upon finding 'the first' possible neighbor
        # Only if none of the neighbors are not candiates to enter, we randomly pick a location to be visited from 
        # candidate list
        if flag: 
            ran = (np.random.choice(set_B)) #If no neighbor is found, randomly jump to one node in all-node list
            set_A.append(ran) #Put the randomly picked choice to visted list
            set_B.remove(ran) #Remove it from the candidate list
        
        # All the restriction in terms of multi-location pick up for an item is taken care here
        # Objective is to empty the bin if possible 
        for k in set_C:
            if (k in set_A) and (set_C.index(k)!=len(set_C)-1):
                if (set_C[set_C.index(k)+1] not in set_A) and (set_C[set_C.index(k)+1] not in set_B):
                    set_B.append(set_C[set_C.index(k)+1])
                if (set_C[set_C.index(k)+1] in set_A):
                    set_C.remove(set_C[set_C.index(k)+1])
                set_C.remove(k)
    tour = set_A
    tour.append(455) # Add origin to the end so that tour is complete
    total = 0
    #Calculate distance of the tour for each iteration
    for i in range(len(tour)-1):
            total=total+distances[tour[i],tour[i+1]]
    list_tour.append(tour) # a list of all tours across iterations
    list_cost.append(total) # Cost of all tours across iterations
best_tour =list_tour[np.argmin(list_cost)] #Choose the least cost tour
best_cost = min(list_cost) #Least cost

# Calculate how a person would satisfy the order without any sort of routing (Pick material one by one as per the order)
bad_cost = 0
for i in range(len(single_list)-1):
    bad_cost += distances[single_list[i],single_list[i+1]]
bad_cost+= (distances[455, single_list[0]] + distances[455,single_list[-1]])

#Printing out the final route, Distance of that tour and Distance of walking without routing
final_route=[]
for t in best_tour:
    final_route.append([i for i in list(dic.keys()) if dic[i]==t][0])
print('---------------------------------------------------------------------')
print('The shortest route:')
for i in range(len(final_route)):
    if i!=len(final_route)-1:
        print(final_route[i]+ ' -----> ', end='')
    else:
        print(final_route[i])
t2 = datetime.datetime.now()
print('---------------------------------------------------------------------')
print('Time: '+str(t2-t1))
print('Best cost: '+str(best_cost))
print('Bad_Cost: ' +str(bad_cost))