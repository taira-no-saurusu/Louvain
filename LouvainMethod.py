#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:40:26 2018

@author: shu-san
"""

from __future__ import print_function


import collections
import copy
import networkx as nx
import numpy as np

from community.community_status import Status


__PASS_MAX = -1
__MIN = 0.0000001
np.set_printoptions(threshold=np.inf)

class data:
    def __init__(self, fileName):
        self.fileName = fileName
        self.information(self.fileName)

    def information(self, fileName):
        G = nx.Graph()
        edge = np.genfromtxt(fileName + ".txt", delimiter=",")

        G.add_weighted_edges_from(edge)
        self.G = G
        self.edge = edge

class louvain:
    
    def __init__(self ,c = 0 ,dataName = "None"):
        self.c = c
        self.dataName = dataName
        
    def get_c(self):
        return self.c
            
    def fit(self):
        
        #クラスタ数がラベル数よりも大きい場合は、generate_dendrogramおよびpartition_at_levelの第二引数を変更
        #例 クラスタ数12,ラベル数5 -> 第二引数をクラスタ数12-ラベル数5=7に設定
        d = data(self.dataName)
        dendro = generate_dendrogram(d.G,0,self.c)
        kekka = partition_at_level(dendro, 0)
   
        cLabel = list(map(int, kekka.values()))
        return cLabel

    
def partition_at_level(dendrogram,level):
    partition = dendrogram[0].copy()
    for index in range(1,level+1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition

def modularity(partition,graph,weight="weight"):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")
    
    inc = dict([])
    deg = dict([])
    links = graph.size(weight= weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")
        
    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com,0.)+graph.degree(node,weight=weight)
        for neighbor,datas in graph[node].items():
            edge_weight = datas.get(weight,1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com,0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com,0.) + float(edge_weight) / 2
    res = 0.
    print (set(partition.values()))
    for com in set(partition.values()):
        res += (inc.get(com,0.)/links) - \
               (deg.get(com,0.) / (2. * links)) ** 2
    
    return res

def generate_dendrogram(graph,counter,c,part_init=None,weight="weight",resolution=1.,randomize=False):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")
        
    if graph.number_of_edges() == 0:
        part = dict([])
        for node in graph.nodes():
            part[node] = node
        return [part]
    
    current_graph = graph.copy() 
    status = Status()
    status.init(current_graph,weight,part_init)  
    status_list = list()
    status = __one_level(current_graph,status,weight,resolution,randomize,c)
    
    new_mod = __modularity(status)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition,current_graph,weight)
    status.init(current_graph,weight)
    #print (status)

    count = 0
    
    while True:
        print ("level : クラスタ数",len(status.node2com.values()))
        status = __two_level(current_graph,status,weight,resolution,randomize)
        new_mod = __modularity(status)
        if count == counter:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
        count += 1 #
    return status_list[:]

def __one_level(graph, status, weight_key, resolution, randomize, c):
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status)
    new_mod = cur_mod
    isFalse = True

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1
        for node in graph.nodes():
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = __neighcom(node, graph, status, weight_key)
            remove_cost = - resolution * neigh_communities.get(com_node,0) + \
                (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in neigh_communities.items():
                incr = remove_cost + resolution * dnc - \
                       status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com, neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True
            
            #クラスタ数が正解ラベルより低い場合は、ここの値をラベル数に変更
            if len(collections.Counter(status.node2com.values())) == c:
                #print(collections.Counter(status.node2com.values()))
                #print("ok")
                isFalse = False
                break
        if isFalse == False:
            break
        #print(max(status.node2com.values()))
        new_mod = __modularity(status)
        if new_mod - cur_mod < __MIN:
            break
    return status


def __two_level(graph, status, weight_key, resolution, randomize):
    modified = True
    nb_pass_done = 0
    cur_mod = -99999999
    new_mod = cur_mod
    nstatus = copy.deepcopy(status)##
    truestatus = None
    
    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1
        
        for node in graph.nodes():
            status = copy.deepcopy(nstatus)##
            
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)
            neigh_communities = __neighcom(node, graph, status, weight_key)
            remove_cost = - resolution * neigh_communities.get(com_node,0) + (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            __remove(node, com_node,neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = -9999999
            for com, dnc in neigh_communities.items():
                incr = remove_cost + resolution * dnc - status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com, neigh_communities.get(best_com, 0.), status)
            new_mod = __modularity(status)
            if new_mod >= cur_mod:
                cur_mod = new_mod
                truestatus = copy.deepcopy(status)
    status = copy.deepcopy(truestatus)
    return status
    
def induced_graph(partition, graph, weight="weight"):
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret

def __neighcom(node, graph, status, weight_key):
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights

def __remove(node, com, weight, status):
    
    status.degrees[com] = (status.degrees.get(com, 0.) - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) - weight - status.loops.get(node, 0.))
    status.node2com[node] = -1

def __insert(node, com, weight, status):
    
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) + status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) + weight + status.loops.get(node, 0.))

def __modularity(status):
    
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree / links - ((degree / (2. * links)) ** 2)
    return result

def __renumber(dictionary):
    count = 0
    ret = dictionary.copy()
    new_values = dict([])
    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value

    return ret


def adjacencymatrix(G):
    # 隣接行列の生成
    dis = np.array(nx.to_numpy_matrix(G))
    return dis
    
        
        
    
