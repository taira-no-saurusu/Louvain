
from __future__ import print_function
import collections
import copy
import networkx as nx
import numpy as np
from community_status import Status

#パスのマックス
__PASS_MAX = -1

#最小値
__MIN = 0.0000001

#配列表示形式変更、省略なくす（np.inf : 無限大）
np.set_printoptions(threshold=np.inf)

class data:
    """
    fileName : ファイル名（string）
    """
    def __init__(self, fileName):
        self.fileName = fileName
        self.information(self.fileName)

    """
    任意のグラフインスタンスを生成
    """
    def information(self, fileName):

        #グラフインスタンス生成
        G = nx.Graph()

        # テキストファイルから読み取りndarray化
        # edge : (始点,終点,重み)の3-tupleのリスト
        edge = np.genfromtxt(fileName + ".txt",delimiter=",")
        self.edge = edge

        # (始点,終点,重み)の3-tupleのリストから重み付きエッジをまとめて追加
        G.add_weighted_edges_from(edge)
        self.G = G


class louvain:

    def __init__(self, c = 0,dataName = "None") -> None:
        self.c = c
        self.dataName = dataName
        pass

    #cのゲッター
    def get_c(self):
        return self.c

    def fit(self):

        #dataインスタンス生成
        d = data(self.dataName)
    
    """
    graph : グラフインスタンス
    counter : 
    part_init : 
    weight : 
    resolution : 
    randomize : 
    """
    def generate_dendrogram(graph, counter, c, part_init=None,weight="weight", resolution=1., randomize=False):
        
        #例外処理（有効グラフを除外）
        if graph.is_directed():
            raise TypeError("Directed_graph can not be used!!!")
        
        #グラフのエッジがないとき、ノード番号をキーとして自身のノード番号を要素にもつ辞書型を返す
        if graph.nomber_of_edges() == 0:
            part = dict([])
            for node in graph.nodes():
                part[node] = node
            return [part]
        
        current_graph = graph.copy()
        status = Status()
        status.init(current_graph,weight,part_init)
        status_list = list()
        status = __one_level(current_graph,status,weight,resolution,randomize,c)

        #途中


    
    def __one_level(graph,status,weight_key, resolution, randomize, c):

        modified = True
        nb_pass_done = 0
        cur_mod = __modularity(status)
        new_mod = cur_mod
        isFalse = True

        while modified and nb_pass_done != __PASS_MAX:
            cur



    def __modularity(status):

        links = float(status.total_weight)
        result = 0.
        for community in set(status.node2com.values()):
            in_degree = status.internals.get(community, 0.)
            degree = status.degrees.get(community, 0.)
            if links > 0:
                result += in_degree / links - ((degree / (2. * links)) ** 2)
        return result
