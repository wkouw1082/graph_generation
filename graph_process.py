from time import time_ns
from networkx.classes import graph
from networkx.readwrite import json_graph
import numpy as np
import networkx as nx
import torch
import dgl
from collections import OrderedDict
import networkx.algorithms.approximation.treewidth as nx_tree
import networkx.algorithms.community as nx_comm
import community
from community import community_louvain
import random
import matplotlib.pyplot as plt
import joblib
import utils
from scipy.optimize import curve_fit
import sympy as sym
from sympy.plotting import plot
import config
import sys
import time
import math
import json
import glob
from tqdm import tqdm
import csv
import os
from sklearn.model_selection import train_test_split
from config import *

# 複雑ネットワークを返すクラス
# datasetはnetworkxのobjのlist
class complex_networks():

    def create_dataset(self, detail, do_type='train'):
        datasets = []
        for i, (key, value) in enumerate(detail.items()):
            generate_num = value[0]
            data_dim = value[1]
            params = value[2]

            for param in params:
                if key == "BA":
                    datas = self.generate_BA(generate_num, data_dim)
                elif key == "fixed_BA":
                    datas = self.generate_fixed_BA(generate_num, data_dim)
                elif key == "NN":
                    datas = self.nearest_neighbor_model(generate_num, data_dim, param)
                elif key == "twitter_train":
                    datas = self.make_twitter_graph('train')
                elif key == "twitter_valid":
                    datas = self.make_twitter_graph('valid')
                elif key == "twitter":
                    datas = self.make_twitter_graph()
                elif key == "reddit":
                    datas = self.make_reddit_graph()
                elif key == "twitter_pickup":
                    datas = self.pickup_twitter_data()
                # NNモデルでの生成時にはこっちを使う　いろんなparamのデータをまとめて一つのデータセットにするため
                if do_type == 'train':
                    datasets.extend(datas)
                elif do_type == 'visualize':
                    # visualizeのみはこっちを使う　paramを分けてデータを分析したいため
                    datasets.append(datas)
                else:
                    exit(1)
        return datasets

    def create_conditional_dataset(self,detail):
        datasets = []
        labelsets = torch.Tensor()
        for i, (key,value) in enumerate(detail.items()):
            # 本来ならlen(power_degree_lable):len(cluster_coefficient_label)がサイズ　簡略化してるためこの数字
            generate_counter = [0]*len(power_degree_label)
            while True:
                generate_num = value[0]
                data_dim = value[1]
                params = value[2]
                end_flag = True
                for param in params:
                    print(generate_counter)
                    if key == "BA":
                        data = self.generate_BA(1, data_dim)
                    elif key == "fixed_BA":
                        data = self.generate_fixed_BA(1, data_dim)
                    elif key == "NN":
                        data = self.nearest_neighbor_model(1, data_dim, param)
                    
                    st = graph_statistic()

                    if len(datasets) == generate_num:
                        break

                    if generate_counter[0] != math.ceil(generate_num/3) and round(st.degree_dist(data[0]),1) == -power_degree_label[0] and abs(round(nx.average_clustering(data[0]),1)) == cluster_coefficient_label[0]:
                        generate_counter[0] += 1
                        labelsets = torch.cat((labelsets,self.create_label(0).unsqueeze(0)),dim=0)
                        datasets.extend(data)
                    
                    if generate_counter[1] != math.ceil(generate_num/3) and round(st.degree_dist(data[0]),1) == -power_degree_label[1] and abs(round(nx.average_clustering(data[0]),1)) == cluster_coefficient_label[1]:
                        generate_counter[1] += 1
                        labelsets = torch.cat((labelsets,self.create_label(1).unsqueeze(0)),dim=0)
                        datasets.extend(data)
                    
                    if generate_counter[2] != math.ceil(generate_num/3) and round(st.degree_dist(data[0]),1) == -power_degree_label[2] and abs(round(nx.average_clustering(data[0]),1)) == cluster_coefficient_label[2]:
                        generate_counter[2] += 1
                        labelsets = torch.cat((labelsets,self.create_label(2).unsqueeze(0)),dim=0)
                        datasets.extend(data)
                if len(datasets) == generate_num:
                    break
        return datasets, labelsets.unsqueeze(1)

    def create_seq_conditional_dataset(self,detail):
        for i, (key, value) in enumerate(detail.items()):
            generate_num = value[0]
            data_dim = value[1]
            params = value[2]

            for param in params:
                if key=='twitter_train':
                    datasets = joblib.load(twitter_train_path)
                    dataset, label = datasets[0], datasets[1]
                elif key=='twitter_valid':
                    datasets = joblib.load(twitter_valid_path)
                    dataset, label = datasets[0], datasets[1]

        return dataset, label

    # 俗に言う修正BAモデルの生成
    def generate_fixed_BA(self, generate_num, data_dim):
        # print("     generate fixed BA:")
        datas = []

        for i in range(generate_num):
            data = np.zeros((data_dim, data_dim))
            if i % 1000 == 0:
                print("     [%d/%d]"%(i, generate_num))
            residual_nodes = range(data_dim)  #追加しなきゃいけないノードのあまり
            residual_nodes = list(residual_nodes)
            current_node = i%data_dim #初期状態(枝を持たない頂点一つの配置)
            node_list = [current_node] # すでに存在するノードを保存
            residual_nodes.remove(current_node)

            # 確率1なのでつなげてしまう
            next_node = np.random.choice(residual_nodes)
            data[current_node, next_node] = 1
            data[next_node, current_node] = 1
            current_node = next_node
            node_list.append(current_node)
            residual_nodes.remove(current_node)

            # 全ノードが現れるまで繰り返す
            while residual_nodes != []:
                next_node = random.choice(residual_nodes)   # 次に接続するノードの選択
                current_node = next_node
                prob_dict = {}#OrderedDict()
                for node in node_list:
                    # すでに存在しているノードと接続確率の導出
                    prob_dict[node] = np.sum(data[node])/np.sum(data)

                # next nodeが接続するか決定
                nodes = list(prob_dict.keys())
                probs = list(prob_dict.values())
                node = np.random.choice(nodes, p=probs)#self.probablistic_choice_node(prob_dict)
                #print(node, current_node)
                data[node, current_node] = 1
                data[current_node, node] = 1
                node_list.append(current_node)
                residual_nodes.remove(current_node)
            datas.append(mat2graph_obj(data))
        return datas

    def generate_BA(self, generate_num, data_dim):
        # print("     generate BA:")
        tmp = []
        for _ in range(generate_num):
            tmp.append(nx.barabasi_albert_graph(data_dim,3,None))

        return tmp

    def nearest_neighbor_model(self, generate_num, data_dim, u):
        print("     generate nearest negibor(u=%.1f):"%(u))
        datas = []

        # 作成するデータ数
        for i in range(generate_num):
            if i % 1000 == 0:
                print("     [%d/%d]"%(i, generate_num))
            data = np.zeros((data_dim, data_dim))    # とりあえず無向で保存
            potential_links = []
            added_nodes = []

            residual_nodes = list(range(data_dim))
            added_nodes.append(np.random.choice(residual_nodes))    # 初期ノード
            residual_nodes.remove(added_nodes[-1])

            while 1:
                # 確率1-uで
                if u < random.random():
                    if len(added_nodes) == data_dim:
                        break
                    add_node = np.random.choice(residual_nodes)
                    connect_node = np.random.choice(added_nodes) # 接続するノード

                    data[add_node, connect_node] = 1
                    data[connect_node, add_node] = 1

                    added_nodes.append(add_node)
                    residual_nodes.remove(add_node)

                    args = np.array(np.where(data[connect_node]==1))
                    args = args[0].tolist()
                    args.remove(add_node)
                    for arg in args:
                        potential_links.append(list(sorted([arg, add_node])))

                    if potential_links != []:
                        potential_links = np.unique(potential_links, axis=0) # 重複の削除
                        potential_links = potential_links.tolist()

                # 確率uで
                else:
                    if potential_links != []:
                        arg = np.random.choice(range(len(potential_links)))
                        args = potential_links[arg]
                        data[args[0], args[1]] = 1
                        data[args[1], args[0]] = 1
                        potential_links.remove(args)
            datas.append(mat2graph_obj(data))
        return datas

    def make_reddit_graph(self):
        """redditのグラフデータを作成する関数

        Returns
        -------
        list
            networkx型のグラフが入ったlist
        """
        with open(reddit_path, 'r') as f:
            json_datas = json.load(f)

        datas = json2graph(json_datas)

        return datas

    def make_twitter_graph(self,do_type=None):
        """twitterのグラフデータを作成する関数

        Returns
        -------
        list:
            networkx型のグラフが入ったlist
        """
        text_datas = utils.get_directory_paths(twitter_path)
        graph_datas = text2graph(text_datas)
        
        split_num = len(graph_datas)

        if do_type == 'train':
            return graph_datas[:int(split_num*0.9)]
        elif do_type == 'valid':
            return graph_datas[int(split_num*0.9):]
        else:
            return graph_datas

    def make_twitter_graph_with_label(self):
        text_datas = utils.get_directory_paths(twitter_path)
        graph_datas = text2graph(text_datas)
        train_data, valid_data = train_test_split(graph_datas, test_size=0.1)

        train_labels = torch.Tensor()
        valid_labels = torch.Tensor()

        st = graph_statistic()
        # conditionalで指定するlabelを取得する

        print('generate train data ...')
        for graph in tqdm(train_data):
            # クラスタ係数と最長距離を指定するためにパラメータを取得してlabelとする
            # paramsはリスト型で渡されるのでindex[0]をつける
            params = st.calc_graph_traits2csv([graph],["cluster_coefficient"])[0]
            tmp_label = []
            for param in params.values():
                # 小数点第1位で四捨五入する
                tmp_label.append(round(param,1))
            tmp_label = torch.tensor(tmp_label).unsqueeze(0)
            train_labels = torch.cat((train_labels, tmp_label),dim=0)
        train_labels.unsqueeze(1)

        print('generate valid data ...')
        for graph in tqdm(valid_data):
            # クラスタ係数と最長距離を指定するためにパラメータを取得してlabelとする
            # paramsはリスト型で渡されるのでindex[0]をつける
            params = st.calc_graph_traits2csv([graph],["cluster_coefficient"])[0]
            tmp_label = []
            for param in params.values():
                # 小数点第1位で四捨五入する
                tmp_label.append(round(param,1))
            tmp_label = torch.tensor(tmp_label).unsqueeze(0)
            valid_labels = torch.cat((valid_labels, tmp_label),dim=0)
        valid_labels.unsqueeze(1)

        joblib.dump([train_data, train_labels], twitter_train_path)
        joblib.dump([valid_data, valid_labels], twitter_valid_path)

    def make_twitter_gcn_dataset(self):
        text_datas = utils.get_directory_paths(twitter_path)
        graph_datas = text2graph(text_datas)
        gcn_graphs = self.gcn_dataset(graph_datas)
        return gcn_graphs

    def gcn_dataset(self, graphs):
        graph_list = []
        node_feats = torch.Tensor()
        for graph in graphs:
            for node in graph.nodes():
                node_feats = torch.cat((node_feats,graph.degree(node)),dim=1)
            graph.ndata['feat'] = node_feats
            graph_list.append(dgl.from_networkx(graph_list))

        return graph_list

    def pickup_twitter_data(self):
        text_datas = utils.get_directory_paths(twitter_path)
        datas = text2graph(text_datas)

        sample_datas = random.sample(datas, graph_num)

        return sample_datas

    def graph2csv(self, graph_datas, file_name):
        '''
        各グラフデータのパラメータをcsvファイルに出力する関数

        Parameters
        ----------
        graph_datas : list
            グラフデータが格納されているリスト [GraphObj, ...]
        file_name : string
            csvファイルの格納先フォルダを指定する変数 例:Twitter/twitter.csv
        '''
        statistic = graph_statistic()
        trait_dict = statistic.calc_graph_traits2csv(graph_datas, eval_params)
        # with open('./data/csv/' + file_name + '.csv', 'w') as f:
        #     writer = csv.DictWriter(f, fieldnames=eval_params)
        #     writer.writeheader()
        #     writer.writerows(trait_dict)
        with open('./results/'+ run_time + '/csv/' + file_name + '.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=eval_params)
            writer.writeheader()
            writer.writerows(trait_dict)

    def graph_data_compress(datas,file_name):
        joblib.dump(datas,'./data/graph_datas/'+file_name+'.pkl.cmp',compress=True)

    def create_label(self,label_index="all"):
        degree_dict = {degree:index for index, degree in enumerate(power_degree_label)}
        cluster_dict = {cluster:index for index, cluster in enumerate(cluster_coefficient_label)}
        power_degree_conditinal_label = utils.convert2onehot(list(degree_dict.values()),power_degree_dim)
        cluster_coefficient_conditinal_label = utils.convert2onehot(list(cluster_dict.values()),cluster_coefficient_dim)
        label = torch.cat((power_degree_conditinal_label,cluster_coefficient_conditinal_label),1)
        
        if type(label_index) is str:
            return label
        else:
            return label[label_index]

class graph_statistic():
    def fitting_function(self, k, a, b):
        return a*k+b

    # 隣接行列を入力として, 次数分布を作成
    def degree_dist(self, graph):
        degree = list(dict(nx.degree(graph)).values())

        import collections
        power_degree = dict(collections.Counter(degree))
        power_degree = sorted(power_degree.items(), key=lambda x:x[0])
        x = []
        y = []
        
        for i in power_degree:
            num = i[0]
            amount = i[1]
            x.append(num)
            y.append(amount)
        y = np.array(y) / sum(y)#次数を確率化
        sum_prob = 0
        for index,prob in enumerate(y):
            sum_prob += prob
            if sum_prob >= power_degree_border_line:
                border_index = index + 1
                break

        x_log = np.log(np.array(x))
        y_log = np.log(np.array(y))

        x_split_plot = x_log[border_index:]
        y_split_plot = y_log[border_index:]
        param =  np.polyfit(x_split_plot,y_split_plot,1)
        return param[0]

    def cluster_coeff(self, graph):
        #graph = np.array(graph)
        #graph = mat2graph_obj(graph)
        return nx.average_clustering(graph)

    def ave_dist(self, graph):
        #graph = np.array(graph)
        #graph = mat2graph_obj(graph)
        return nx.average_shortest_path_length(graph)

    def ave_degree(self, graph):
        degree_count = 0
        degree_dict = graph.degree()
        for node in degree_dict:
            node_num, node_degree = node
            degree_count += node_degree
        return degree_count/graph.number_of_nodes()

    def density(self,graph):
        """グラフの密度を求める

        Args:
            graph (nx.graph)): [計算したいnetworkx型のグラフ]

        Returns:
            [int]: [密度の値]
        """
        try:
            return nx.density(graph)
        except:
            return 0

    # 未完成
    def clique(self,graph):
        return nx.graph_number_of_cliques(graph)

    def modularity(self,graph):
        """グラフのmodularityを求める関数

        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ

        Returns:
            float: modularityの値
        """
        partition = community_louvain.best_partition(graph)
        return community_louvain.modularity(partition,graph)

    def number_of_clusters(self,graph):
        clusters_dict = community_louvain.best_partition(graph)
        clusters_set = set(value for key, value in clusters_dict.items())
        return len(clusters_set)

    # 未完成
    def giant_component(self,graph):
        print(graph.subgraph(max(nx.connected_components(graph), key=len)))
        return graph.subgraph(max(nx.connected_components(graph), key=len))

    def number_of_connected_component(self,graph):
        pass

    def degree_assortativity(self,graph):
        pass

    def reciprocity(self,graph):
        pass

    def maximum_of_shortest_path_lengths(self,graph):
        """最長の長さを持つ最短経路長の長さを求める関数

        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ

        Returns:
            int: 経路の長さ
        """
        max_shortest_path_length = 0
        path_dict = nx.shortest_path(graph)
        for node_num, paths in path_dict.items():
            for connect_node_num, path in paths.items():
                if len(path) >= max_shortest_path_length:
                    max_shortest_path_length = len(path)

        return max_shortest_path_length
        

    def degree_centrality(self,graph):
        '''
        グラフの各ノードの次数中心性を導出する
        Parameters
        ----------
            graph:計算したいグラフ
        
        Returns
        -------
            degree_centrality:グラフの各ノードの次数中心性のdict
        '''
        degree_centers = nx.degree_centrality(graph)
        return degree_centers

    def betweenness_centrality(self,graph):
        '''
        グラフの各ノードの媒介中心性を導出する
        Parameters
        ----------
            graph:計算したいグラフ
        
        Returns
        -------
            betweenness_centrality:グラフの各ノードの媒介中心性のdict
        '''
        betweenness_centers = nx.betweenness_centrality(graph)
        return betweenness_centers

    def closeness_centrality(self,graph):
        '''
        グラフの各ノードの近接中心性を導出する
        Parameters
        ----------
            graph:計算したいグラフ
        
        Returns
        -------
            closeness_centrality:グラフの各ノードの近接中心性のdict
        '''
        closeness_centers = nx.closeness_centrality(graph)
        return closeness_centers

    # 全グラフのパラメータを導出してリスト形式で保存
    def calc_graph_traits(self, graphs, eval_params):
        """
        グラフごとにeval_paramsで指定されている特性値をcalc. {特性名: [値,...,]}
        Args:
            graphs: [graph_obj, ....]
            eval_params: 計算を行う特性値の名前のlist
        """
        trait_dict = {key: [] for key in eval_params}
        for graph in graphs:
            for key in eval_params:
                if "degree" in key:
                    gamma = self.degree_dist(graph)
                    trait_dict[key].append(gamma)
                if "cluster" in key:
                    trait_dict[key].append(self.cluster_coeff(graph))
                if "distance" in key:
                    trait_dict[key].append(self.ave_dist(graph))
                if "size" in key:
                    trait_dict[key].append(graph.number_of_nodes())
        return trait_dict

    def calc_graph_traits2csv(self,graphs,eval_params):
        '''
        グラフごとにeval_paramsで指定されている特性値を計算してcsvへの保存形式に変換
        Prameters
        ---------
            graphs: [graph_obj, ....]
            eval_params: 計算を行う特性値の名前のlist

        Returns
        -------
            trait_list: 各グラフのparamのdictのlist
        '''
        trait_list=[]
        for index, graph in enumerate(graphs):
            tmp_dict = {}
            try:
                for key in eval_params:
                    #if "id" in key:
                    #    param = index
                    if "power_degree" in key:
                        param = self.degree_dist(graph)
                    if "cluster_coefficient" in key:
                        param = self.cluster_coeff(graph)
                    if "distance" in key:
                        param = self.ave_dist(graph)
                    if "average_degree" in key:
                        param = self.ave_degree(graph)
                    if "density" in key:
                        param = self.density(graph)
                    if "modularity" in key:
                        param = self.modularity(graph)
                    if "maximum_distance" in key:
                        param = self.maximum_of_shortest_path_lengths(graph)
                    if "degree_centrality" in key:
                        param = self.degree_centrality(graph)
                    if "betweenness_centrality" in key:
                        param = self.betweenness_centrality(graph)
                    if "closeness_centrality" in key:
                        param = self.closeness_centrality(graph)
                    if "size" in key:
                        param = graph.number_of_nodes()
                    tmp_dict.update({key:param})
                trait_list.append(tmp_dict)
            except:
                continue
        return trait_list

# 隣接行列を隣接リストに変換
def mat_to_list(adj_mat):
    adj_list = []
    for i in range(len(adj_mat)):
        args = list(np.where(np.array(adj_mat[i])>=1))[0]
        adj_list.append(args.tolist())
    return adj_list

# 隣接リストを隣接行列に変換
def list_to_mat(adj_list):
    adj_mat = np.zeros((len(adj_list), len(adj_list)))
    for i, adj_nodes in enumerate(adj_list):
        for adj_node in adj_nodes:
            adj_mat[i, adj_node] = 1
            adj_mat[adj_node, i] = 1
    return adj_mat

# 隣接行列をnetworkxのobjに変換
def mat2graph_obj(adj_mat):
    adj_mat = np.array(adj_mat, dtype=np.int)
    G = nx.Graph()
    args1, args2 = np.where(adj_mat==1)
    args1 = args1.reshape(-1, 1)
    args2 = args2.reshape(-1, 1)
    args = np.concatenate((args1, args2), axis=1)
    G.add_nodes_from(range(len(adj_mat)))
    G.add_edges_from(args)
    return G

# networkxのobjを隣接行列に変換
# 同じラベルのノードは無い前提
def graph_obj2mat(G):
    nodes = G.nodes
    edges = G.edges
    nodes = {i: node_label for i, node_label in enumerate(nodes)}

    adj_mat = np.zeros((len(nodes), len(nodes)))

    # forでぶん回している. smartにしたい
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]

        node1_arg = None
        node2_arg = None
        for key, node_label in nodes.items():
            if node1 == node_label:
                node1_arg = key
            if node2 == node_label:
                node2_arg = key

            # for短縮のため
            if not node1_arg is None and not node2_arg is None:
                break
        adj_mat[node1_arg, node2_arg] = 1
        adj_mat[node2_arg, node1_arg] = 1
    return adj_mat

# 連結グラフかどうかの判定
def is_connect(graph):
    #graph = np.array(graph)
    #graph = mat2graph_obj(graph)
    if nx.is_empty(graph):
        return False
    
    return nx.is_connected(graph)

# グラフの描画
# 隣接行列かnetworkxのノードを引数とする
def draw_graph(adj_mat, pic_dir="./pic.png", node_color=None, label=None):
    if type(adj_mat) is np.ndarray:
        G = mat2graph_obj(adj_mat)
    else:
        G = adj_mat
    plt.figure()
    plt.axis("off")
    nx.draw_networkx(G, node_color=node_color, labels=label)
    plt.savefig(pic_dir)

class ConvertToDfsCode():
    def __init__(self,graph,mode="normal"):
        """
        Args:
            graph:
            mode: ["normal", "low_degree_first", "high_degree_first"]
        """

        self.G = graph
        self.node_tree = [node for node in graph.nodes()]
        self.edge_tree = [edge for edge in graph.edges()]
        self.dfs_code = list()
        self.visited_edges = list()
        self.time_stamp = 0
        self.node_time_stamp = [-1 for i in range(graph.number_of_nodes())]
        self.mode=mode

    def get_max_degree_index(self):
        max_degree = 0
        max_degree_index = 0
        for i in range(self.G.number_of_nodes()):
            if(self.G.degree(i) >= max_degree):
                max_degree = self.G.degree(i)
                max_degree_index = i

        return max_degree_index

    def dfs(self,current_node):
        neightbor_node_dict = OrderedDict({neightbor:self.node_time_stamp[neightbor] for neightbor in self.G.neighbors(current_node)})
        neighbor_degree_dict = OrderedDict({neighbor: self.G.degree[neighbor] for neighbor in neightbor_node_dict.keys()})
        if self.mode=="high_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=True))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        elif self.mode=="low_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=False))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        else:
            sorted_neightbor_node = OrderedDict(sorted(neightbor_node_dict.items(), key=lambda x: x[1], reverse=True))

        if(len(self.visited_edges) == len(self.edge_tree)):
            return

        for next_node in sorted_neightbor_node.keys():
            # visited_edgesにすでに訪れたエッジの組み合わせがあったらスルー
            if((current_node, next_node) in self.visited_edges or (next_node, current_node)in self.visited_edges):
                continue
            else:
                if(self.node_time_stamp[next_node] != -1):
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1

                    self.visited_edges.append((current_node,next_node))
                    self.dfs_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),0])
                else:
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1
                    # 次のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[next_node] == -1):
                        self.node_time_stamp[next_node] = self.time_stamp
                        self.time_stamp += 1
                    # timeStamp_u, timeStamp_v, nodeLabel u, nodeLable_v ,edgeLable(u,v)の順のリストを作成
                    self.dfs_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),0])
                    self.visited_edges.append((current_node,next_node))
                    self.dfs(next_node)

    def get_dfs_code(self):
        self.dfs(self.get_max_degree_index())
        return np.array(self.dfs_code)

class SimpleDfsCode():
    def __init__(self,graph):
        self.G = graph
        self.node_tree = [node for node in graph.nodes()]
        self.edge_tree = [edge for edge in graph.edges()]
        self.dfs_code = list()
        self.visited_edges = list()
        self.node_time_stamp = [None for i in range(graph.number_of_nodes())]

    def get_max_degree_index(self):
        max_degree = 0
        max_degree_index = 0
        for i in range(self.G.number_of_nodes()):
            if(self.G.degree(i) >= max_degree):
                max_degree = self.G.degree(i)
                max_degree_index = i

        return max_degree_index

    def dfs(self,current_node,time_stamp=0):
        neightbor_node_list = self.G.neighbors(current_node)
        if(len(self.visited_edges) == len(self.edge_tree)):
            return

        for i in neightbor_node_list:
            # visited_edgesにすでに訪れたエッジの組み合わせがあったらスルー
            if((current_node, i) in self.visited_edges or (i, current_node)in self.visited_edges):
                continue
            else:
                # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                if(self.node_time_stamp[current_node] == None):
                    self.node_time_stamp[current_node] = time_stamp
                    time_stamp += 1
                # 次のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                if(self.node_time_stamp[i] == None):
                    self.node_time_stamp[i] = time_stamp
                    time_stamp += 1
                # timeStamp_u, timeStamp_v, nodeLabel u, nodeLable_v ,edgeLable(u,v)の順のリストを作成
                self.dfs_code.append([self.node_time_stamp[current_node],self.node_time_stamp[i],self.G.degree(current_node),self.G.degree(i),0])
                self.visited_edges.append((current_node,i))
                self.dfs(i,time_stamp)

    def get_dfs_code(self):
        self.dfs(self.get_max_degree_index())
        return np.array(self.dfs_code)

    def get_sequence_dfs_code(self):
        self.dfs(self.get_max_degree_index())
        dfs_array = np.array(self.dfs_code)
        
        return dfs_array.T

def dfs_code_to_graph_obj(dfs_code,end_value_list):
    """DFScodeをnetworkxのグラフオブジェクトに変換する関数

    Args:
        dfs_code ([np.array]): [(sequence,5)のnp.array]
        end_value_list ([list]): [終了コード[5]]

    Returns:
        [networkx_graph]: [networkxのグラフオブジェクトを返す]
    """    
    G = nx.Graph()
    for current_code in dfs_code:
        for i in range(len(current_code)):
            # 長さ自体はend_value_listの値だが実際の値は0から始まっているため-1する
            if current_code[i] == end_value_list[i]-1:
                print('end code desu')
                return G
        tu,tv,_,_,_ = current_code
        G.add_edge(tu,tv)
    
    return G

def divide_label(label,end_value_list):
    label_data_num = power_degree_dim*1 # ちゃんと3x3通りになったらこっちを使う power_degree_dim*cluster_coefficient_dim
    divide_list = list()
    
    st = graph_statistic()
    
    # correct graphs
    train_label = [code.unsqueeze(2) for code in label]
    dfs_code = torch.cat(train_label, dim=2)
    correct_graph = [
            dfs_code_to_graph_obj(
                code.detach().numpy(),
                end_value_list
                )
            for code in dfs_code]
    
    for i in range(label_data_num):
        divide_list.append(list())
        
    for graph in correct_graph:
        if abs(round(st.degree_dist(graph),1)) == power_degree_label[0]:
            if abs(round(nx.average_clustering(graph),1)) == cluster_coefficient_label[0]:
                divide_list[0].append(graph)
        elif abs(round(st.degree_dist(graph),1)) == power_degree_label[1]:
            if abs(round(nx.average_clustering(graph),1)) == cluster_coefficient_label[1]:
                divide_list[1].append(graph)
        elif abs(round(st.degree_dist(graph),1)) == power_degree_label[2]:
            if abs(round(nx.average_clustering(graph),1)) == cluster_coefficient_label[2]:
                divide_list[2].append(graph)
                
    return divide_list

def json2graph(json_datas):
    graph_data = []
    for json_data in json_datas.values():
        G = nx.Graph()
        G.add_edges_from(json_data)
        graph_data.append(G)
    return graph_data

def text2graph(text_datas):
    graph_data = []
    for text_data in text_datas:
        with open(text_data, 'rb') as f:
            G = nx.read_edgelist(f,nodetype=int)
        graph_data.append(G)
    return graph_data


if __name__ == "__main__":
    complex_network = complex_networks()
    # datasets,labelsets = complex_network.create_conditional_dataset(train_generate_detail)
    # print(datasets)
    # print(labelsets.unsqueeze(1).size())
    dataset = complex_network.make_twitter_graph()
    # for graph
    gs = graph_statistic()
    print(gs.degree_dist(dataset[0]))

