import numpy as np
import networkx as nx
import networkx.algorithms.approximation.treewidth as nx_tree
import random
import matplotlib.pyplot as plt
import joblib
import hoge
from scipy.optimize import curve_fit
import config
import sys
import time

# 複雑ネットワークを返すクラス
# datasetはnetworkxのobjのlist
class complex_networks():
    def create_dataset(self, detail, directory):
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
                joblib.dump(datas, directory+"%d"%(i), compress=3)
                datasets.extend(datas)
        return datasets

    # 俗に言う修正BAモデルの生成
    def generate_fixed_BA(self, generate_num, data_dim):
        print("     generate fixed BA:")
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
        print("     generate BA:")
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

class graph_statistic():
    def fitting_function(self, k, a, b):
        return a*k+b

    # 隣接行列を入力として, 次数分布を作成
    def degree_dist(self, graph):
        degree_list = np.sum(graph, axis=0)
        degree_dist_dict = {}

        # もしひとつだけ孤立しているようなノードが存在するのならば
        if 0 in degree_list:
            return None

        for degree in degree_list:
            if degree in degree_dist_dict:
                degree_dist_dict[degree] += 1
            else:
                if degree != 0:
                    degree_dist_dict[degree] = 1

        x = np.log(np.array(list(degree_dist_dict.keys())))
        y = np.log(np.array(list(degree_dist_dict.values())))
        param, cov = curve_fit(self.fitting_function, x, y)
        return param[0]

    def cluster_coeff(self, graph):
        graph = np.array(graph)
        graph = mat2graph_obj(graph)
        return nx.average_clustering(graph)

    def ave_dist(self, graph):
        graph = np.array(graph)
        graph = mat2graph_obj(graph)
        return nx.average_shortest_path_length(graph)

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
    graph = np.array(graph)
    graph = mat2graph_obj(graph)
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
