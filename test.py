import sys
import networkx as nx
from collections import deque

# 再起の上限回数を10^9
sys.setrecursionlimit(10**9)

G = nx.barabasi_albert_graph(20,3,1)
node_tree = [node for node in G.nodes()]
node_time_stamp = [None for i in range(G.number_of_nodes())]
edge_tree = [edge for edge in G.edges()]
print(len(edge_tree))
visited_edges = list()
dfs_code = list()
max_node = 0
first_node = 0
for i in range(G.number_of_nodes()):
    if(G.degree(i) >= max_node):
        max_node = G.degree(i)
        first_node = i

def dfs(current_node=first_node,time_stamp=0):
    neightbor_node_list = G.neighbors(current_node)
    if(len(visited_edges) == len(edge_tree)):
        print(len(visited_edges))
        print(len(dfs_code))
        print(dfs_code)
        return

    for i in neightbor_node_list:
        if((current_node, i) in visited_edges or (i, current_node)in visited_edges):
            continue
        else:
            # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
            if(node_time_stamp[current_node] == None):
                node_time_stamp[current_node] = time_stamp
                time_stamp += 1
            # 次のノードにタイムスタンプが登録されていなければタイムスタンプを登録
            if(node_time_stamp[i] == None):
                node_time_stamp[i] = time_stamp
                time_stamp += 1
            # timeStamp_u, timeStamp_v, nodeLabel u, nodeLable_v ,edgeLable(u,v)の順のタプルを作成
            dfs_code.append((node_time_stamp[current_node],node_time_stamp[i],G.degree(current_node),G.degree(i),0))
            visited_edges.append((current_node,i))
            dfs(i,time_stamp)

dfs()