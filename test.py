import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
import networkx as nx
from collections import deque

# 再起の上限回数を10^9
sys.setrecursionlimit(10**9)

G = nx.barabasi_albert_graph(20,3,1)
node_tree = [node for node in G.nodes()]
node_time_stamp = [-1 for i in range(G.number_of_nodes())]
print(node_time_stamp)
edge_tree = [edge for edge in G.edges()]
print(len(edge_tree))
# print(len(edge_tree))
visited_edges = list()
dfs_code = list()
max_node = 0
first_node = 0
for i in range(G.number_of_nodes()):
    if(G.degree(i) >= max_node):
        max_node = G.degree(i)
        first_node = i

def dfs(current_node=first_node,time_stamp=0,backward=False):
    print("current node = "+ str(current_node))
    # もしbackward edgeなら探索せずに帰る
    # if(backward == True):
    #     return

    neightbor_node_dict = OrderedDict({neightbor:node_time_stamp[neightbor] for neightbor in G.neighbors(current_node)})
    print(neightbor_node_dict)
    sorted_neightbor_node = OrderedDict(sorted(neightbor_node_dict.items(), key=lambda x: x[1], reverse=True))
    print(list(sorted_neightbor_node.keys()))

    for next_node in sorted_neightbor_node.keys():
        # print(sorted_neightbor_node.keys())
        # print(next_node)
        if((current_node, next_node) in visited_edges or (next_node, current_node)in visited_edges):
            continue
        else:
            # 次のノードにタイムスタンプが登録されている場合、backward
            if (node_time_stamp[next_node] != -1):
                # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                if(node_time_stamp[current_node] == -1):
                    node_time_stamp[current_node] = time_stamp
                    time_stamp += 1

                visited_edges.append((current_node,next_node))
                dfs_code.append((node_time_stamp[current_node],node_time_stamp[next_node],G.degree(current_node),G.degree(next_node),0))
                print(visited_edges)
                print("backward")
                # dfs(next_node,time_stamp,backward=True)

            # 次のノードにタイムスタンプが登録されていない場合、forward
            else:
                # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                if(node_time_stamp[current_node] == -1):
                    node_time_stamp[current_node] = time_stamp
                    time_stamp += 1
                # 次のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                if(node_time_stamp[next_node] == -1):
                    node_time_stamp[next_node] = time_stamp
                    time_stamp += 1
                # timeStamp_u, timeStamp_v, nodeLabel u, nodeLable_v ,edgeLable(u,v)の順のタプルを作成
                dfs_code.append((node_time_stamp[current_node],node_time_stamp[next_node],G.degree(current_node),G.degree(next_node),0))
                visited_edges.append((current_node,next_node))
                print(visited_edges)
                print("forward")
                dfs(next_node,time_stamp,backward=False)
    
    if(len(visited_edges) == len(edge_tree)):
        print(len(visited_edges))
        print(len(dfs_code))
        print(dfs_code)
        return

dfs()
nx.draw_networkx(G)
plt.show()