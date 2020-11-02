from graph_process import complex_networks
import graph_process
import utils
import preprocess as pp
from config import *
import model
import graph_process as gp
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
import joblib
import shutil

import torch

args = utils.get_args()
is_preprocess = args.preprocess

# recreate directory
if utils.is_dir_existed("eval_result"):
    print("delete file...")
    print("- eval_result")
    shutil.rmtree("./eval_result")

required_dirs = [
        "param",
        "eval_result",
        "eval_result/dist_compare",
        "eval_result/generated",
        "eval_result/reconstruct",
        "dataset"]
utils.make_dir(required_dirs)

train_label = joblib.load("dataset/train/label")
time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size+conditional_size

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("--------------")

vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param)
vae.load_state_dict(torch.load("param/weight", map_location="cpu"))
vae = utils.try_gpu(vae)

keys = ["tu", "tv", "lu", "lv", "le"]

cx = complex_networks()
conditional_label = cx.create_label()

result_low = vae.generate(500,conditional_label[0])
result_middle = vae.generate(500,conditional_label[1])
result_high = vae.generate(500,conditional_label[2])

result_all = [result_low,result_middle,result_high]

end_value_list = [time_size, time_size, node_size, node_size, edge_size]
correct_all = graph_process.divide_label(train_label,end_value_list)

results = {}

for index,(result,correct_graph) in enumerate(zip(result_all,correct_all)):
# generated graphs
    result = [code.unsqueeze(2) for code in result]
    dfs_code = torch.cat(result, dim=2)
    generated_graph = []
    for code in dfs_code:
        graph = gp.dfs_code_to_graph_obj(
                code.cpu().detach().numpy(),
                [time_size, time_size, node_size, node_size, edge_size])
        if gp.is_connect(graph):
            generated_graph.append(graph)

    gs = gp.graph_statistic()
    dict_tmp = {"correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index]): {key: [] for key in eval_params}}
    results.update(dict_tmp)
    dict_tmp = {str(power_degree_label[index])+" "+str(cluster_coefficient_label[index]): {key: [] for key in eval_params}}
    results.update(dict_tmp)
    for generated, correct in zip(generated_graph, correct_graph):
        for key in eval_params:
            if "degree" in key:
                gamma = gs.degree_dist(generated)
                results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gamma)
                gamma = gs.degree_dist(correct)
                results["correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gamma)
            if "cluster" in key:
                results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gs.cluster_coeff(generated))
                results["correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gs.cluster_coeff(correct))
            if "distance" in key:
                results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gs.ave_dist(generated))
                results["correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gs.ave_dist(correct))
            if "size" in key:
                results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(generated.number_of_nodes())
                results["correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(correct.number_of_nodes())
                
# display result
for key, value in results.items():
    print("====================================")
    print("%s:"%(key))
    print("====================================")
    for trait_key in value.keys():
        print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])))
        print(" %s var: %lf"%(trait_key, np.var(value[trait_key])))
        print("------------------------------------")
    print("\n")

# boxplot
for param_key in eval_params:
    dict = {}
    keys = list(results.keys())
    utils.box_plot(
            {keys[1]: results[keys[1]][param_key],
             keys[3]: results[keys[3]][param_key],
             keys[5]: results[keys[5]][param_key]},
            {keys[0]: results[keys[0]][param_key],
             keys[2]: results[keys[2]][param_key],
             keys[4]: results[keys[4]][param_key]},
            param_key,
            "eval_result/generated/%s_box_plot.png"%(param_key)
            )

# 散布図
combinations = utils.combination(eval_params, 2)
for index in range(power_degree_dim):
    for key1, key2 in combinations:
        plt.figure()
        plt.scatter(results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key1], results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key2])
        plt.xlabel(key1)
        plt.ylabel(key2)
        plt.savefig("eval_result/dist_compare/%s_%s.png"%(key1, key2))
        plt.close()

# t-SNE
train_dataset = joblib.load("dataset/train/onehot")
train_conditional = joblib.load("dataset/train/conditional")
train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1)
train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
train_dataset = utils.try_gpu(train_dataset)

# conditionalのlabelと同じラベルの引数のget
tmp=train_conditional.squeeze()
uniqued, inverses=torch.unique(tmp, return_inverse=True, dim=0)
conditional_vecs=uniqued
same_conditional_args=[[j for j in range(len(inverses)) if inverses[j]==i] for i in range(uniqued.shape[0])]

result={}
for i, args in enumerate(same_conditional_args):
    z = vae.encode(train_dataset[args]).cpu().detach().numpy()
    result["conditional: %d"%(i)]=z
utils.tsne(result, "eval_result/tsne.png")

# 入力に応じたencode+generate
reconstruct_graphs={}
for i, args in enumerate(same_conditional_args):
    # trait keyの作成
    tmp_dict = {key: [] for key in eval_params}
    conditional_vec=conditional_vecs[i][0]
    degree_value=power_degree_label[torch.argmax(conditional_vec[:3])]
    cluster_value=cluster_coefficient_label[torch.argmax(conditional_vec[3:])]
    traitkey=str(degree_value)+" "+str(cluster_value)

    # predict
    mu, sigma, *result = vae(train_dataset[args])

    # argmax
    tmps=[]
    for tmp in result:
        tmps.append(torch.argmax(tmp, dim=2).unsqueeze(2))

    # graphに変換
    dfs_code = torch.cat(tmps, dim=2)
    generated_graph=[]
    for code in dfs_code:
        graph = gp.dfs_code_to_graph_obj(
                code.cpu().detach().numpy(),
                [time_size, time_size, node_size, node_size, edge_size])
        if gp.is_connect(graph):
            generated_graph.append(graph)

    # 特性値をcalc
    for graph in generated_graph:
        for key in eval_params:
            if "degree" in key:
                gamma = gs.degree_dist(graph)
                tmp_dict[key].append(gamma)
            if "cluster" in key:
                tmp_dict[key].append(gs.cluster_coeff(graph))
            if "distance" in key:
                tmp_dict[key].append(gs.ave_dist(graph))
            if "size" in key:
                tmp_dict[key].append(graph.number_of_nodes())
    reconstruct_graphs[traitkey]=tmp_dict

# display result
for key, value in reconstruct_graphs.items():
    print("====================================")
    print("%s:"%(key))
    print("====================================")
    for trait_key in value.keys():
        print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])))
        print(" %s var: %lf"%(trait_key, np.var(value[trait_key])))
        print("------------------------------------")
    print("\n")

# boxplot
for param_key in eval_params:
    dict = {}
    keys = list(results.keys())
    keys = list(sorted([keys[0], keys[2], keys[4]]))
    reconstructkeys=list(sorted(list(reconstruct_graphs.keys())))
    utils.box_plot(
            {reconstructkeys[0]: reconstruct_graphs[reconstructkeys[0]][param_key][:500],
             reconstructkeys[1]: reconstruct_graphs[reconstructkeys[1]][param_key][:500],
             reconstructkeys[2]: reconstruct_graphs[reconstructkeys[2]][param_key][:500]},
            {keys[0]: results[keys[0]][param_key][:500],
             keys[1]: results[keys[1]][param_key][:500],
             keys[2]: results[keys[2]][param_key][:500]},
            param_key,
            "eval_result/reconstruct/%s_box_plot.png"%(param_key)
            )
