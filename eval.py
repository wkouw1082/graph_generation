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

required_dirs = ["param", "eval_result", "dataset"]
utils.make_dir(required_dirs)

train_label = joblib.load("dataset/train/label")
time_size, node_size, edge_size = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("--------------")

vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param)
vae.load_state_dict(torch.load("param/weight", map_location="cpu"))
vae = utils.try_gpu(vae)

keys = ["tu", "tv", "lu", "lv", "le"]

result = vae.generate(500)

# generated graphs
result = [code.unsqueeze(2) for code in result]
dfs_code = torch.cat(result, dim=2)

generated_graph = []
for code in dfs_code:
    graph = gp.dfs_code_to_graph_obj(
            code.detach().numpy(),
            [time_size, time_size, node_size, node_size, edge_size])
    if gp.is_connect(graph):
        generated_graph.append(graph)

# correct graphs
train_label = [code.unsqueeze(2) for code in train_label]
dfs_code = torch.cat(train_label, dim=2)
correct_graph = [
        gp.dfs_code_to_graph_obj(
            code.detach().numpy(),
            [time_size, time_size, node_size, node_size, edge_size]
            )
        for code in dfs_code]

gs = gp.graph_statistic()
results = {"generated": {key: [] for key in eval_params},
           "correct": {key: [] for key in eval_params}}
for generated, correct in zip(generated_graph, correct_graph):
    for key in eval_params:
        if "degree" in key:
            gamma = gs.degree_dist(generated)
            results["generated"][key].append(gamma)
            gamma = gs.degree_dist(correct)
            results["correct"][key].append(gamma)
        if "cluster" in key:
            results["generated"][key].append(gs.cluster_coeff(generated))
            results["correct"][key].append(gs.cluster_coeff(correct))
        if "distance" in key:
            results["generated"][key].append(gs.ave_dist(generated))
            results["correct"][key].append(gs.ave_dist(correct))
        if "size" in key:
            results["generated"][key].append(generated.number_of_nodes())
            results["correct"][key].append(correct.number_of_nodes())


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
            {keys[0]: results[keys[0]][param_key]},
            {keys[1]: results[keys[1]][param_key]},
            param_key,
            "eval_result/%s_box_plot.png"%(param_key)
            )

# 散布図
combinations = utils.combination(eval_params, 2)
for key1, key2 in combinations:
    plt.figure()
    plt.scatter(results["generated"][key1], results["generated"][key2])
    plt.xlabel(key1)
    plt.ylabel(key2)
    plt.savefig("eval_result/%s_%s.png"%(key1, key2))
    plt.close()

# t-SNE
train_dataset = joblib.load("dataset/train/onehot")[0]
z = vae.encode(train_dataset).detach().numpy()
utils.tsne({"train": z}, "eval_result/tsne.png")
