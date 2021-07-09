import shutil
import joblib
import torch
import matplotlib.pyplot as plt
import numpy as np

import utils
import model
from config import *
import preprocess as pp
from graph_process import complex_networks

args = utils.get_args()
is_preprocess = args.preprocess

device = utils.get_gpu_info()

# recreate directory
if utils.is_dir_existed("analysis_result"):
    print("delete file...")
    print("- analysis_result")
    shutil.rmtree("./analysis_result")

required_dirs = [
        "dataset",
        "analysis_result",
        "analysis_result/correct_dist",
        "analysis_result/predict_accuracy",
        "analysis_result/generated_dist",
        "analysis_result/generated_dist_sampling",
        "analysis_result/dist_compare"
        ]
utils.make_dir(required_dirs)
print("start preprocess...")

# preprocess
if is_preprocess:
    shutil.rmtree("dataset")
    required_dirs = ["dataset", "dataset/train", "dataset/valid"]
    utils.make_dir(required_dirs)
    pp.preprocess(train_generate_detail, valid_generate_detail)

# data load
train_dataset = joblib.load("dataset/train/onehot")
train_label = joblib.load("dataset/train/label")
valid_dataset = joblib.load("dataset/valid/onehot")
valid_label = joblib.load("dataset/valid/label")
train_conditional = joblib.load("dataset/train/conditional")

# conditionalのlabelと同じラベルの引数のget
tmp=train_conditional.squeeze()
uniqued, inverses=torch.unique(tmp, return_inverse=True, dim=0)
conditional_labels=uniqued
same_conditional_args=[[j for j in range(len(inverses)) if inverses[j]==i] for i in range(uniqued.shape[0])]
get_key=lambda vec: str(power_degree_label[torch.argmax(vec[:3])])+"_"+\
            str(cluster_coefficient_label[torch.argmax(vec[3:])]) # conditional vec->key
conditional_keys=[get_key(conditional_label) for conditional_label in conditional_labels]

# conditionalをcat
train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1)
train_dataset = torch.cat((train_dataset,train_conditional),dim=2)

# gpu
train_dataset=utils.try_gpu(device,train_dataset)
train_conditional=utils.try_gpu(device,train_conditional)

time_size, node_size, edge_size,_ = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size
sizes = [time_size, time_size, node_size, node_size, edge_size]
keys = ["tu", "tv", "lu", "lv", "le"]
data_dict={key1:{key2:{key3: 0 for key3 in keys}for key2 in conditional_keys}for key1 in ["correct", "generate", "generate_sampling"]}
flow_len=train_dataset.shape[1]

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("--------------")

# correct data
# 2d colormesh
for i, args in enumerate(same_conditional_args):
    total_size = 0
    conditional_label=conditional_labels[i]
    conditional_label=get_key(conditional_label)
    tmp_dict={}

    for key, size in zip(keys, sizes):
        data = train_dataset[args, :, total_size:total_size+size]
        data = torch.sum(data, dim=0).cpu()
        total_size+=size
        data_dict["correct"][conditional_label][key]=data

        plt.figure()
        plt.imshow(data)
        plt.colorbar()
        plt.xlabel("dim")
        plt.ylabel("seq")
        plt.savefig("analysis_result/correct_dist/%s_%s.png"%(conditional_label, key))
        plt.close()

# model_param load
    import yaml
    with open('results/best_tune.yml', 'r') as yml:
        model_param = yaml.load(yml) 
    # print(f"model_param = {model_param}")

# resultの可視化
time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
vae = model.VAE(dfs_size+conditional_size, time_size, node_size, edge_size, model_param)
vae.load_state_dict(torch.load("param/weight", map_location="cpu"))

vae = utils.try_gpu(device,vae)

pred = vae(train_dataset)

mu, sigma, *result = vae(train_dataset)
for j, pred in enumerate(result):
    correct = train_label[j]
    correct = correct.transpose(1,0)
    pred = torch.argmax(pred, dim=2)  # predicted onehot->label
    pred = pred.transpose(1, 0)

    acc_transition = []
    for k in range(pred.shape[0]):
        tmp = utils.calc_calssification_acc(pred[k].cpu(), correct[k].cpu(), ignore_label=1000)
        acc_transition.append(tmp)

    plt.figure()
    plt.plot(acc_transition)
    plt.xlabel("seq")
    plt.ylabel("acc")
    plt.savefig("analysis_result/predict_accuracy/pred_acc_%s.png"%(keys[j]))
    plt.close()


# generate&sampling(onehot)
for i, conditional_label in enumerate(conditional_labels):
    conditional_label = utils.try_gpu(device,conditional_label)
    result = vae.generate(500, conditional_label)
    conditional_label=get_key(conditional_label)
    for j, pred in enumerate(result):
        pred = pred.cpu().detach().numpy()
        pred = [utils.convert2onehot(data, sizes[j]).detach().numpy() for data in pred]
        pred = np.array(pred)
        data = np.sum(pred, 0)
        data = data[:flow_len, :]
        data_dict["generate_sampling"][conditional_label][keys[j]]=data

        plt.figure()
        plt.imshow(data)
        plt.colorbar()
        plt.xlabel("dim")
        plt.ylabel("seq")
        plt.savefig("analysis_result/generated_dist_sampling/generated_%s_%s.png"%(conditional_label, keys[j]))
        plt.close()

# generate distribution
for i, conditional_label in enumerate(conditional_labels):
    conditional_label = utils.try_gpu(device,conditional_label)
    result = vae.generate(500, conditional_label, is_output_sampling=False)
    conditional_label=get_key(conditional_label)

    for j, pred in enumerate(result):
        pred = torch.sum(pred, dim=0)
        data = pred.cpu().detach().numpy()
        data = data[:flow_len, :]
        data_dict["generate"][conditional_label][keys[j]]=data

        plt.figure()
        plt.imshow(data)
        plt.colorbar()
        plt.xlabel("dim")
        plt.ylabel("seq")
        plt.savefig("analysis_result/generated_dist/generated_%s_%s.png"%(conditional_label, keys[j]))
        plt.close()

# codeごと特性値ごとに
for codekey in keys:
    for conditional_key in conditional_keys:
        fig=plt.figure()
        for i, key in enumerate(data_dict.keys()):
            ax=fig.add_subplot(1, len(data_dict.keys()), i+1)
            im=ax.imshow(data_dict[key][conditional_key][codekey])
            ax.set(title=key, xlabel='seq', ylabel='dim')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig("analysis_result/dist_compare/%s_%s.png"%(conditional_key, codekey))

def plot_dist(dictionary, directory):
    for conditional_key in conditional_keys:
        fig=plt.figure()
        for i, key in enumerate(keys):
            ax=fig.add_subplot(1, len(keys), i+1)
            im=ax.imshow(dictionary[conditional_key][key])
            ax.set(title=key, xlabel='seq', ylabel='dim')
        fig.subplots_adjust(wspace=2, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(directory+"%s.png"%(conditional_key))
        plt.close()

    for key in keys:
        fig=plt.figure()
        for i, conditional_key in enumerate(conditional_keys):
            ax=fig.add_subplot(1, len(conditional_keys), i+1)
            im=ax.imshow(dictionary[conditional_key][key])
            ax.set(title=conditional_key, xlabel='seq', ylabel='dim')
        fig.subplots_adjust(wspace=1.5, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(directory+"%s.png"%(key))
        plt.close()
plot_dist(data_dict["correct"], "analysis_result/correct_dist/")
plot_dist(data_dict["generate"], "analysis_result/generated_dist/")
plot_dist(data_dict["generate_sampling"], "analysis_result/generated_dist_sampling/")
