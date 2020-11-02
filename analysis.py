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

required_dirs = ["dataset", "analysis_result"]
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

# conditionalをcat
train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1)
train_dataset = torch.cat((train_dataset,train_conditional),dim=2)

# gpu
train_dataset=utils.try_gpu(train_dataset)
train_conditional=utils.try_gpu(train_conditional)

time_size, node_size, edge_size,_ = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size
sizes = [time_size, time_size, node_size, node_size, edge_size]
keys = ["tu", "tv", "lu", "lv", "le"]

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("--------------")

# 2d colormesh
for i, args in enumerate(same_conditional_args):
    total_size = 0
    for key, size in zip(keys, sizes):
        data = train_dataset[args, :, total_size:total_size+size]
        data = torch.sum(data, dim=0)
        total_size+=size

        plt.figure()
        plt.imshow(data.cpu())
        plt.colorbar()
        plt.xlabel("dim")
        plt.ylabel("seq")
        plt.savefig("analysis_result/c%d_%s.png"%(i, key))
        plt.close()

# resultの可視化
time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
vae = model.VAE(dfs_size+conditional_size, time_size, node_size, edge_size, model_param)
vae.load_state_dict(torch.load("param/weight", map_location="cpu"))

vae = utils.try_gpu(vae)

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
    plt.savefig("analysis_result/pred_acc_%s.png"%(keys[j]))
    plt.close()


#cx = complex_networks()
#conditional_labels = cx.create_label()

for i, conditional_label in enumerate(conditional_labels):
    conditional_label = utils.try_gpu(conditional_label)
    result = vae.generate(500, conditional_label)
    for j, pred in enumerate(result):
        pred = pred.cpu().detach().numpy()
        pred = [utils.convert2onehot(data, sizes[j]).detach().numpy() for data in pred]
        pred = np.array(pred)
        data = np.sum(pred, 0)

        plt.figure()
        plt.imshow(data)
        plt.colorbar()
        plt.xlabel("dim")
        plt.ylabel("seq")
        plt.savefig("analysis_result/generated_c%d_%s.png"%(i, keys[j]))
        plt.close()
