import shutil
import joblib
import torch
import matplotlib.pyplot as plt

import utils
import model
from config import *
import preprocess as pp

args = utils.get_args()
is_preprocess = args.preprocess

required_dirs = ["dataset", "analysis_result"]
utils.make_dir(required_dirs)
print("start preprocess...")

# preprocess
if is_preprocess:
    shutil.rmtree("dataset")
    required_dirs = ["dataset", "dataset/train", "dataset/test"]
    utils.make_dir(required_dirs)
    pp.preprocess(train_generate_detail, test_generate_detail)

# data load
train_dataset = joblib.load("dataset/train/onehot")[0]
train_label = joblib.load("dataset/train/label")
test_dataset = joblib.load("dataset/test/onehot")[0]
test_label = joblib.load("dataset/test/label")

time_size, node_size, edge_size = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size
sizes = [time_size, time_size, node_size, node_size, edge_size]
keys = ["tu", "tv", "lu", "lv", "le"]

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("--------------")

# 2d colormesh
total_size = 0
for key, size in zip(keys, sizes):
    data = train_dataset[:, :, total_size:total_size+size]
    data = torch.sum(data, dim=0)
    total_size+=size

    plt.figure()
    plt.imshow(data)
    plt.colorbar()
    plt.xlabel("dim")
    plt.ylabel("seq")
    plt.savefig("analysis_result/%s.png"%(key))
    plt.close()

# resultの可視化
time_size, node_size, edge_size = joblib.load("dataset/param")
vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param)
vae = utils.try_gpu(vae)

vae.load_state_dict(torch.load("./param/weight"))
pred = vae(train_dataset)

mu, sigma, *result = vae(train_dataset)
for j, pred in enumerate(result):
    correct = train_label[j]
    correct = correct.transpose(1,0)
    pred = torch.argmax(pred, dim=2)  # predicted onehot->label
    pred = pred.transpose(1, 0)

    acc_transition = []
    for k in range(pred.shape[0]):
        tmp = utils.calc_calssification_acc(pred[k], correct[k], ignore_label=1000)
        acc_transition.append(tmp)

    plt.figure()
    plt.plot(acc_transition)
    plt.xlabel("seq")
    plt.ylabel("acc")
    plt.savefig("analysis_result/pred_acc_%s.png"%(keys[j]))
    plt.close()


