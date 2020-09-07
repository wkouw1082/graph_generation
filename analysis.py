import shutil
import joblib
import torch
import matplotlib.pyplot as plt

import utils

args = utils.get_args()
is_preprocess = args.preprocess

# recreate directory
if utils.is_dir_existed("train_result"):
    print("delete file...")
    print("- train_result")
    shutil.rmtree("./train_result")

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

