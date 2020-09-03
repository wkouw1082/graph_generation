import glob
import argparse
import os
from config import *
from collections import OrderedDict
import collections
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--load", action="store_true")

    args = parser.parse_args()
    return args

#--- pytorch用---
def cpu(x):
    return x.cpu().detach().numpy()

def try_gpu(obj):
    import torch
    if torch.cuda.is_available():
        return obj.cuda(device=0)
    return obj

# ---汎用的---
def make_dir(required_dirs):
    dirs = glob.glob("*")
    for required_dir in required_dirs:
        if not required_dir in dirs:
            print("generate file in current dir...")
            print("+ "+required_dir)
            os.mkdir(required_dir)
        print("\n")

def is_dir_existed(directory):
    dirs = glob.glob("*")
    if directory in dirs:
        return True
    else:
        return False

def methods(obj):
    for method in dir(obj):
        print(method)

def flatten(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, collections.Iterable) and not isinstance(element, str):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result

def recreate_dir(directory):
    for dir in directory:
        for dir in directory:
            shutil.rmtree(dir)
        make_dir(directory)

# ---研究用---
# vecは入れ子になっている前提
def tsne(multi_vecs, dir):
    datas = []
    color = []
    dim = 0
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

    for i, vecs in enumerate(multi_vecs.values()):
        for vec in vecs:
            dim = np.array(vec).shape[-1]
            datas.append(vec)
            color.append(colorlist[i])
    datas = np.array(datas).reshape((-1, dim))

    result = TSNE(n_components=2).fit_transform(datas)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for key, c in zip(multi_vecs.keys(), set(color)):
        same_c_datas = np.array(result)[np.array(color)==c]
        ax.scatter(same_c_datas[:,0], same_c_datas[:,1], c=c, label=key)
    ax.legend(loc='upper right')
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    plt.savefig(dir)


