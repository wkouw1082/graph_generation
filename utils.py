import glob
import argparse
import os
from config import *
from collections import OrderedDict
import collections
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--classifier", action="store_true")

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

def convert2onehot(vec, dim):
    """
    特徴量のnumpy配列をonehotベクトルに変換
    :param vec: 特徴量のnumpy行列, int型 (サンプル数分の1次元行列)．
    :param dim: onehot vectorの次元
    :return: onehot vectorのtensor行列
    """
    import torch
    return torch.Tensor(np.identity(dim)[vec])

def padding(vecs, flow_len, value=0):
    """
    flowの長さを最大flow長に合わせるためにzeropadding
    :param vecs: flow数分のリスト. リストの各要素はflow長*特徴量長の二次元numpy配列
    :param flow_len: flow長. int
    :param value: paddingするvectorの要素値 int
    :return: データ数*最大flow長*特徴量長の3次元配列
    """
    for i in range(len(vecs)):
        flow = vecs[i]
        if len(flow.shape)==2:
            diff_vec = np.ones((flow_len-flow.shape[0], flow.shape[1]))
        else:
            diff_vec = np.ones((flow_len-flow.shape[0]))
        diff_vec *= value
        vecs[i] = np.concatenate((flow, diff_vec), 0)
    return np.array(vecs)

def calc_calssification_acc(pred_label, correct_label, ignore_label=None):
    """
    分類精度をcalcする関数
    Args:
        pred_label: n次元の予測ラベル. torch.Tensor
        correct_label: n次元の教師ラベル torch.Tensor
        ignore_label: int. accを計算する上で無視するlabelが存在すれば設定
    Returns:
        score: accuracy
    """
    score = torch.zeros(pred_label.shape[0])
    score[pred_label==correct_label] = 1
    data_len = pred_label.shape[0]
    if not ignore_label is None:
        correct_label = correct_label.cpu()
        ignore_args = np.where(correct_label==ignore_label)[0]
        data_len-=len(ignore_args)
        score[ignore_args] = 0
    score = torch.sum(score)/data_len
    return score

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

def time_draw(x, ys, directory, xlabel="", ylabel=""):
    """
    複数の時系列をまとめて可視化
    :param x: x軸のデータ
    :param ys: y軸のデータら. dictionaryでkeyを時系列のlabel, valueをデータとする
    :param directory: 出力するdirectory
    :param xlabel: x軸のラベル
    :param ylabel: y軸のラベル
    """
    plt.figure()
    for label, y in ys.items():
        plt.plot(y, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(directory)
    plt.close()

def combination(list_, num):
    import itertools
    return list(itertools.combinations(list_, num))

def get_directory_datas(path):
    return glob.glob(path)

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

def box_plot(predict, correct, trait_name, directory):
    fig = plt.figure()
    sns.set_context("paper", 1.2)
    ax = fig.add_subplot(1, 1, 1)
    correct = pd.DataFrame(correct)
    correct_melt = pd.melt(correct)
    correct_melt["species"] = "train"
    predict = pd.DataFrame(predict)
    predict_melt = pd.melt(predict)
    predict_melt["species"] = "generated"
    df = pd.concat([correct_melt, predict_melt], axis=0)

    sns.boxplot(x='variable', y='value', data=df, hue='species', showfliers=False, palette='Set3', ax=ax)
    sns.stripplot(x='variable', y='value', data=df, hue='species', dodge=True, jitter=True, color='black', ax=ax)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles[0:2], labels[0:2])
    ax.set_xlabel('network')
    ax.set_ylabel('%s'%(trait_name))
    plt.savefig(directory)

def graph_visualize(graph):
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(graph)
    nx.draw_networkx(G,pos)

    plt.axis("off")
    # plt.savefig("default.png")
    plt.show()