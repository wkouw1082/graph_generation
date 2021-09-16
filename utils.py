import glob
import argparse
import os
import shutil
from config import *
from collections import OrderedDict
import collections
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import subprocess
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--classifier", action="store_true")

    args = parser.parse_args()
    return args

#--- pytorch用---
def cpu(x):
    return x.cpu().detach().numpy()

def try_gpu(device,obj):
    import torch
    return obj.to(device)
    # if torch.cuda.is_available():
    #     return obj.cuda(device)
    # return obj

def convert2onehot(vec, dim):
    """
    特徴量のnumpy配列をonehotベクトルに変換
    :param vec: 特徴量のnumpy行列, int型 (サンプル数分の1次元行列)．
    :param dim: onehot vectorの次元
    :return: onehot vectorのtensor行列
    """
    import torch
    return torch.Tensor(np.identity(dim)[vec])

def onehot2scalar(onehot_vec):
    onehot_vec = onehot_vec.to('cpu').detach().numpy().copy()
    return np.argmax(onehot_vec, axis=1)

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
        elif len(flow.shape) == 3:
            diff_vec = np.ones((flow_len-flow.shape[0], flow.shape[1], flow.shape[2]))
        else:
            diff_vec = np.ones(flow_len-flow.shape[0])
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

def classification_metric(preds, labels):
    total = 0
    correct = 0
    for pred, label in zip(preds, labels):
        pred = torch.gt(pred, 0)
        label = torch.gt(label, 0)
        if torch.equal(pred, label):
            correct += 1
        total += 1

    return correct/total

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

def get_directory_paths(path):
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

def load_model_param(file_path=None):
    """ハイパーパラメータを読み込む関数

    file_pathを指定しなければ, results/best_tune.yml, config.pyのmodel_params の順に探して読み込む。

    Args:
        file_path (str, optional): 指定したいハイパーパラメータが保存されているymlファイルのパス. Defaults to None.
        （例）file_path = "results/20210101_0000/best_tune.yml"

    Returns:
        model_param (dict): ハイパーパラメータの名前がkey, 値がvalue
    """
    if file_path is not None:
        if os.path.exists(file_path):
            with open(file_path, 'r') as yml:
                model_param = yaml.load(yml) 
            print(f"load model_param from '{file_path}'")
            return model_param
        else:
            print(f"{file_path} が存在しません。")
    
    if os.path.isfile('results/best_tune.yml'):
        with open('results/best_tune.yml', 'r') as yml:
            model_param = yaml.load(yml)
        print("load model_param from 'results/best_tune.yml.'")
    else:
        print("load model_param from config.")
        model_param = model_params

    return model_param

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

def get_keys_from_value(dict, val):
    return [k for k, v in dict.items() if v == val]

def concat_csv(csv_paths):
    """複数のcsvファイルを結合する関数

    Parameters
    ----------
    csv_paths : list
        結合したいcsvファイルのパスのリスト

    Returns
    -------
    pandas.df
        csvファイルを結合してtypeを追加したpandasのデータフレーム
    """
    df_concat = pd.read_csv(csv_paths[0])
    df_concat['type'] = os.path.splitext(os.path.basename(csv_paths[0]))[0]

    for path in csv_paths[1:]:
        df_add = pd.read_csv(path)
        df_add['type'] = os.path.splitext(os.path.basename(path))[0]
        df_concat = pd.concat([df_concat,df_add])

    return df_concat


def get_latest_dir_name(path="./results"):
    """最新の時刻名のディレクトリを取得する関数

    Parameters
    ----------
    path    :   str
                名前が時刻で記述されているdir or fileが格納されているdirのパス
    
    Returns
    -------
    latest_folder   :   str
                        最新時刻のdir or fileの名前
    """
    folders = os.listdir(path)
    sorted_folders = sorted(folders, reverse=True)
    print('ディレクトリ    : ', path)
    print('ディレクトリ一覧: ', sorted_folders)

    # フォルダ一覧から最新の日付文字列のフォルダを選択する
    import re
    latest_folder = ''
    for folder in sorted_folders:
        if re.findall(r'^[0-9]', folder):
            latest_folder = folder
            break
        else:
            print(folder)
    if latest_folder == '':
        print('[ERROR] コピー対象のフォルダが見つかりませんでした')
        #exit()
        raise

    return latest_folder

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    if torch.cuda.is_available():
        nu_opt = '' if not no_units else ',nounits'
        cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
        output = subprocess.check_output(cmd, shell=True)
        lines = output.decode().split('\n')
        lines = [ line.strip() for line in lines if line.strip() != '' ]

        gpu_info =  [{ k: v for k, v in zip(keys, line.split(', ')) } for line in lines]

        min_gpu_index = 0
        min_gpu_memory_used = 100
        for gpu in gpu_info:
            gpu_index = gpu['index']
            gpu_memory = int(gpu['utilization.gpu'])
            if min_gpu_memory_used >= gpu_memory:
                min_gpu_memory_used = gpu_memory
                min_gpu_index = gpu_index

        return int(min_gpu_index)
    else:
        return 'cpu'

if __name__=='__main__':
    print(get_gpu_info())

