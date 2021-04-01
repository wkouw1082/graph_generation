import joblib
from graph_process import graph_statistic
import csv
import re
import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from config import *
import utils


def graph_visualize(graph):
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph,pos)

    plt.axis("off")
    # plt.savefig("default.png")
    plt.show()

def convert_csv2image(csv_path):
    '''
    csvファイルからeval_paramsのパラメータの棒グラフを作成する関数

    Parameters
    ----------
    csv_path : string
        読み込むcsvファイルを指定する変数
    '''
    data = pd.read_csv(csv_path, encoding = 'UTF8')
    for param in eval_params:
        sns.histplot(data[param], kde=False)
        plt.savefig('./visualize/' + param + '.png')
        plt.clf()

def histogram_visualize(csv_path):
    """ヒストグラムを作成する関数

    Parameters
    ----------
    csv_path : str
        ヒストグラムを作成したいcsvファイルのパス
    """
    dir_name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    for param in eval_params:
        fig = plt.figure()
        if re.search('centrality', param):
            # 全グラフのノードのパラメータを１つのリストにまとめる
            # 原因はわからないがなぜかstrで保存されてしまうのでdictに再変換:ast.literal_eval(graph_centrality)
            total_param = [centrality for graph_centrality in df[param] for centrality in ast.literal_eval(graph_centrality).values()]
            sns.histplot(total_param, kde=False)
        else:
            if eval_params_limit[param] is not None:
                plt.xlim(eval_params_limit[param][0],eval_params_limit[param][1])
            sns.histplot(df[param],kde=False)

        plt.savefig('./visualize/histogram/'+dir_name+'/'+ param + '.png')
        plt.clf()
        plt.close('all')

def concat_histogram_visualize(save_dir,csv_paths):
    """複数のデータを結合したヒストグラムを作成する関数

    Parameters
    ----------
    save_dir : str
        保存するディレクトリの名前
    csv_paths : list
        ヒストグラムを作成するcsvファイルパスのリスト
    """
    for param in eval_params:
        fig = plt.figure()
        for path in csv_paths:
            df = pd.read_csv(path)
            label_name = os.path.splitext(os.path.basename(path))[0]
            if re.search('centrality', param):
                # 全グラフのノードのパラメータを１つのリストにまとめる
                # 原因はわからないがなぜかstrで保存されてしまうのでdictに再変換:ast.literal_eval(graph_centrality)
                total_param = [centrality for graph_centrality in df[param] for centrality in ast.literal_eval(graph_centrality).values()]
                sns.histplot(total_param,label=label_name, kde=False)
            else:
                if eval_params_limit[param] is not None:
                    plt.xlim(eval_params_limit[param][0],eval_params_limit[param][1])
                sns.histplot(df[param],label=label_name, kde=False)

        plt.legend(frameon=True)
        plt.savefig('./visualize/concat_histogram/'+save_dir+'/'+ param + '.png')
        plt.clf()
        plt.close('all')

def scatter_diagram_visualize(csv_path):
    """散布図を作成する関数
       なお、作成時にはeval paramsの全ての組み合わせが作成される

    Parameters
    ----------
    csv_path : str
        散布図を作成したいcsvfileのpath

    >>> scatter_diagram_visualize('./data/Twitter/twitter.csv')
    """
    dir_name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    for param_v in eval_params:
        for param_u in eval_params:
            if re.search('centrality', param_v) or re.search('centrality', param_u) or param_v == param_u:
                continue
            fig = plt.figure()
            x_data = df[param_v]
            y_data = df[param_u]
            sns.jointplot(x=x_data,y=y_data,data=df)
            plt.savefig('./visualize/scatter_diagram/'+dir_name+'/'+param_v+'_'+param_u+'.png')
            fig.clf()
            plt.close('all')

def concat_scatter_diagram_visualize(save_dir,csv_paths):
    for param_v in eval_params:
        for param_u in eval_params:
            if re.search('centrality', param_v) or re.search('centrality', param_u) or param_v == param_u:
                continue
            fig = plt.figure()
            df = utils.concat_csv(csv_paths)
            sns.jointplot(x=df[param_v],y=df[param_u],data=df,hue='type')

            plt.savefig('./visualize/concat_scatter_diagram/'+save_dir+'/'+param_v+'_'+param_u+'.png')
            fig.clf()
            plt.close('all')

def pair_plot(csv_paths):
    fig = plt.figure()
    df = utils.concat_csv(csv_paths)
    sns.pairplot(df,data=df,hue='type')
    plt.savefig('./visualize/pair_plot/pair_plot.png')
    fig.clf()
    plt.close('all')

def box_plot(generate, correct, trait_name, directory):
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

if __name__ == '__main__':
    #graph = joblib.load('./data/Twitter/twitter.pkl.cmp')
    #graph2csv(graph, 'Twitter/twitter.csv')
    import doctest
    doctest.testmod()