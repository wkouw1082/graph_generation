import joblib
from graph_process import graph_statistic
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import *


def graph_visualize(graph):
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(graph)
    nx.draw_networkx(G,pos)

    plt.axis("off")
    # plt.savefig("default.png")
    plt.show()

def make_graph_data_distribution(graph_data, save_path):
    '''
    各グラフデータのパラメータをcsvファイルに出力する関数

    Parameters
    ----------
    graph_data : list
        グラフデータが格納されているリスト [GraphObj, ...]
    save_path : string
        csvファイルの格納先フォルダを指定する変数 例:Twitter/twitter.csv
    '''
    statistic = graph_statistic()
    trait_dict = statistic.calc_graph_traits2csv(graph_data, eval_params)
    with open('./data/' + save_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=eval_params)
        writer.writeheader()
        writer.writerows(trait_dict)

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



if __name__ == '__main__':
    #graph = joblib.load('./data/Twitter/twitter.pkl.cmp')
    #make_graph_data_distribution(graph, 'Twitter/twitter.csv')
    convert_csv2image('./data/Twitter/twitter.csv')