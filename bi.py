import joblib
import graph_process
from graph_process import graph_statistic
import csv
import re
import os
import ast
import shutil
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
            total_param = []
            for graph_centrality in df[param]:
                for centrality in ast.literal_eval(graph_centrality).values():
                    total_param.append(centrality)
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
    color_list = ['blue','red','green','gray']
    for param in eval_params:
        fig = plt.figure()
        for path,color in zip(csv_paths,color_list):
            df = pd.read_csv(path)
            label_name = os.path.splitext(os.path.basename(path))[0]
            if re.search('centrality', param):
                # 全グラフのノードのパラメータを１つのリストにまとめる
                # 原因はわからないがなぜかstrで保存されてしまうのでdictに再変換:ast.literal_eval(graph_centrality)
                total_param = []
                for graph_centrality in df[param]:
                    for centrality in ast.literal_eval(graph_centrality).values():
                        total_param.append(centrality)
                print('here')
                sns.histplot(total_param,label=label_name, kde=False, color=color)
            else:
                if eval_params_limit[param] is not None:
                    plt.xlim(eval_params_limit[param][0],eval_params_limit[param][1])
                sns.histplot(df[param],label=label_name, kde=False, color=color)

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
    plt.savefig('./visualize/pair_plot/pair_plot.pdf')
    fig.clf()
    plt.close('all')

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

def log_plot():
    cx = graph_process.complex_networks()
    dataset = cx.create_dataset({"twitter":[None,None,[None]]})

    all_nodes = 0
    for graph in dataset:
        all_nodes += graph.number_of_nodes() #全ノード数を受け取る
        degree_dist_dict = {}
        for node_num in graph.nodes():
            num_of_degree = graph.degree(node_num)
            if num_of_degree not in degree_dist_dict.keys():
                degree_dist_dict.update({num_of_degree:1})
            else:
                degree_dist_dict[num_of_degree] += 1

    for key,value in degree_dist_dict.items():
        degree_dist_dict[key] = value/all_nodes

    degree_dist_dict = sorted(degree_dist_dict.items(), key=lambda x:x[0])

    x = [i[0] for i in degree_dist_dict]
    y = [i[1] for i in degree_dist_dict]

    x = np.log(np.array(x))
    y = np.log(np.array(y))

    plt.scatter(x,y)
    plt.show()

    # param, cov = curve_fit(self.fitting_function, x, y)
    # return param[0]

def log_log():
    cx = graph_process.complex_networks()

    required_dirs = ["visualize/power_degree_line"]
    names = ["NN_0.1","NN_0.5","NN_0.9","twitter"]
    if os.path.isdir("visualize/power_degree_line/"):
        shutil.rmtree("visualize/power_degree_line")
    utils.make_dir(required_dirs)

    for detail,name in zip([{"NN":[1,10000,[0.1]]},{"NN":[1,10000,[0.5]]},{"NN":[1,10000,[0.9]]},{"twitter":[1,10000,[0.1]]}],names):
        dataset = cx.create_dataset(detail)

        degree = []
        for G in dataset:
            degree.extend(list(dict(nx.degree(G)).values()))
        # degree = list(dict(nx.degree(dataset[0])).values())

        import collections
        power_degree = dict(collections.Counter(degree))
        power_degree = sorted(power_degree.items(), key=lambda x:x[0])
        x = []
        y = []
        
        for i in power_degree:
            num = i[0]
            amount = i[1]
            x.append(num)
            y.append(amount)
        y = np.array(y) / sum(y)#次数を確率化
        sum_prob = 0
        for index,prob in enumerate(y):
            sum_prob += prob
            if sum_prob >= power_degree_border_line:
                border_index = index + 1
                break

        x_log = np.log(np.array(x))
        y_log = np.log(np.array(y))

        x_split_plot = x_log[border_index:]
        y_split_plot = y_log[border_index:]

        print(np.polyfit(x_log,y_log,1))
        print(np.polyfit(x_split_plot,y_split_plot,1))

        plt.figure(dpi=50, figsize=(10, 10))
        plt.scatter(x_log, y_log, marker='o',lw=0)
        plt.plot(x_log, np.poly1d(np.polyfit(x_log, y_log, 1))(x_log), label='d=1')
        plt.plot(x_split_plot, np.poly1d(np.polyfit(x_split_plot, y_split_plot, 1))(x_split_plot), label='split')
        # plt.yscale('log')
        # plt.xscale('log')
        plt.yticks(fontsize=20)
        plt.xlabel('degree', fontsize=24)
        plt.savefig("visualize/power_degree_line/" + name + ".png")
        plt.clf()

def generate_result2csv():
    gene_result = []
    for index in range(len(cluster_coefficient_label)):
        gene_result.append(joblib.load('./eval_result/generated_graph_'+str(index)))

    cn = graph_process.complex_networks()

    for index,result in enumerate(gene_result):
        cn.graph2csv(result, './data/csv/generated_graph_'+str(index)+'.csv')

if __name__ == '__main__':
    graphs = joblib.load('./data/result_graph')
    graph_visualize(graphs[0])
    for graph in graphs:
        print(graph.number_of_nodes())
    print(len(graphs))