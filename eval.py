from graph_process import complex_networks
import graph_process
import utils
import preprocess as pp
from config import *
import model
import graph_process as gp
import networkx as nx
import matplotlib.pyplot as plt
import os

import numpy as np
import joblib
import shutil

import torch

def eval(args):
    is_preprocess = args.preprocess

    device = utils.get_gpu_info()

    # recreate directory
    if utils.is_dir_existed("eval_result"):
        print("delete file...")
        print("- eval_result")
        shutil.rmtree("./eval_result")

    # 必須ディレクトリの作成
    required_dirs = ["dataset", "param", "results"]
    remove_dirs = []
    for dir in required_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        required_dirs.remove(dir)
    if len(required_dirs) > 0:
        utils.make_dir(required_dirs)

    # results内のディレクトリの候補を作成
    if args.result_dir:
        if not os.path.exists("./" + args.result_dir):
            print(f"{args.result_dir} が存在しません.")
            exit()
        result_dirs = [args.result_dir, args.result_dir+"/train", args.result_dir+"/eval", args.result_dir+"/visualize"]
    else:
        result_dirs = ["results/"+run_time, "results/"+run_time+"/train", "results/"+run_time+"/eval", "results/"+run_time+"/visualize"]
    train_dir = "./" + result_dirs[1] + "/"
    eval_dir = "./" + result_dirs[2] + "/"
    remove_dirs = []
    for dir in result_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        result_dirs.remove(dir)

    # evalディレクトリ内のディレクトリ候補を作成
    eval_dirs = [
                eval_dir + "statistic",
                eval_dir + "tsne",
                eval_dir + "dist_compare",
                eval_dir + "generated_normal",
                eval_dir + "generated_encoded",
                eval_dir + "reconstruct"]
    remove_dirs = []
    for dir in eval_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        eval_dirs.remove(dir)


    train_label = joblib.load("dataset/train/label")
    time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
    dfs_size = 2*time_size+2*node_size+edge_size+conditional_size

    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("--------------")

    # load model_param
    model_param = utils.load_model_param(file_path=args.model_param)
    print(f"model_param = {model_param}")

    is_sufficient_size=lambda graph: True if graph.number_of_nodes()>size_th else False

    vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param, device)
    # specify eval_model if it is exist, otherwise specify run_time
    if args.eval_model:
        vae.load_state_dict(torch.load(args.eval_model, map_location="cpu"))
    else:
        # vae.load_state_dict(torch.load("param/weight", map_location="cpu"))
        vae.load_state_dict(torch.load(train_dir + "weight", map_location="cpu"))

    vae = utils.try_gpu(device,vae)

    vae.eval()

    keys = ["tu", "tv", "lu", "lv", "le"]

    cx = complex_networks()
    conditional_label = [[cluster_coefficient_label[i]]\
        for i in range(len(cluster_coefficient_label))]
    conditional_label = torch.tensor(conditional_label)

    result_low = vae.generate(300,torch.tensor([1,0,0]))
    result_middle = vae.generate(300,torch.tensor([0,1,0]))
    result_high = vae.generate(300,torch.tensor([0,0,1]))

    result_all = [result_low,result_middle,result_high]

    end_value_list = [time_size, time_size, node_size, node_size, edge_size]
    correct_all = graph_process.divide_label(train_label,end_value_list)

    results = {}
    generated_keys=[]

    for index,(result,correct_graph) in enumerate(zip(result_all,correct_all)):
    # generated graphs
        result = [code.unsqueeze(2) for code in result]
        dfs_code = torch.cat(result, dim=2)
        generated_graph = []
        for code in dfs_code:
            graph = gp.dfs_code_to_graph_obj(
                    code.cpu().detach().numpy(),
                    [time_size, time_size, node_size, node_size, edge_size])
            #if gp.is_connect(graph):
            if gp.is_connect(graph) and is_sufficient_size(graph):
                generated_graph.append(graph)
        
        # make result_dirs, eval_dirs once
        if len(result_dirs) > 0:
            utils.make_dir(result_dirs)
            result_dirs = []
        if len(eval_dirs) > 0:
            utils.make_dir(eval_dirs)
            eval_dirs = []

        joblib.dump(generated_graph, eval_dir + 'generated_graph_'+str(index))

    #     gs = gp.graph_statistic()
    #     dict_tmp = {"correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index]): {key: [] for key in eval_params}}
    #     results.update(dict_tmp)
    #     dict_tmp = {str(power_degree_label[index])+" "+str(cluster_coefficient_label[index]): {key: [] for key in eval_params}}
    #     results.update(dict_tmp)

    #     #生成グラフのkeyの保存
    #     key=str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])
    #     generated_keys.append(key)

    #     for generated, correct in zip(generated_graph, correct_graph):
    #         for key in eval_params:
    #             if "degree" in key:
    #                 gamma = gs.degree_dist(generated)
    #                 results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gamma)
    #                 gamma = gs.degree_dist(correct)
    #                 results["correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gamma)
    #             if "cluster" in key:
    #                 results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gs.cluster_coeff(generated))
    #                 results["correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gs.cluster_coeff(correct))
    #             if "distance" in key:
    #                 results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gs.ave_dist(generated))
    #                 results["correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(gs.ave_dist(correct))
    #             if "size" in key:
    #                 results[str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(generated.number_of_nodes())
    #                 results["correct"+str(power_degree_label[index])+" "+str(cluster_coefficient_label[index])][key].append(correct.number_of_nodes())
                    
    # # display result
    # with open('eval_result/statistic/log.txt', 'w') as f:
    #     for key, value in results.items():
    #         print("====================================")
    #         print("%s:"%(key), file=f)
    #         print("%s:"%(key))
    #         print("====================================")
    #         for trait_key in value.keys():
    #             print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])), file=f)
    #             print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])))
    #             print(" %s var: %lf"%(trait_key, np.var(value[trait_key])), file=f)
    #             print(" %s var: %lf"%(trait_key, np.var(value[trait_key])))
    #             print("------------------------------------")
    #         print("\n")

    # # boxplot
    # for param_key in eval_params:
    #     dict = {}
    #     keys = list(results.keys())
    #     utils.box_plot(
    #             {keys[1]: results[keys[1]][param_key][:graph_num],
    #             keys[3]: results[keys[3]][param_key][:graph_num],
    #             keys[5]: results[keys[5]][param_key][:graph_num]},
    #             {keys[0]: results[keys[0]][param_key][:graph_num],
    #             keys[2]: results[keys[2]][param_key][:graph_num],
    #             keys[4]: results[keys[4]][param_key][:graph_num]},
    #             param_key,
    #             "eval_result/generated_normal/%s_box_plot.png"%(param_key)
    #             )

    # # 散布図
    # combinations = utils.combination(eval_params, 2)
    # colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    # for key1, key2 in combinations:
    #     plt.figure()
    #     for i, conditionkey in enumerate(generated_keys):
    #         plt.scatter(
    #             results[conditionkey][key1],
    #             results[conditionkey][key2],
    #             c=colorlist[i],
    #             label=conditionkey,
    #             )
    #     plt.legend()
    #     plt.xlabel(key1)
    #     plt.ylabel(key2)
    #     plt.savefig("eval_result/dist_compare/%s_%s.png"%(key1, key2))
    #     plt.close()

    # # datasetの読み込み
    # train_dataset = joblib.load("dataset/train/onehot")
    # train_conditional = joblib.load("dataset/train/conditional")
    # train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1)
    # train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
    # train_dataset = utils.try_gpu(device,train_dataset)

    # # conditionalのlabelと同じラベルの引数のget
    # tmp=train_conditional.squeeze()
    # uniqued, inverses=torch.unique(tmp, return_inverse=True, dim=0)
    # conditional_vecs=uniqued
    # same_conditional_args=[[j for j in range(len(inverses)) if inverses[j]==i] for i in range(uniqued.shape[0])]

    # # 入力に応じたencode+predict
    # # 入力に応じたencode+generate
    # get_key=lambda vec: str(power_degree_label[torch.argmax(vec[:3])])+" "+\
    #             str(cluster_coefficient_label[torch.argmax(vec[3:])]) # conditional vec->key
    # reconstruct_graphs={}
    # encoded_generate_graphs={}
    # for i, args in enumerate(same_conditional_args):
    #     # trait keyの作成
    #     traitkey=get_key(conditional_vecs[i][0])

    #     # predict
    #     mu, sigma, *reconstruct_result = vae(train_dataset[args], 0.0)
    #     # generate
    #     z=vae.encode(train_dataset[args])
    #     generated_result=vae.generate(1000, utils.try_gpu(device,conditional_vecs[i][0]), z=z)

    #     # graphに変換
    #     # reconstruct
    #     tmps=[]
    #     for tmp in reconstruct_result:
    #         tmps.append(torch.argmax(tmp, dim=2).unsqueeze(2))
    #     dfs_code = torch.cat(tmps, dim=2)
    #     reconstruct_graph=[]
    #     for code in dfs_code:
    #         graph = gp.dfs_code_to_graph_obj(
    #                 code.cpu().detach().numpy(),
    #                 [time_size, time_size, node_size, node_size, edge_size])
    #         if gp.is_connect(graph):
    #         #if gp.is_connect(graph) and is_sufficient_size(graph):
    #             reconstruct_graph.append(graph)
    #     reconstruct_graphs[traitkey]=gs.calc_graph_traits(reconstruct_graph, eval_params) # 特性値をcalc
    #     # generated
    #     tmps=[]
    #     for tmp in generated_result:
    #         tmps.append(tmp.unsqueeze(2))
    #     dfs_code = torch.cat(tmps, dim=2)
    #     generated_graph=[]
    #     for code in dfs_code:
    #         graph = gp.dfs_code_to_graph_obj(
    #                 code.cpu().detach().numpy(),
    #                 [time_size, time_size, node_size, node_size, edge_size])
    #         if gp.is_connect(graph):
    #         #if gp.is_connect(graph) and is_sufficient_size(graph):
    #             generated_graph.append(graph)
    #     encoded_generate_graphs[traitkey]=gs.calc_graph_traits(generated_graph, eval_params) # 特性値をcalc

    # # display result
    # for key, value in reconstruct_graphs.items():
    #     print("====================================")
    #     print("reconstruct %s:"%(key))
    #     print("====================================")
    #     for trait_key in value.keys():
    #         print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])))
    #         print(" %s var: %lf"%(trait_key, np.var(value[trait_key])))
    #         print("------------------------------------")
    #     print("\n")
    # for key, value in encoded_generate_graphs.items():
    #     print("====================================")
    #     print("encoded generate %s:"%(key))
    #     print("====================================")
    #     for trait_key in value.keys():
    #         print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])))
    #         print(" %s var: %lf"%(trait_key, np.var(value[trait_key])))
    #         print("------------------------------------")
    #     print("\n")

    # # boxplot
    # for param_key in eval_params:
    #     keys = list(results.keys())
    #     keys = list(sorted([keys[0], keys[2], keys[4]]))
    #     reconstructkeys=list(sorted(list(reconstruct_graphs.keys())))
    #     utils.box_plot(
    #             {reconstructkeys[0]: reconstruct_graphs[reconstructkeys[0]][param_key][:100],
    #             reconstructkeys[1]: reconstruct_graphs[reconstructkeys[1]][param_key][:100],
    #             reconstructkeys[2]: reconstruct_graphs[reconstructkeys[2]][param_key][:100]},
    #             {keys[0]: results[keys[0]][param_key][:100],
    #             keys[1]: results[keys[1]][param_key][:100],
    #             keys[2]: results[keys[2]][param_key][:100]},
    #             param_key,
    #             "eval_result/reconstruct/%s_box_plot.png"%(param_key)
    #             )
    # for param_key in eval_params:
    #     keys = list(results.keys())
    #     keys = list(sorted([keys[0], keys[2], keys[4]]))
    #     encoded_generatekeys=list(sorted(list(encoded_generate_graphs.keys())))
    #     utils.box_plot(
    #             {encoded_generatekeys[0]: encoded_generate_graphs[encoded_generatekeys[0]][param_key][:graph_num],
    #             encoded_generatekeys[1]: encoded_generate_graphs[encoded_generatekeys[1]][param_key][:graph_num],
    #             encoded_generatekeys[2]: encoded_generate_graphs[encoded_generatekeys[2]][param_key][:graph_num]},
    #             {keys[0]: results[keys[0]][param_key][:graph_num],
    #             keys[1]: results[keys[1]][param_key][:graph_num],
    #             keys[2]: results[keys[2]][param_key][:graph_num]},
    #             param_key,
    #             "eval_result/generated_encoded/%s_box_plot.png"%(param_key)
    #             )

    # # t-SNE
    # # conditional vectorをcatしていない状態での埋め込み
    # result={}
    # for i, args in enumerate(same_conditional_args):
    #     # trait keyの作成
    #     traitkey=get_key(conditional_vecs[i][0])

    #     z = vae.encode(train_dataset[args]).cpu().detach().numpy()
    #     result[traitkey]=z
    # result["N(0, I)"]=vae.noise_generator(
    #         model_param["rep_size"], len(args)).cpu().unsqueeze(1).detach().numpy()
    # utils.tsne(result, "eval_result/tsne/raw_tsne.png")

    # # conditional vectorをcatして埋め込み
    # result={}
    # for i, args in enumerate(same_conditional_args):
    #     # trait keyの作成
    #     traitkey=get_key(conditional_vecs[i][0])
    #     # encode
    #     z = vae.encode(train_dataset[args])
    #     # conditional vecをcat
    #     tmp=conditional_vecs[i][0].unsqueeze(0).unsqueeze(0)
    #     catconditional=utils.try_gpu(device,torch.cat([tmp for _ in range(len(args))], dim=0))
    #     z=torch.cat([z, catconditional], dim=2)
    #     # save
    #     result[traitkey]=z.cpu().detach().numpy()

    #     # noiseにもcat
    #     noise=vae.noise_generator(
    #             model_param["rep_size"], len(args)).unsqueeze(1)
    #     noise=utils.try_gpu(device,noise)
    #     noise=torch.cat([noise, catconditional], dim=2)
    #     result["N(0, I) cat %s"%(traitkey)]=noise.cpu().detach().numpy()
    # utils.tsne(result, "eval_result/tsne/condition_cat_tsne1.png")
    # result={}
    # for i, args in enumerate(same_conditional_args):
    #     # trait keyの作成
    #     traitkey=get_key(conditional_vecs[i][0])
    #     # encode
    #     z = vae.encode(train_dataset[args])
    #     # conditional vecをcat
    #     tmp=conditional_vecs[i][0].unsqueeze(0).unsqueeze(0)
    #     catconditional=utils.try_gpu(device,torch.cat([tmp for _ in range(len(args))], dim=0))
    #     z=torch.cat([z, catconditional], dim=2)
    #     # save
    #     result[traitkey]=z.cpu().detach().numpy()

    #     # noiseにもcat
    #     noise=vae.noise_generator(
    #             model_param["rep_size"], len(args)).unsqueeze(1)
    #     noise=utils.try_gpu(device,noise)
    #     noise=torch.cat([noise, catconditional], dim=2)
    #     #result["N(0, I) cat %s"%(traitkey)]=noise.cpu().detach().numpy()
    # utils.tsne(result, "eval_result/tsne/condition_cat_tsne.png")


def non_conditional_eval(args):
    is_preprocess = args.preprocess

    device = utils.get_gpu_info()

    # recreate directory
    if utils.is_dir_existed("eval_result"):
        print("delete file...")
        print("- eval_result")
        shutil.rmtree("./eval_result")

    # 必須ディレクトリの作成
    required_dirs = ["dataset", "param", "results"]
    remove_dirs = []
    for dir in required_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        required_dirs.remove(dir)
    if len(required_dirs) > 0:
        utils.make_dir(required_dirs)
        
    # results内のディレクトリの候補を作成
    if args.result_dir:
        if not os.path.exists("./" + args.result_dir):
            print(f"{args.result_dir} が存在しません.")
            exit()
        result_dirs = [args.result_dir, args.result_dir+"/train", args.result_dir+"/eval", args.result_dir+"/visualize"]
    else:
        result_dirs = ["results/"+run_time, "results/"+run_time+"/train", "results/"+run_time+"/eval", "results/"+run_time+"/visualize"]
    train_dir = "./" + result_dirs[1] + "/"
    eval_dir = "./" + result_dirs[2] + "/"
    remove_dirs = []
    for dir in result_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        result_dirs.remove(dir)

    # evalディレクトリ内のディレクトリ候補を作成
    eval_dirs = [
                eval_dir + "statistic",
                eval_dir + "tsne",
                eval_dir + "dist_compare",
                eval_dir + "generated_normal",
                eval_dir + "generated_encoded",
                eval_dir + "reconstruct",
                eval_dir + "result_csv"]
    remove_dirs = []
    for dir in eval_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        eval_dirs.remove(dir)


    # train_label = joblib.load("dataset/train/label")
    time_size, node_size, edge_size, _ = joblib.load("dataset/param")
    dfs_size = 2*time_size + 2*node_size + edge_size

    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("--------------")

    # load model_param
    model_param = utils.load_model_param(file_path=args.model_param)
    print(f"model_param = {model_param}")

    is_sufficient_size=lambda graph: True if graph.number_of_nodes()>size_th else False

    vae = model.VAENonConditional(dfs_size, time_size, node_size, edge_size, model_param, device)
    # if args.eval_model:
    #     vae.load_state_dict(torch.load(args.eval_model, map_location="cpu"))
    # else:
        # vae.load_state_dict(torch.load("param/weight", map_location="cpu"))
    vae.load_state_dict(torch.load(train_dir + "weight", map_location="cpu"))
    vae = utils.try_gpu(device,vae)
    vae.eval()

    keys = ["tu", "tv", "lu", "lv", "le"]

    cx = complex_networks()

    # graph_num個のラベルなしグラフ生成
    result_nonc = vae.generate(graph_num)
    result_all = [result_nonc]

    end_value_list = [time_size, time_size, node_size, node_size, edge_size]

    results = {}
    generated_keys=[]

    for index,result in enumerate(result_all):
    # generated graphs
        # convert onehot to scalar
        scalar_graph = []
        for code in result:
            print(code)
            code = torch.argmax(code, dim=2)
            scalar_graph.append(code)
        scalar_graph = [code.unsqueeze(2) for code in scalar_graph]
        dfs_code = torch.cat(scalar_graph, dim=2)
        generated_graph = []
        for code in dfs_code:
            graph = gp.dfs_code_to_graph_obj(
                    code.cpu().detach().numpy(),
                    [time_size, time_size, node_size, node_size, edge_size])
            #if gp.is_connect(graph):
            if gp.is_connect(graph) and is_sufficient_size(graph):
                generated_graph.append(graph)

    # make result_dirs, eval_dirs once
    if len(result_dirs) > 0:
        utils.make_dir(result_dirs)
        result_dirs = []
    if len(eval_dirs) > 0:
        utils.make_dir(eval_dirs)
        eval_dirs = []

    # joblib.dump(generated_graph,'./data/result_graph')
    joblib.dump(generated_graph, eval_dir + 'result_graph')

    # 生成グラフをcsvファイルに書き出し
    # cx.graph2csv(generated_graph, 'result_csv/twitter_result')
    cx.graph2csv(generated_graph, 'twitter_result', result_csv_dir=eval_dir+"result_csv/")
                    
    # display result
    # with open('eval_result/statistic/log.txt', 'w') as f:
    #     for key, value in results.items():
    #         print("====================================")
    #         print("%s:"%(key), file=f)
    #         print("%s:"%(key))
    #         print("====================================")
    #         for trait_key in value.keys():
    #             print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])), file=f)
    #             print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])))
    #             print(" %s var: %lf"%(trait_key, np.var(value[trait_key])), file=f)
    #             print(" %s var: %lf"%(trait_key, np.var(value[trait_key])))
    #             print("------------------------------------")
    #         print("\n")

    # # boxplot
    # for param_key in eval_params:
    #     dict = {}
    #     keys = list(results.keys())
    #     utils.box_plot(
    #             {keys[1]: results[keys[1]][param_key][:graph_num],
    #             keys[3]: results[keys[3]][param_key][:graph_num],
    #             keys[5]: results[keys[5]][param_key][:graph_num]},
    #             {keys[0]: results[keys[0]][param_key][:graph_num],
    #             keys[2]: results[keys[2]][param_key][:graph_num],
    #             keys[4]: results[keys[4]][param_key][:graph_num]},
    #             param_key,
    #             "eval_result/generated_normal/%s_box_plot.png"%(param_key)
    #             )

    # # 散布図
    # combinations = utils.combination(eval_params, 2)
    # colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    # for key1, key2 in combinations:
    #     plt.figure()
    #     for i, conditionkey in enumerate(generated_keys):
    #         plt.scatter(
    #             results[conditionkey][key1],
    #             results[conditionkey][key2],
    #             c=colorlist[i],
    #             label=conditionkey,
    #             )
    #     plt.legend()
    #     plt.xlabel(key1)
    #     plt.ylabel(key2)
    #     plt.savefig("eval_result/dist_compare/%s_%s.png"%(key1, key2))
    #     plt.close()

    # # datasetの読み込み
    # train_dataset = joblib.load("dataset/train/onehot")
    # train_conditional = joblib.load("dataset/train/conditional")
    # train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1)
    # train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
    # train_dataset = utils.try_gpu(device,train_dataset)

    # # conditionalのlabelと同じラベルの引数のget
    # tmp=train_conditional.squeeze()
    # uniqued, inverses=torch.unique(tmp, return_inverse=True, dim=0)
    # conditional_vecs=uniqued
    # same_conditional_args=[[j for j in range(len(inverses)) if inverses[j]==i] for i in range(uniqued.shape[0])]

    # # 入力に応じたencode+predict
    # # 入力に応じたencode+generate
    # get_key=lambda vec: str(power_degree_label[torch.argmax(vec[:3])])+" "+\
    #             str(cluster_coefficient_label[torch.argmax(vec[3:])]) # conditional vec->key
    # reconstruct_graphs={}
    # encoded_generate_graphs={}
    # for i, args in enumerate(same_conditional_args):
    #     # trait keyの作成
    #     traitkey=get_key(conditional_vecs[i][0])

    #     # predict
    #     mu, sigma, *reconstruct_result = vae(train_dataset[args], 0.0)
    #     # generate
    #     z=vae.encode(train_dataset[args])
    #     generated_result=vae.generate(1000, utils.try_gpu(device,conditional_vecs[i][0]), z=z)

    #     # graphに変換
    #     # reconstruct
    #     tmps=[]
    #     for tmp in reconstruct_result:
    #         tmps.append(torch.argmax(tmp, dim=2).unsqueeze(2))
    #     dfs_code = torch.cat(tmps, dim=2)
    #     reconstruct_graph=[]
    #     for code in dfs_code:
    #         graph = gp.dfs_code_to_graph_obj(
    #                 code.cpu().detach().numpy(),
    #                 [time_size, time_size, node_size, node_size, edge_size])
    #         if gp.is_connect(graph):
    #         #if gp.is_connect(graph) and is_sufficient_size(graph):
    #             reconstruct_graph.append(graph)
    #     reconstruct_graphs[traitkey]=gs.calc_graph_traits(reconstruct_graph, eval_params) # 特性値をcalc
    #     # generated
    #     tmps=[]
    #     for tmp in generated_result:
    #         tmps.append(tmp.unsqueeze(2))
    #     dfs_code = torch.cat(tmps, dim=2)
    #     generated_graph=[]
    #     for code in dfs_code:
    #         graph = gp.dfs_code_to_graph_obj(
    #                 code.cpu().detach().numpy(),
    #                 [time_size, time_size, node_size, node_size, edge_size])
    #         if gp.is_connect(graph):
    #         #if gp.is_connect(graph) and is_sufficient_size(graph):
    #             generated_graph.append(graph)
    #     encoded_generate_graphs[traitkey]=gs.calc_graph_traits(generated_graph, eval_params) # 特性値をcalc

    # # display result
    # for key, value in reconstruct_graphs.items():
    #     print("====================================")
    #     print("reconstruct %s:"%(key))
    #     print("====================================")
    #     for trait_key in value.keys():
    #         print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])))
    #         print(" %s var: %lf"%(trait_key, np.var(value[trait_key])))
    #         print("------------------------------------")
    #     print("\n")
    # for key, value in encoded_generate_graphs.items():
    #     print("====================================")
    #     print("encoded generate %s:"%(key))
    #     print("====================================")
    #     for trait_key in value.keys():
    #         print(" %s ave: %lf"%(trait_key, np.average(value[trait_key])))
    #         print(" %s var: %lf"%(trait_key, np.var(value[trait_key])))
    #         print("------------------------------------")
    #     print("\n")

    # # boxplot
    # for param_key in eval_params:
    #     keys = list(results.keys())
    #     keys = list(sorted([keys[0], keys[2], keys[4]]))
    #     reconstructkeys=list(sorted(list(reconstruct_graphs.keys())))
    #     utils.box_plot(
    #             {reconstructkeys[0]: reconstruct_graphs[reconstructkeys[0]][param_key][:100],
    #             reconstructkeys[1]: reconstruct_graphs[reconstructkeys[1]][param_key][:100],
    #             reconstructkeys[2]: reconstruct_graphs[reconstructkeys[2]][param_key][:100]},
    #             {keys[0]: results[keys[0]][param_key][:100],
    #             keys[1]: results[keys[1]][param_key][:100],
    #             keys[2]: results[keys[2]][param_key][:100]},
    #             param_key,
    #             "eval_result/reconstruct/%s_box_plot.png"%(param_key)
    #             )
    # for param_key in eval_params:
    #     keys = list(results.keys())
    #     keys = list(sorted([keys[0], keys[2], keys[4]]))
    #     encoded_generatekeys=list(sorted(list(encoded_generate_graphs.keys())))
    #     utils.box_plot(
    #             {encoded_generatekeys[0]: encoded_generate_graphs[encoded_generatekeys[0]][param_key][:graph_num],
    #             encoded_generatekeys[1]: encoded_generate_graphs[encoded_generatekeys[1]][param_key][:graph_num],
    #             encoded_generatekeys[2]: encoded_generate_graphs[encoded_generatekeys[2]][param_key][:graph_num]},
    #             {keys[0]: results[keys[0]][param_key][:graph_num],
    #             keys[1]: results[keys[1]][param_key][:graph_num],
    #             keys[2]: results[keys[2]][param_key][:graph_num]},
    #             param_key,
    #             "eval_result/generated_encoded/%s_box_plot.png"%(param_key)
    #             )

    # # t-SNE
    # # conditional vectorをcatしていない状態での埋め込み
    # result={}
    # for i, args in enumerate(same_conditional_args):
    #     # trait keyの作成
    #     traitkey=get_key(conditional_vecs[i][0])

    #     z = vae.encode(train_dataset[args]).cpu().detach().numpy()
    #     result[traitkey]=z
    # result["N(0, I)"]=vae.noise_generator(
    #         model_param["rep_size"], len(args)).cpu().unsqueeze(1).detach().numpy()
    # utils.tsne(result, "eval_result/tsne/raw_tsne.png")

    # # conditional vectorをcatして埋め込み
    # result={}
    # for i, args in enumerate(same_conditional_args):
    #     # trait keyの作成
    #     traitkey=get_key(conditional_vecs[i][0])
    #     # encode
    #     z = vae.encode(train_dataset[args])
    #     # conditional vecをcat
    #     tmp=conditional_vecs[i][0].unsqueeze(0).unsqueeze(0)
    #     catconditional=utils.try_gpu(device,torch.cat([tmp for _ in range(len(args))], dim=0))
    #     z=torch.cat([z, catconditional], dim=2)
    #     # save
    #     result[traitkey]=z.cpu().detach().numpy()

    #     # noiseにもcat
    #     noise=vae.noise_generator(
    #             model_param["rep_size"], len(args)).unsqueeze(1)
    #     noise=utils.try_gpu(device,noise)
    #     noise=torch.cat([noise, catconditional], dim=2)
    #     result["N(0, I) cat %s"%(traitkey)]=noise.cpu().detach().numpy()
    # utils.tsne(result, "eval_result/tsne/condition_cat_tsne1.png")
    # result={}
    # for i, args in enumerate(same_conditional_args):
    #     # trait keyの作成
    #     traitkey=get_key(conditional_vecs[i][0])
    #     # encode
    #     z = vae.encode(train_dataset[args])
    #     # conditional vecをcat
    #     tmp=conditional_vecs[i][0].unsqueeze(0).unsqueeze(0)
    #     catconditional=utils.try_gpu(device,torch.cat([tmp for _ in range(len(args))], dim=0))
    #     z=torch.cat([z, catconditional], dim=2)
    #     # save
    #     result[traitkey]=z.cpu().detach().numpy()

    #     # noiseにもcat
    #     noise=vae.noise_generator(
    #             model_param["rep_size"], len(args)).unsqueeze(1)
    #     noise=utils.try_gpu(device,noise)
    #     noise=torch.cat([noise, catconditional], dim=2)
    #     #result["N(0, I) cat %s"%(traitkey)]=noise.cpu().detach().numpy()
    # utils.tsne(result, "eval_result/tsne/condition_cat_tsne.png")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--classifier',action='store_true')
    parser.add_argument('--condition', action='store_true')
    parser.add_argument('--use_model')

    parser.add_argument('--model_param')
    parser.add_argument('--result_dir')
    parser.add_argument('--eval_model')

    args = parser.parse_args()
    if args.condition:
        if args.use_model == 'lstm' or args.use_model == 'LSTM':
            eval(args)
    else:
        if args.use_model == 'lstm' or args.use_model == 'LSTM':
            non_conditional_eval(args)
