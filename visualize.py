import bi
import graph_process
import utils

import os
import argparse
import joblib
import shutil

from config import *

def main(args):
    # args = parser.parse_args()

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

    # results内のディレクトリを作成
    if args.result_dir:
        if not os.path.exists("./" + args.result_dir):
            print(f"{args.result_dir} が存在しません.")
            exit()
        result_dirs = [args.result_dir, args.result_dir+"/train", args.result_dir+"/eval", args.result_dir+"/visualize"]
    else:
        result_dirs = ["results/"+run_time, "results/"+run_time+"/train", "results/"+run_time+"/eval", "results/"+run_time+"/visualize"]
    visualize_dir = "./" + result_dirs[3] + "/"
    remove_dirs = []
    for dir in result_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        result_dirs.remove(dir)
    if len(result_dirs) > 0:
        utils.make_dir(result_dirs)


    if args.preprocess:
        # 危険なのでコメントアウト
        # すでにあるディレクトリを削除して再作成
        # if os.path.isdir("data/csv/"):
        #     shutil.rmtree("data/csv")
        # required_dirs = ["data/csv"]
        # utils.make_dir(required_dirs)

        # resultsの時系列ディレクトリ直下にcsvディレクトリを作成
        if not os.path.exists(visualize_dir + "csv"):
            required_dirs = [visualize_dir + "csv"]
            utils.make_dir(required_dirs)

        # visualize_detailをもとにデータセットを作成
        cn = graph_process.complex_networks()
        dataset = cn.create_dataset(visualize_detail, do_type='visualize')
        # csvファイルに変換
        for name,data in zip(visualize_types.keys(),dataset):
            cn.graph2csv(data,name)

        # evalで生成したグラフの特徴量をcsvへ吐き出す
        result_dir_name = visualize_dir.split("/")[-3]
        bi.generate_result2csv(result_path=result_dir_name)


    # args type が何も指定されていない場合は全てのtypeが指定され、指定がある場合はそのtypeのcsv pathが持ってこられる
    # csv_paths = [visualize_types[key] for key in args.type] if args.type is not None else utils.get_directory_paths(visualize_dir + 'csv/*')
    csv_paths = [visualize_types[key] for key in args.type] if args.type is not None else utils.get_directory_paths('./data/csv/*')
    csv_paths = sorted(csv_paths)

    if args.scatter:
        if os.path.isdir(visualize_dir + "scatter_diagram/"):
            shutil.rmtree(visualize_dir + "scatter_diagram")
        # csvのファイルパスからdir名を持ってくる
        dir_names = [os.path.splitext(os.path.basename(csv_path))[0] for csv_path in csv_paths]
        # dir名からdirを生成
        required_dirs = [visualize_dir + "scatter_diagram"] + [visualize_dir + "scatter_diagram/" + dir_name for dir_name in dir_names]
        utils.make_dir(required_dirs)

        for path in csv_paths:
                bi.scatter_diagram_visualize(path, output_path=visualize_dir)

    if args.histogram:
        if os.path.isdir(visualize_dir + "histogram/"):
            shutil.rmtree(visualize_dir + "histogram")
        # csvのファイルパスからdir名を持ってくる
        dir_names = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]
        # dir名からdirを生成
        required_dirs = [visualize_dir + "histogram"] + [visualize_dir + "histogram/" + dir_name for dir_name in dir_names]
        utils.make_dir(required_dirs)

        for path in csv_paths:
            bi.histogram_visualize(path, output_path=visualize_dir)
        
    if args.concat_scatter:
        if os.path.isdir(visualize_dir + "concat_scatter_diagram/"):
            shutil.rmtree(visualize_dir + "concat_scatter_diagram")
        dir_name = ''
        for index,path in enumerate(csv_paths):
            dir_name += os.path.splitext(os.path.basename(path))[0]
            if index != len(csv_paths)-1:
                dir_name += '&'
        # dir名からdirを生成
        required_dirs = [visualize_dir + "concat_scatter_diagram"] + [visualize_dir + "concat_scatter_diagram/" + dir_name]
        utils.make_dir(required_dirs)
        
        bi.concat_scatter_diagram_visualize(dir_name,csv_paths, output_path=visualize_dir)

    if args.concat_histogram:
        if os.path.isdir(visualize_dir + "concat_histogram/"):
            shutil.rmtree(visualize_dir + "concat_histogram")
        dir_name = ''
        for index,path in enumerate(csv_paths):
            dir_name += os.path.splitext(os.path.basename(path))[0]
            if index != len(csv_paths)-1:
                dir_name += '&'
        # dir名からdirを生成
        required_dirs = [visualize_dir + "concat_histogram"] + [visualize_dir + "concat_histogram/" + dir_name]
        utils.make_dir(required_dirs)

        bi.concat_histogram_visualize(dir_name,csv_paths, output_path=visualize_dir)

    if args.pair:
        if os.path.isdir(visualize_dir + "pair_plot/"):
            shutil.rmtree(visualize_dir + "pair_plot")
        required_dirs = [visualize_dir + "pair_plot"]
        utils.make_dir(required_dirs)

        bi.pair_plot(csv_paths, output_path=visualize_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='グラフのデータを可視化するプログラム')
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--histogram',action='store_true')
    parser.add_argument('--scatter',action='store_true')
    parser.add_argument('--concat_histogram',action='store_true')
    parser.add_argument('--concat_scatter',action='store_true')
    parser.add_argument('--pair',action='store_true')
    parser.add_argument('--result', action='store')
    parser.add_argument('--type',nargs='*')

    parser.add_argument('--result_dir')

    args = parser.parse_args()
    
    main(args)
