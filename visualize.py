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

    if args.preprocess:
        # 危険なのでコメントアウト
        # すでにあるディレクトリを削除して再作成
        # if os.path.isdir("data/csv/"):
        #     shutil.rmtree("data/csv")
        # required_dirs = ["data/csv"]
        # utils.make_dir(required_dirs)

        # visualize_detailをもとにデータセットを作成
        cn = graph_process.complex_networks()
        dataset = cn.create_dataset(visualize_detail, do_type='visualize')
        # csvファイルに変換
        for name,data in zip(visualize_types.keys(),dataset):
            cn.graph2csv(data,name)

        bi.generate_result2csv()


    # if os.path.isdir("visualize") is False:
    #     utils.make_dir(["visualize"])

    # args type が何も指定されていない場合は全てのtypeが指定され、指定がある場合はそのtypeのcsv pathが持ってこられる
    csv_paths = [visualize_types[key] for key in args.type] if args.type is not None else utils.get_directory_paths('./data/csv/*')
    if args.scatter:
        if os.path.isdir("results/"+run_time+"/visualize/scatter_diagram/"):
            shutil.rmtree("results/"+run_time+"/visualize/scatter_diagram")
        # csvのファイルパスからdir名を持ってくる
        dir_names = [os.path.splitext(os.path.basename(csv_path))[0] for csv_path in csv_paths]
        # dir名からdirを生成
        required_dirs = ["results/"+run_time+"/visualize/scatter_diagram"] + ["results/"+run_time+"/visualize/scatter_diagram/" + dir_name for dir_name in dir_names]
        utils.make_dir(required_dirs)

        for path in csv_paths:
                bi.scatter_diagram_visualize(path)

    if args.histogram:
        if os.path.isdir("results/"+run_time+"/visualize/histogram/"):
            shutil.rmtree("results/"+run_time+"/visualize/histogram")
        # csvのファイルパスからdir名を持ってくる
        dir_names = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]
        # dir名からdirを生成
        required_dirs = ["results/"+run_time+"/visualize/histogram"] + ["results/"+run_time+"/visualize/histogram/" + dir_name for dir_name in dir_names]
        utils.make_dir(required_dirs)

        for path in csv_paths:
            bi.histogram_visualize(path)
        
    if args.concat_scatter:
        if os.path.isdir("results/"+run_time+"/visualize/concat_scatter_diagram/"):
            shutil.rmtree("results/"+run_time+"/visualize/concat_scatter_diagram")
        dir_name = ''
        for index,path in enumerate(csv_paths):
            dir_name += os.path.splitext(os.path.basename(path))[0]
            if index != len(csv_paths)-1:
                dir_name += '&'
        # dir名からdirを生成
        required_dirs = ["results/"+run_time+"/visualize/concat_scatter_diagram"] + ["results/"+run_time+"/visualize/concat_scatter_diagram/" + dir_name]
        utils.make_dir(required_dirs)
        
        bi.concat_scatter_diagram_visualize(dir_name,csv_paths)

    if args.concat_histogram:
        if os.path.isdir("results/"+run_time+"/visualize/concat_histogram/"):
            shutil.rmtree("results/"+run_time+"/visualize/concat_histogram")
        dir_name = ''
        for index,path in enumerate(csv_paths):
            dir_name += os.path.splitext(os.path.basename(path))[0]
            if index != len(csv_paths)-1:
                dir_name += '&'
        # dir名からdirを生成
        required_dirs = ["results/"+run_time+"/visualize/concat_histogram"] + ["results/"+run_time+"/visualize/concat_histogram/" + dir_name]
        utils.make_dir(required_dirs)

        bi.concat_histogram_visualize(dir_name,csv_paths)

    if args.pair:
        if os.path.isdir("results/"+run_time+"/visualize/pair_plot/"):
            shutil.rmtree("results/"+run_time+"/visualize/pair_plot")
        required_dirs = ["results/"+run_time+"/visualize/pair_plot"]
        utils.make_dir(required_dirs)

        bi.pair_plot(csv_paths)

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

    args = parser.parse_args()
    
    main(args)