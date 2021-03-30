import bi
import graph_process
import utils

import os
import argparse
import joblib
import shutil

from config import *

def main(parser):
    args = parser.parse_args()
    if args.preprocess:
        # すでにあるディレクトリを削除して再作成
        if os.path.isdir("data/csv/"):
            shutil.rmtree("data/csv")
        required_dirs = ["data/csv"]
        utils.make_dir(required_dirs)

        # visualize_detailをもとにデータセットを作成
        cn = graph_process.complex_networks()
        dataset = cn.create_dataset(visualize_detail)
        # csvファイルに変換
        for name,data in zip(visualize_names,dataset):
            cn.graph2csv(data,name)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='グラフのデータを可視化するプログラム')
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--histgram',action='store_true')
    parser.add_argument('--scatter',action='store_true')
    
    main(parser)
