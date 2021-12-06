import argparse
import yaml
import os
import random

import utils
import tune
import train
import eval
import visualize
import config
import graph_process
import bi
from config import *

def main(args):
    """実行したいプログラムを引数で指定して、実行する関数
       現在、全部実行することが推奨

    Args:
        args        (argparse.ArgumentParser().parse_args()): 実行するプログラムとpreprocessなどのプロパティ
    """

    device = 'cuda:0'
    print('using device is {}.'.format(device))
    print('---------------------')
    print('tuning parameter is {}'.format(condition_params))

    # preprocess が必要か、既に終了したかを意味するフラグを作成
    if args.preprocess is False:
        # preprocessは不要
        is_finished_preprocess = True
    else:
        # preprocessが必要
        is_finished_preprocess = False

    # 大まかなディレクトリ作成. より細かいディレクトリは各関数で作成
    ## 必須ディレクトリの作成
    required_dirs = ["dataset", "param", "results"]
    for dir in required_dirs:
        if os.path.exists("./" + dir):
            required_dirs.remove(dir)
    if len(required_dirs) > 0:
        utils.make_dir(required_dirs)
    ## 指定したresult_dir の存在を確認
    if args.result_dir:
        if not os.path.exists("./" + args.result_dir):
            print(f"{args.result_dir} が存在しません.")
            exit()
    

    # tune
    if args.tune:
        if args.condition:
            tune.conditional_tune(args)
        else:
            tune.tune(args)
        if is_finished_preprocess is False:
            is_finished_preprocess = True
            args.preprocess = False

    # train
    if args.train:
        if args.condition:
            train.conditional_train(args, device)
        elif args.seq_condition:
            train.train_with_sequential_conditions(args)
        else:
            train.train(args)
        if is_finished_preprocess is False:
            is_finished_preprocess = True
            args.preprocess = False
    
    # eval以降のpreprocessは仕様が違うのでtrueにする必要がある
    args.preprocess = True

    # eval
    if args.eval:
        if args.condition:
            eval.eval(args, device)
        else:
            eval.non_conditional_eval(args)

    # visualize
    if args.visualize:
        visualize.main(args)
        # ここのpreprocessは動作は他のpreprocessとは動作が違うので常に行うこと

    # if args.graph_visualize:
    #     text_datas = utils.get_directory_paths(twitter_path)
    #     graph_datas = graph_process.text2graph(text_datas)
    #     bi.graph_visualize(random.choice(graph_datas))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='選択実行プログラム')
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--classifier',action='store_true')
    parser.add_argument('--condition', action='store_true')
    parser.add_argument('--seq_condition', action='store_true')
    parser.add_argument('--tune',      action='store_true')
    parser.add_argument('--train',     action='store_true')
    parser.add_argument('--eval',      action='store_true')
    # 生成したグラフのパラメータごとの可視化
    parser.add_argument('--visualize', action='store_true')
    

    parser.add_argument('--model_param',action='store', help="ハイパーパラメータのパス")
    parser.add_argument('--eval_model',action='store', help="eval時に読み込むweightのパス")
    parser.add_argument('--result_dir', action='store', help="results/ 直下に存在するディレクトリのパス")

    parser.add_argument('--histogram', action='store_true')
    parser.add_argument('--scatter',   action='store_true')
    parser.add_argument('--concat_histogram',action='store_true')
    parser.add_argument('--concat_scatter',action='store_true')
    parser.add_argument('--pair',      action='store_true')
    parser.add_argument('--type',      nargs='*')
    parser.add_argument('--graph_struct', action='store_true')

    args = parser.parse_args()

    main(args)
