import argparse
import yaml
import os

import utils
import tune
import train
import eval
import visualize
from config import *

def main(args):
    """実行したいプログラムを引数で指定して、実行する関数
       現在、全部実行することが推奨

    Args:
        args        (argparse.ArgumentParser().parse_args()): 実行するプログラムとpreprocessなどのプロパティ
    """
    # preprocee が必要か、既に終了したかを意味するフラグを作成
    if args.preprocess is False:
        # preprocessは不要
        is_finished_preprocess = True
    else:
        # preprocessが必要
        is_finished_preprocess = False

    # 大まかなディレクトリ作成. より細かいディレクトリは各関数で作成
    required_dirs = ["dataset", "param", "results/"+run_time, "results/"+run_time+"/train", "results/"+run_time+"/eval", "results/"+run_time+"/visualize"]
    if not os.path.exists("./results"):
        required_dirs.remove("results")
    #utils.make_dir(required_dirs)

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
            train.conditional_train(args)
        else:
            train.train(args)
        if is_finished_preprocess is False:
            is_finished_preprocess = True
            args.preprocess = False
    
    # eval
    if args.eval:
        if args.condition:
            eval.eval(args)
        else:
            eval.non_conditional_eval(args)
        if is_finished_preprocess is False:
            is_finished_preprocess = True
            args.preprocess = False

    # visualize
    if args.visualize:
        visualize.main(args)
        if is_finished_preprocess is False:
            is_finished_preprocess = True
            args.preprocess = False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='選択実行プログラム')
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--classifier',action='store_true')
    parser.add_argument('--condition', action='store_true')
    parser.add_argument('--tune',      action='store_true')
    parser.add_argument('--train',     action='store_true')
    parser.add_argument('--eval',      action='store_true')
    parser.add_argument('--eval_model',action='store')
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--histogram', action='store_true')
    parser.add_argument('--scatter',   action='store_true')
    parser.add_argument('--concat_histogram',action='store_true')
    parser.add_argument('--concat_scatter',action='store_true')
    parser.add_argument('--pair',      action='store_true')
    parser.add_argument('--type',      nargs='*')

    args = parser.parse_args()

    main(args)
