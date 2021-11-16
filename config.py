# basic hyper parameters
epochs = 10000
dropout = 0.5
word_drop_rate=0

"""
model_param = {'batch_size': 30, 
               'lr': 0.001, 
               'weight_decay': 0.005, 
               'clip_th': 0.040380857476336032, 
               'emb_size': 116, 
               'en_hidden_size': 160, 
               'de_hidden_size': 225, 
               'rep_size': 171}
"""
# model_param={'batch_size': 69, 'lr': 0.0013549389234627585, 'weight_decay': 0.010660900921725731, 'clip_th': 0.014512071116018106, 'emb_size': 165, 'en_hidden_size': 203, 'de_hidden_size': 227, 'rep_size': 20}
# model_param = {'batch_size': 16, 'lr': 0.012039945025630484, 'weight_decay': 0.08034509665909136, 'clip_th': 0.015421355009867785, 'emb_size': 201, 'en_hidden_size': 102, 'de_hidden_size': 232, 'rep_size': 247}
# model_param = {'batch_size': 16, 'lr': 0.01082131764731544, 'weight_decay': 0.09174137908657536, 'clip_th': 0.0004241902009358986, 'emb_size': 189, 'en_hidden_size': 30, 'de_hidden_size': 247, 'rep_size': 220}
# tuneされていない場合に指定されるハイパーパラメータ
# model_params = {'batch_size': 32, 'clip_th': 0.03, 'emb_size': 150, 'en_hidden_size': 40, 'de_hidden_size': 230, 'rep_size': 175}
# 以前に頑張ってチューニングしたハイパーパラメータ、半年くらいかかるってやつ
model_params = {'batch_size': 21, 'clip_th': 0.00018132953639126497, 'emb_size': 233, 'en_hidden_size': 108, 'de_hidden_size': 252, 'rep_size': 65}

classifier_epochs=200
classifier_param={'batch_size': 26, 'lr': 0.00640815633081063, 'emb_size': 43, 'hidden_size': 92}

# graph_paremeter
data_dim = 25 # トポロジーの頂点数
classifier_bias=100
encoder_bias=3
# key: generate_topo(BA, fixed_BA, ...), value: [num, dim, [param]]
# train_generate_detail = {"BA": [500, data_dim, [None]],\
#                          "NN": [500, data_dim, [0.6]]}
# test_generate_detail = {"fixed_BA": [100, data_dim, [None]],\
#                         "NN": [100, data_dim, [0.6]]}
"""
train_generate_detail = {"BA": [2000, data_dim, [None]]}
valid_generate_detail = {"BA": [200, data_dim, [None]]}

train_generate_detail = {"NN": [2000, data_dim, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]}
valid_generate_detail = {"NN": [200, data_dim, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]}
"""

train_generate_detail = {"twitter_train":[None,None,[None]]}
valid_generate_detail = {"twitter_valid":[None,None,[None]]}

# データ分析時のdetail
visualize_detail = {
    "NN": [100, 100, [0.1,0.5,0.9]],
                    "twitter": [None,None,[None]],
                    "twitter_pickup": [None,None,[None]]}
# NNのparamのkeyがだぶるので名前だけ別で定義
visualize_types = {"NN_0.1":'./data/csv/NN_0.1.csv',"NN_0.5":'./data/csv/NN_0.5.csv',"NN_0.9":'./data/csv/NN_0.9.csv',"twitter":'./data/csv/twitter.csv',"Real_data":'./data/csv/Real_data.csv',\
    "graphgen":'./data/csv/GG_2000_300.csv',"Power-law_0.1":'./data/csv/generated_graph_0.csv',"Power-law_0.2":'./data/csv/generated_graph_1.csv',"Power-law_0.3":'./data/csv/generated_graph_2.csv'}

ignore_label=1500

opt_epoch=100

# コンディショナルで生成時に指定する値
power_degree_label = [0.4,0.5,0.6]
power_degree_dim = len(power_degree_label)
# クラスター係数 実際の値は[0.1~0.4]くらい
cluster_coefficient_label = [0.1,0.2,0.3]
modularity_label = [0.5, 0.6, 0.7]
cluster_coefficient_dim = len(cluster_coefficient_label)
# 最大最小距離　実際の値は[7~20]くらい
maximum_shortest_path_label = [7.0, 14.0, 20.0]

condition_values = {
    "Power-law exponent": [1.8, 2.0, 2.2], 
    "Clustering coefficient":[0.1,0.2,0.3], 
    "Average path length":[3,4,5], 
    "Average degree":[3,4,5], 
    "Edge density":[0.05,0.075,0.10], 
    "Modularity":[0.5, 0.6, 0.7], 
    "Diameter":[10,20,30], 
    "Largest component size":[7.0, 14.0, 20.0],
}
# conditionとして与えるparameter
condition_params = ["Power-law exponent", "Clustering coefficient"]
# conditionの値の丸める桁数
condition_round = 1

#　補完する値
interpolation_cluster_cofficient = [0.2, 0.3]
interpolation_maximum_path = [10.0, 16.0]

condition_size = 1

# 評価を行うパラメータら
# 現状、"power_degree", "cluster_coefficient", "distance", "size"
eval_params = ["Power-law exponent", "Clustering coefficient", "Average path length", "Average degree" ,"Edge density", "Modularity", "Diameter","Largest component size"]
# 評価を行うパラメータの外れ値を除くために値の上限を設定 指定は値を[最小値,最大値]の形で、指定なしはNoneで
# 仕様変更がめんどくさいので別で定義　いつか誰か一つにして欲しい
eval_params_limit = {"power_degree":None,"cluster_coefficient":[0,0.5],"distance":[0,8],\
                    "average_degree":[0,10],"density":[0,0.1],"modularity":[0,1],"maximum_distance":None,\
                    "degree_centrality":None,"betweenness_centrality":None,"closeness_centrality":None, "size":None}
# 評価に用いるネットワークのサイズの閾値
size_th=0
# 評価に用いるネットワーク数
graph_num=300
# 生成する最大dfsコード長
generate_max_len=2000

reddit_path = "./data/reddit_threads/reddit_edges.json"
twitter_path = "./data/edgelists_50/renum*"
twitter_train_path = './data/twitter_train'
twitter_valid_path = './data/twitter_eval'

# 次数分布の冪指数を出すときに大多数のデータに引っ張られるせいで１次元プロットが正しい値から離れてしまうのでいくつかの値を除いて導出するための除く割合
power_degree_border_line = 0.7

# config.pyを呼び出しているプログラムを実行した時間(と思われる)
from datetime import datetime
run_time = '{0:%Y%m%d_%H%M}'.format(datetime.now())

# gpuの情報用
DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

# datasetのノードサイズ（train_with_sequential_conditions）
dataset_node_size = 50      # dfs_code(model入力前)からグラフを再構築する時に使用
# cluster_coefficientを少数第何位で四捨五入するか
round_num = 4