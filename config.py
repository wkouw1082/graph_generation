# basic hyper parameters
epochs = 500
batch_size = 72
lr = 8e-3

decay=0
clip_th=4e-4

# model_param = {
#     "emb_size" : 100,
#     "en_hidden_size" : 50,
#     "de_hidden_size" : 256,
#     "rep_size" : 50
# }

model_param = {'batch_size': 122, 
               'lr': 0.009, 
               'weight_decay': 0.05, 
               'clip_th': 0.008380857476336032, 
               'emb_size': 253, 
               'en_hidden_size': 33, 
               'de_hidden_size': 249, 
               'rep_size': 58}

classifier_epochs=200
classifier_param = {
               'batch_size': 100, 
               'lr': 0.0006, 
               'emb_size': 256, 
               "hidden_size": 195,
               }

# graph_paremeter
data_dim = 25 # トポロジーの頂点数
# key: generate_topo(BA, fixed_BA, ...), value: [num, dim, [param]]
"""
train_generate_detail = {"BA": [500, data_dim, [None]],\
                         "NN": [500, data_dim, [0.6]]}
test_generate_detail = {"fixed_BA": [100, data_dim, [None]],\
                        "NN": [100, data_dim, [0.6]]}
"""
train_generate_detail = {"BA": [2000, data_dim, [None]]}
valid_generate_detail = {"BA": [200, data_dim, [None]]}

ignore_label = 1000

opt_epoch=100

# コンディショナルで指定する値
power_degree_label = [0.3,0.4,0.6]
power_degree_dim = len(power_degree_label)
cluster_coefficient_label = [0.2,0.3,0.4]
cluster_coefficient_dim = len(cluster_coefficient_label)

# 評価を行うパラメータら
# 現状、"power_degree", "cluster_coefficient", "distance", "size"
eval_params = ["power_degree", "cluster_coefficient", "distance", "size"]
