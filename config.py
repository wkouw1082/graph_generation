# basic hyper parameters
epochs = 500
batch_size = 75
lr = 2e-2

decay=0.098
clip_th=5e-4

model_param = {
    "emb_size" : 100,
    "en_hidden_size" : 50,
    "de_hidden_size" : 256,
    "rep_size" : 50
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

# 評価を行うパラメータら
# 現状、"power_degree", "cluster_coefficient", "distance", "size"
eval_params = ["power_degree", "cluster_coefficient", "distance", "size"]
