# basic hyper parameters
epochs = 500
batch_size = 32
lr = 1e-2

decay=1e-4
clip_th=1e-2

model_param = {
    "emb_size" : 50,
    "en_hidden_size" : 64,
    "de_hidden_size" : 64,
    "rep_size" : 64
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
train_generate_detail = {"BA": [500, data_dim, [None]]}
test_generate_detail = {"BA": [100, data_dim, [None]]}
trait_dim = 2   # conditionalのベクトルの次元

ignore_label = 1000
