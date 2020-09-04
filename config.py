# basic hyper parameters
epochs = 300
batch_size = 2
lr = 3e-2
#test_rate = 0.2 # testデータの割合
valid_rate = 0.3
emb_size = 8
en_hidden_size = 4
de_hidden_size = 4

# graph_paremeter
data_dim = 25 # トポロジーの頂点数
# key: generate_topo(BA, fixed_BA, ...), value: [num, dim, [param]]
train_generate_detail = {"BA": [10, data_dim, [None]],\
                         "NN": [10, data_dim, [0.6]]}
test_generate_detail = {"BA": [4, data_dim, [None]],\
                        "NN": [4, data_dim, [0.6]]}
trait_dim = 2   # conditionalのベクトルの次元
rep_size = 4

ignore_label = 1000
