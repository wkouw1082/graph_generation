# basic hyper parameters
epochs = 300
batch_size = 32
#test_rate = 0.2 # testデータの割合
valid_rate = 0.3
emb_size = 128
en_hidden_size = 32
de_hidden_size = 32

# graph_paremeter
data_dim = 25 # トポロジーの頂点数
# key: generate_topo(BA, fixed_BA, ...), value: [num, dim, [param]]
train_generate_detail = {"BA": [500, data_dim, [None]],\
                         "NN": [500, data_dim, [0.6]]}
test_generate_detail = {"BA": [100, data_dim, [None]],\
                        "NN": [100, data_dim, [0.6]]}
trait_dim = 2   # conditionalのベクトルの次元
rep_size = 10
