import torch
from torch import nn
import torch.nn.functional as F
# import dgl
# from dgl.nn import SAGEConv
from graph_process import complex_networks
import numpy as np
import utils
from config import *
from utils import try_gpu
import random

class Classifier(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layer=1):
        super(Classifier, self).__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True)
        self.degree = nn.Linear(hidden_size, len(power_degree_label))
        self.cluster = nn.Linear(hidden_size, len(cluster_coefficient_label))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.emb(x)
        x, (h,c) = self.lstm(x)
        x = x[:, -1, :].unsqueeze(1)
        return self.softmax(self.degree(x)), self.softmax(self.cluster(x))

# 数種類用意して使うものだけコメントアウトを外す
# NormalなLSTM
# class Encoder(nn.Module):
#     def __init__(self, input_size, emb_size, hidden_size, rep_size, num_layer=1):
#         super(Encoder, self).__init__()
#         self.emb = nn.Linear(input_size, emb_size)
#         self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True)
#         self.mu = nn.Linear(hidden_size, rep_size)
#         self.sigma = nn.Linear(hidden_size, rep_size)

#     def forward(self, x):
#         x = self.emb(x)
#         # x = torch.cat((x,label),dim=2)
#         x, (h,c) = self.lstm(x)
#         x = x[:, -1, :].unsqueeze(1)
#         return self.mu(x), self.sigma(x)

# 2層 LSTM
class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, rep_size, num_layer=2):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True)
        self.mu = nn.Linear(hidden_size, rep_size)
        self.sigma = nn.Linear(hidden_size, rep_size)

    def forward(self, x):
        x = self.emb(x)
        # x = torch.cat((x,label),dim=2)
        x, (h,c) = self.lstm(x)
        x = x[:, -1, :].unsqueeze(1)
        return self.mu(x), self.sigma(x)

# bidirectional 1層 LSTM
# class Encoder(nn.Module):
#     def __init__(self, input_size, emb_size, hidden_size, rep_size, num_layer=1):
#         super(Encoder, self).__init__()
#         self.emb = nn.Linear(input_size, emb_size)
#         self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True, bidirectional=True)
#         self.mu = nn.Linear(hidden_size, rep_size)
#         self.sigma = nn.Linear(hidden_size, rep_size)

#     def forward(self, x):
#         x = self.emb(x)
#         # x = torch.cat((x,label),dim=2)
#         x, (h,c) = self.lstm(x)
#         # print(x.shape)
#         x = x[:, -1, :].unsqueeze(1)
#         # print("---")
#         # print(x.shape)
#         return self.mu(x), self.sigma(x)

# bidirectional 2層 LSTM
# class Encoder(nn.Module):
#     def __init__(self, input_size, emb_size, hidden_size, rep_size, num_layer=2):
#         super(Encoder, self).__init__()
#         self.emb = nn.Linear(input_size, emb_size)
#         self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True, bidirectional=True)
#         self.mu = nn.Linear(hidden_size, rep_size)
#         self.sigma = nn.Linear(hidden_size, rep_size)

#     def forward(self, x):
#         x = self.emb(x)
#         # x = torch.cat((x,label),dim=2)
#         x, (h,c) = self.lstm(x)
#         x = x[:, -1, :].unsqueeze(1)
#         return self.mu(x), self.sigma(x)

# GCNを使ったencoder
class GCNEncoder(nn.Module):
    def __init__(self, dfs_size, input_size, feat_size, hidden_size, rep_size):
        super(GCNEncoder, self).__init__()
        self.gcn1 = SAGEConv(input_size, feat_size, 'lstm')
        self.gcn2 = SAGEConv(feat_size, hidden_size, 'lstm')
        self.mu = nn.Linear(hidden_size, rep_size)
        self.sigma = nn.Linear(hidden_size, rep_size)

    def forward(self, g, in_feat):
        h = self.gcn1(g, in_feat)
        h = F.relu(h)
        h = self.gcn2(g, h)
        g.ndata['h'] = h
        x = dgl.mean_nodes(g, 'h')
        return self.mu(x), self.sigma(x)

class Decoder(nn.Module):
    def __init__(self, rep_size, input_size, emb_size, hidden_size, time_size, node_label_size, edge_label_size, device, num_layer=3):
        super(Decoder, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.emb = nn.Linear(input_size, emb_size)
        # onehot vectorではなく連続値なためサイズは+2
        self.f_rep = nn.Linear(rep_size+condition_size, input_size)
        self.lstm = nn.LSTM(emb_size+rep_size+condition_size, hidden_size, num_layers=self.num_layer, batch_first=True)
        self.f_tu = nn.Linear(hidden_size, time_size)
        self.f_tv = nn.Linear(hidden_size, time_size)
        self.f_lu = nn.Linear(hidden_size, node_label_size)
        self.f_lv = nn.Linear(hidden_size, node_label_size)
        self.f_le = nn.Linear(hidden_size, edge_label_size)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.5)

        self.f_h = nn.Linear(hidden_size, hidden_size)
        self.f_c = nn.Linear(hidden_size, hidden_size)

        self.time_size = time_size
        self.node_label_size = node_label_size
        self.edge_label_size = edge_label_size

        self.device = device

    def forward(self, rep, x, word_drop=0):
        """
        学習時のforward
        Args:
            rep: encoderの出力
            x: dfs code
        Returns:
            tu: source time
            tv: sink time
            lu: source node label
            lv: sink node label
            le: edge label
        """
        conditional =x[:,0,-1*condition_size:].unsqueeze(1)
        
        rep = torch.cat([rep, conditional], dim=2)

        origin_rep=rep
        rep = self.f_rep(rep)
        #rep = self.dropout(rep)

        x = torch.cat((rep, x), dim=1)[:,:-1,:]
        h_0 = try_gpu(self.device, torch.Tensor())
        c_0 = try_gpu(self.device, torch.Tensor())

        batch_size = x.shape[0]

        for batch in range(x.shape[0]):
            conditional_value = x[batch, 0, -1*condition_size].item()
            h_0 = torch.cat((h_0, try_gpu(self.device, torch.Tensor(self.num_layer, 1, self.hidden_size).fill_(conditional_value))), dim=1)
            c_0 = torch.cat((c_0, try_gpu(self.device, torch.Tensor(self.num_layer, 1, self.hidden_size).fill_(conditional_value))), dim=1)

        # word drop
        for batch in range(x.shape[0]):
            args=random.choices([i for i in range(x.shape[1])], k=int(x.shape[1]*word_drop))
            zero=utils.try_gpu(self.device,torch.zeros([1, 1, x.shape[2]-condition_size]))
            x[batch,args,:-1*condition_size]=zero
        x = self.emb(x)
        rep = torch.cat([origin_rep for _ in range(x.shape[1])],dim=1)
        x = torch.cat((x,rep),dim=2)

        # h_0 = try_gpu(self.device, torch.Tensor(self.num_layer, batch_size, self.hidden_size).fill_(conditional_value))
        # c_0 = try_gpu(self.device, torch.Tensor(self.num_layer, batch_size, self.hidden_size).fill_(conditional_value))

        x, (h, c) = self.lstm(x, (h_0, c_0))
        x = self.dropout(x)
        
        tu = self.softmax(self.f_tu(x))
        tv = self.softmax(self.f_tv(x))
        lu = self.softmax(self.f_lu(x))
        lv = self.softmax(self.f_lv(x))
        le = self.softmax(self.f_le(x))
        return tu, tv, lu, lv, le

    def generate(self, rep, conditional_label, max_size=100, is_output_sampling=True):
        """
        生成時のforward. 生成したdfsコードを用いて、新たなコードを生成していく
        Args:
            rep: encoderの出力
            max_size: 生成を続ける最大サイズ(生成を続けるエッジの最大数)
            is_output_sampling: Trueなら返り値を予測dfsコードからargmaxしたものに. Falseなら予測分布を返す
        Returns:
        """
        conditional_value = conditional_label.item()
        conditional_label = conditional_label.unsqueeze(0).unsqueeze(1)
        conditional_label = torch.cat([conditional_label for _ in range(rep.shape[0])], dim=0)
        conditional_label = utils.try_gpu(self.device,conditional_label)

        rep = torch.cat([rep, conditional_label], dim=2)

        origin_rep=rep

        rep = self.f_rep(rep)
        rep = self.emb(rep)
        x = rep
        x = torch.cat((x,origin_rep),dim=2)
        batch_size = x.shape[0]

        tus = torch.LongTensor()
        tus = try_gpu(self.device,tus)
        tvs = torch.LongTensor()
        tvs = try_gpu(self.device,tvs)
        lus = torch.LongTensor()
        lus = try_gpu(self.device,lus)
        lvs = torch.LongTensor()
        lvs = try_gpu(self.device,lvs)
        les = torch.LongTensor()
        les = try_gpu(self.device,les)

        tus_dist=try_gpu(self.device,torch.Tensor())
        tvs_dist=try_gpu(self.device,torch.Tensor())
        lus_dist=try_gpu(self.device,torch.Tensor())
        lvs_dist=try_gpu(self.device,torch.Tensor())
        les_dist=try_gpu(self.device,torch.Tensor())

        h_0 = try_gpu(self.device, torch.Tensor(self.num_layer, batch_size, self.hidden_size).fill_(conditional_value))
        c_0 = try_gpu(self.device, torch.Tensor(self.num_layer, batch_size, self.hidden_size).fill_(conditional_value))

        for i in range(max_size):
            if i == 0:
                x, (h, c) = self.lstm(x, (h_0, c_0))
                # x, (h, c) = self.lstm(x)
            else:
                x = self.emb(x)
                x = torch.cat((x,origin_rep),dim=2)
                x, (h, c) = self.lstm(x, (h, c))

            tus_dist = torch.cat([tus_dist,self.softmax(self.f_tu(x))], dim=1)
            tvs_dist = torch.cat([tvs_dist,self.softmax(self.f_tv(x))], dim=1)
            lus_dist = torch.cat([lus_dist,self.softmax(self.f_lu(x))], dim=1)
            lvs_dist = torch.cat([lvs_dist,self.softmax(self.f_lv(x))], dim=1)
            les_dist = torch.cat([les_dist,self.softmax(self.f_le(x))], dim=1)

            tu = torch.argmax(self.softmax(self.f_tu(x)), dim=2)
            tv = torch.argmax(self.softmax(self.f_tv(x)), dim=2)
            lu = torch.argmax(self.softmax(self.f_lu(x)), dim=2)
            lv = torch.argmax(self.softmax(self.f_lv(x)), dim=2)
            le = torch.argmax(self.softmax(self.f_le(x)), dim=2)

            tus = torch.cat((tus, tu), dim=1)
            tvs = torch.cat((tvs, tv), dim=1)
            lus = torch.cat((lus, lu), dim=1)
            lvs = torch.cat((lvs, lv), dim=1)
            les = torch.cat((les, le), dim=1)
            tu = tu.squeeze().cpu().detach().numpy()
            tv = tv.squeeze().cpu().detach().numpy()
            lu = lu.squeeze().cpu().detach().numpy()
            lv = lv.squeeze().cpu().detach().numpy()
            le = le.squeeze().cpu().detach().numpy()

            tu = utils.convert2onehot(tu, self.time_size)
            tv = utils.convert2onehot(tv, self.time_size)
            lu = utils.convert2onehot(lu, self.node_label_size)
            lv = utils.convert2onehot(lv, self.node_label_size)
            le = utils.convert2onehot(le, self.edge_label_size)
            x = torch.cat((tu, tv, lu, lv, le), dim=1).unsqueeze(1)
            x = try_gpu(self.device,x)
    
            x = torch.cat((x, conditional_label),dim=2)
        if is_output_sampling:
            return tus, tvs, lus, lvs, les
        else:
            return tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist

class DecoderNonConditional(nn.Module):
    def __init__(self, rep_size, input_size, emb_size, hidden_size, time_size, node_label_size, edge_label_size, device, num_layer=1):
        super(DecoderNonConditional, self).__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.f_rep = nn.Linear(rep_size, input_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True)
        self.f_tu = nn.Linear(hidden_size, time_size)
        self.f_tv = nn.Linear(hidden_size, time_size)
        self.f_lu = nn.Linear(hidden_size, node_label_size)
        self.f_lv = nn.Linear(hidden_size, node_label_size)
        self.f_le = nn.Linear(hidden_size, edge_label_size)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

        self.time_size = time_size
        self.node_label_size = node_label_size
        self.edge_label_size = edge_label_size

        self.device = device

    def forward(self, z, x):
        """
        学習時のforward
        Args:
            z: 潜在変数からのベクトル
            x: 入力のdfs code
        Returns:
            tu: source time
            tv: sink time
            lu: source node label
            lv: sink node label
            le: edge label
        """
        # 潜在変数をLSTMにcatできる形にするために線形層をかます
        z = self.f_rep(z)
        # ここでsequenceデータの頭に潜在変数をつける
        x = torch.cat((z, x), dim=1)
        # つけた分sequenceデータを短くする
        x = x[:,:-1,:]
        # shapeはbatch x sequence_length x 1step ごとのdfs 本来なら5(各要素)だがonehotになっているので
        # onehotlength をそれぞれ足したものになっている
        x = self.emb(x)
        x, (h, c) = self.lstm(x)
        # x = self.dropout(x)
        tu = self.f_tu(x)
        tv = self.f_tv(x)
        lu = self.f_lu(x)
        lv = self.f_lv(x)
        le = self.f_le(x)
        # return torch.cat((tu, tv, lu, lv, le), dim=2)
        return tu, tv, lu, lv, le, torch.cat((tu, tv, lu, lv, le), dim=2)

    def generate(self, rep, max_size=100, is_output_sampling=True):
        """
        生成時のforward. 生成したdfsコードを用いて、新たなコードを生成していく
        Args:
            rep: encoderの出力
            max_size: 生成を続ける最大サイズ(生成を続けるエッジの最大数)
            is_output_sampling: Trueなら返り値を予測dfsコードからargmaxしたものに. Falseなら予測分布を返す
        Returns:
        """

        rep = self.f_rep(rep)
        rep = self.emb(rep)
        x = rep

        tus = try_gpu(self.device, torch.LongTensor())
        tvs = try_gpu(self.device, torch.LongTensor())
        lus = try_gpu(self.device, torch.LongTensor())
        lvs = try_gpu(self.device, torch.LongTensor())
        les = try_gpu(self.device, torch.LongTensor())

        tus_dist=try_gpu(self.device,torch.Tensor())
        tvs_dist=try_gpu(self.device,torch.Tensor())
        lus_dist=try_gpu(self.device,torch.Tensor())
        lvs_dist=try_gpu(self.device,torch.Tensor())
        les_dist=try_gpu(self.device,torch.Tensor())

        for i in range(max_size):
            if i == 0:
                x, (h, c) = self.lstm(x)
            else:
                # x = self.emb(x)
                x, (h, c) = self.lstm(x, (h, c))

            tus_dist = torch.cat([tus_dist,self.f_tu(x)], dim=1)
            tvs_dist = torch.cat([tvs_dist,self.f_tv(x)], dim=1)
            lus_dist = torch.cat([lus_dist,self.f_lu(x)], dim=1)
            lvs_dist = torch.cat([lvs_dist,self.f_lv(x)], dim=1)
            les_dist = torch.cat([les_dist,self.f_le(x)], dim=1)

            tu = self.f_tu(x)
            tv = self.f_tv(x)
            lu = self.f_lu(x)
            lv = self.f_lv(x)
            le = self.f_le(x)

            tus = torch.cat((tus, tu), dim=1)
            tvs = torch.cat((tvs, tv), dim=1)
            lus = torch.cat((lus, lu), dim=1)
            lvs = torch.cat((lvs, lv), dim=1)
            les = torch.cat((les, le), dim=1)
            # tu = tu.squeeze().cpu().detach().numpy()
            # tv = tv.squeeze().cpu().detach().numpy()
            # lu = lu.squeeze().cpu().detach().numpy()
            # lv = lv.squeeze().cpu().detach().numpy()
            # le = le.squeeze().cpu().detach().numpy()

            # tu = utils.convert2onehot(tu, self.time_size)
            # tv = utils.convert2onehot(tv, self.time_size)
            # lu = utils.convert2onehot(lu, self.node_label_size)
            # lv = utils.convert2onehot(lv, self.node_label_size)
            # le = utils.convert2onehot(le, self.edge_label_size)
            # x = torch.cat((tu, tv, lu, lv, le), dim=1).unsqueeze(1)
            x = torch.cat((tu, tv, lu, lv, le), dim=2)
            x = try_gpu(self.device,x)

        if is_output_sampling:
            return tus, tvs, lus, lvs, les
        else:
            return tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist

class GCNDecoder(nn.Module):
    def __init__(self, rep_size, input_size, emb_size, hidden_size, time_size, node_label_size, edge_label_size, device, num_layer=1):
        super(GCNDecoder, self).__init__()
        self.lstm = nn.LSTM(rep_size, hidden_size, num_layers=num_layer, batch_first=True)
        self.f_tu = nn.Linear(hidden_size, time_size)
        self.f_tv = nn.Linear(hidden_size, time_size)
        self.f_lu = nn.Linear(hidden_size, node_label_size)
        self.f_lv = nn.Linear(hidden_size, node_label_size)
        self.f_le = nn.Linear(hidden_size, edge_label_size)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.5)

        self.time_size = time_size
        self.node_label_size = node_label_size
        self.edge_label_size = edge_label_size

        self.device = device
        self.input_size = rep_size

    def forward(self, z):
        """
        学習時のforward
        Args:
            z: 潜在空間からのサンプリング
        Returns:
            tu: source time
            tv: sink time
            lu: source node label
            lv: sink node label
            le: edge label
        """

        z = z.unsqueeze(0)

        x, (h, c) = self.lstm(z)
        x = self.dropout(x)
        tu = self.softmax(self.f_tu(x))
        tv = self.softmax(self.f_tv(x))
        lu = self.softmax(self.f_lu(x))
        lv = self.softmax(self.f_lv(x))
        le = self.softmax(self.f_le(x))
        return tu, tv, lu, lv, le

    def generate(self, rep, conditional_label, max_size=100, is_output_sampling=True):
        """
        生成時のforward. 生成したdfsコードを用いて、新たなコードを生成していく
        Args:
            rep: encoderの出力
            max_size: 生成を続ける最大サイズ(生成を続けるエッジの最大数)
            is_output_sampling: Trueなら返り値を予測dfsコードからargmaxしたものに. Falseなら予測分布を返す
        Returns:
        """
        conditional_label = conditional_label.unsqueeze(0).unsqueeze(1)
        conditional_label = torch.cat([conditional_label for _ in range(rep.shape[0])], dim=0)
        conditional_label = utils.try_gpu(self.device,conditional_label)

        rep = torch.cat([rep, conditional_label], dim=2)

        origin_rep=rep

        rep = self.f_rep(rep)
        rep = self.emb(rep)
        x = rep
        x = torch.cat((x,origin_rep),dim=2)
        batch_size = x.shape[0]

        tus = torch.LongTensor()
        tus = try_gpu(self.device,tus)
        tvs = torch.LongTensor()
        tvs = try_gpu(self.device,tvs)
        lus = torch.LongTensor()
        lus = try_gpu(self.device,lus)
        lvs = torch.LongTensor()
        lvs = try_gpu(self.device,lvs)
        les = torch.LongTensor()
        les = try_gpu(self.device,les)

        tus_dist=try_gpu(self.device,torch.Tensor())
        tvs_dist=try_gpu(self.device,torch.Tensor())
        lus_dist=try_gpu(self.device,torch.Tensor())
        lvs_dist=try_gpu(self.device,torch.Tensor())
        les_dist=try_gpu(self.device,torch.Tensor())

        for i in range(max_size):
            if i == 0:
                x, (h, c) = self.lstm(x)
            else:
                x = self.emb(x)
                x = torch.cat((x,origin_rep),dim=2)
                x, (h, c) = self.lstm(x, (h, c))

            tus_dist = torch.cat([tus_dist,self.softmax(self.f_tu(x))], dim=1)
            tvs_dist = torch.cat([tvs_dist,self.softmax(self.f_tv(x))], dim=1)
            lus_dist = torch.cat([lus_dist,self.softmax(self.f_lu(x))], dim=1)
            lvs_dist = torch.cat([lvs_dist,self.softmax(self.f_lv(x))], dim=1)
            les_dist = torch.cat([les_dist,self.softmax(self.f_le(x))], dim=1)

            tu = torch.argmax(self.softmax(self.f_tu(x)), dim=2)
            tv = torch.argmax(self.softmax(self.f_tv(x)), dim=2)
            lu = torch.argmax(self.softmax(self.f_lu(x)), dim=2)
            lv = torch.argmax(self.softmax(self.f_lv(x)), dim=2)
            le = torch.argmax(self.softmax(self.f_le(x)), dim=2)

            tus = torch.cat((tus, tu), dim=1)
            tvs = torch.cat((tvs, tv), dim=1)
            lus = torch.cat((lus, lu), dim=1)
            lvs = torch.cat((lvs, lv), dim=1)
            les = torch.cat((les, le), dim=1)
            tu = tu.squeeze().cpu().detach().numpy()
            tv = tv.squeeze().cpu().detach().numpy()
            lu = lu.squeeze().cpu().detach().numpy()
            lv = lv.squeeze().cpu().detach().numpy()
            le = le.squeeze().cpu().detach().numpy()

            tu = utils.convert2onehot(tu, self.time_size)
            tv = utils.convert2onehot(tv, self.time_size)
            lu = utils.convert2onehot(lu, self.node_label_size)
            lv = utils.convert2onehot(lv, self.node_label_size)
            le = utils.convert2onehot(le, self.edge_label_size)
            x = torch.cat((tu, tv, lu, lv, le), dim=1).unsqueeze(1)
            x = try_gpu(self.device,x)
    
            x = torch.cat((x, conditional_label),dim=2)
        if is_output_sampling:
            return tus, tvs, lus, lvs, les
        else:
            return tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist

class VAE(nn.Module):
    def __init__(self, dfs_size, time_size, node_size, edge_size, model_param, device):
        super(VAE, self).__init__()
        emb_size = model_param["emb_size"]
        en_hidden_size = model_param["en_hidden_size"]
        de_hidden_size = model_param["de_hidden_size"]
        rep_size = model_param["rep_size"]
        self.rep_size = rep_size
        self.device = device
        self.encoder = Encoder(dfs_size, emb_size, en_hidden_size, rep_size)
        self.decoder = Decoder(rep_size, dfs_size, emb_size, de_hidden_size, time_size, node_size, edge_size, self.device)

    def noise_generator(self, rep_size, batch_num):
        return torch.randn(batch_num, rep_size)

    def forward(self, x, word_drop=0):
        mu, sigma = self.encoder(x)
        z = transformation(mu, sigma, self.device)
        tu, tv, lu, lv, le = self.decoder(z, x)
        return mu, sigma, tu, tv, lu, lv, le

    def generate(self, data_num, conditional_label, z=None, max_size=generate_max_len, is_output_sampling=True):
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = utils.try_gpu(self.device,z)
        tu, tv, lu, lv, le =\
            self.decoder.generate(z, conditional_label, max_size, is_output_sampling)
        return tu, tv, lu, lv, le

    def encode(self, x):
        mu, sigma = self.encoder(x)
        z = transformation(mu, sigma, self.device)
        return z

class VAENonConditional(nn.Module):
    def __init__(self, dfs_size, time_size, node_size, edge_size, model_param,device):
        super(VAENonConditional, self).__init__()
        emb_size = model_param["emb_size"]
        en_hidden_size = model_param["en_hidden_size"]
        de_hidden_size = model_param["de_hidden_size"]
        rep_size = model_param["rep_size"]
        self.rep_size = rep_size
        self.device = device
        self.encoder = Encoder(dfs_size, emb_size, en_hidden_size, rep_size)
        self.decoder = DecoderNonConditional(rep_size, dfs_size, emb_size, de_hidden_size, time_size, node_size, edge_size, self.device)

    def noise_generator(self, rep_size, batch_num):
        return torch.randn(batch_num, rep_size)

    def forward(self, x, word_drop=0):
        mu, sigma = self.encoder(x)
        z = transformation(mu, sigma, self.device)
        tu, tv, lu, lv, le, outputs = self.decoder(z, x)
        return mu, sigma, outputs, tu, tv, lu, lv, le

    def generate(self, data_num, z=None, max_size=generate_max_len, is_output_sampling=True):
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = utils.try_gpu(self.device,z)
        tu, tv, lu, lv, le =\
            self.decoder.generate(z, max_size, is_output_sampling)
        return tu, tv, lu, lv, le

    def encode(self, x):
        mu, sigma = self.encoder(x)
        z = transformation(mu, sigma, self.device)
        return z

class GCNVAE(nn.Module):
    def __init__(self, dfs_size, time_size, node_size, edge_size, model_param, device):
        super(GCNVAE, self).__init__()
        emb_size = model_param["emb_size"]
        en_hidden_size = model_param["en_hidden_size"]
        de_hidden_size = model_param["de_hidden_size"]
        rep_size = model_param["rep_size"]
        self.rep_size = rep_size
        self.device = device
        self.encoder = GCNEncoder(dfs_size, 2, en_hidden_size, 100, rep_size)
        self.decoder = GCNDecoder(rep_size, dfs_size, emb_size, de_hidden_size, time_size, node_size, edge_size, self.device)

    def noise_generator(self, rep_size, batch_num):
        return torch.randn(batch_num, rep_size)

    def forward(self, x, feat, word_drop=0):
        mu, sigma = self.encoder(x, feat)
        z = transformation(mu, sigma, self.device)
        tu, tv, lu, lv, le = self.decoder(z)
        return mu, sigma, tu, tv, lu, lv, le

    def generate(self, data_num, conditional_label, z=None, max_size=generate_max_len, is_output_sampling=True):
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = utils.try_gpu(self.device,z)
        tu, tv, lu, lv, le =\
            self.decoder.generate(z, conditional_label, max_size, is_output_sampling)
        return tu, tv, lu, lv, le

    def encode(self, x, feat):
        mu, sigma = self.encoder(x, feat)
        z = transformation(mu, sigma, self.device)
        return z

def transformation(mu, sigma,device):
    return mu + torch.exp(0.5*sigma) * utils.try_gpu(device,torch.randn(sigma.shape))

if __name__=="__main__":
    import numpy as np
    x = torch.randn(batch_size,200,10)
    y = torch.randn(batch_size,200,10)
    vae = VAE(10,2,2,2)
    vae(x,y)
    vae.generate()
    """
    encoder = Encoder(20, 10, 10, 10)
    mu, sigma = encoder(x)
    z = transformation(mu, sigma)
    decoder = Decoder(10, 20, 10, 10, 2, 2, 2)
    tu, tv, lu, lv, le = decoder(z, y)
    tu, tv, lu, lv, le = decoder.generate(z, 10)
    """
