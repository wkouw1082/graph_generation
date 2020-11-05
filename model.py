import torch
from torch import empty, nn
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

class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, rep_size, num_layer=1):
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

class Decoder(nn.Module):
    def __init__(self, rep_size, input_size, emb_size, hidden_size, time_size, node_label_size, edge_label_size, num_layer=1):
        super(Decoder, self).__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.f_rep = nn.Linear(rep_size+6, input_size)
        #self.f_rep = nn.Linear(rep_size, input_size)
        self.lstm = nn.LSTM(emb_size+rep_size+6, hidden_size, num_layers=num_layer, batch_first=True)
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

        conditional=x[:,0,-6:].unsqueeze(1)
        rep = torch.cat([rep, conditional], dim=2)

        origin_rep=rep
        rep = self.f_rep(rep)
        #rep = self.dropout(rep)

        x = torch.cat((rep, x), dim=1)[:,:-1,:]

        # word drop
        for batch in range(x.shape[0]):
            args=random.choices([i for i in range(x.shape[1])], k=int(x.shape[1]*word_drop))
            zero=utils.try_gpu(torch.zeros([1, 1, x.shape[2]-6]))
            x[batch,args,:-6]=zero

        x = self.emb(x)
        rep = torch.cat([origin_rep for _ in range(x.shape[1])],dim=1)
        x = torch.cat((x,rep),dim=2)

        x, (h, c) = self.lstm(x)
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
        conditional_label = utils.try_gpu(conditional_label)

        rep = torch.cat([rep, conditional_label], dim=2)

        origin_rep=rep

        rep = self.f_rep(rep)
        rep = self.emb(rep)
        x = rep
        x = torch.cat((x,origin_rep),dim=2)
        batch_size = x.shape[0]

        tus = torch.LongTensor()
        tus = try_gpu(tus)
        tvs = torch.LongTensor()
        tvs = try_gpu(tvs)
        lus = torch.LongTensor()
        lus = try_gpu(lus)
        lvs = torch.LongTensor()
        lvs = try_gpu(lvs)
        les = torch.LongTensor()
        les = try_gpu(les)

        tus_dist=try_gpu(torch.Tensor())
        tvs_dist=try_gpu(torch.Tensor())
        lus_dist=try_gpu(torch.Tensor())
        lvs_dist=try_gpu(torch.Tensor())
        les_dist=try_gpu(torch.Tensor())

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
            x = try_gpu(x)
    
            x = torch.cat((x, conditional_label),dim=2)
        if is_output_sampling:
            return tus, tvs, lus, lvs, les
        else:
            return tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist

class VAE(nn.Module):
    def __init__(self, dfs_size, time_size, node_size, edge_size, model_param):
        super(VAE, self).__init__()
        emb_size = model_param["emb_size"]
        en_hidden_size = model_param["en_hidden_size"]
        de_hidden_size = model_param["de_hidden_size"]
        rep_size = model_param["rep_size"]
        self.rep_size = rep_size
        self.encoder = Encoder(dfs_size, emb_size, en_hidden_size, rep_size)
        self.decoder = Decoder(rep_size, dfs_size, emb_size, de_hidden_size, time_size, node_size, edge_size)

    def noise_generator(self, rep_size, batch_num):
        return torch.randn(batch_num, rep_size)

    def forward(self, x, word_drop=0):
        mu, sigma = self.encoder(x)
        z = transformation(mu, sigma)
        tu, tv, lu, lv, le = self.decoder(z, x)
        return mu, sigma, tu, tv, lu, lv, le

    def generate(self, data_num, conditional_label, z=None, max_size=generate_max_len, is_output_sampling=True):
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = utils.try_gpu(z)
        tu, tv, lu, lv, le =\
            self.decoder.generate(z, conditional_label, max_size, is_output_sampling)
        return tu, tv, lu, lv, le

    def encode(self, x):
        mu, sigma = self.encoder(x)
        z = transformation(mu, sigma)
        return z

def transformation(mu, sigma):
    return mu + torch.exp(0.5*sigma) * utils.try_gpu(torch.randn(sigma.shape))

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
