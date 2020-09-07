import torch
from torch import nn
import utils
from config import *

class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, rep_size, num_layer=1):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True)
        self.mu = nn.Linear(hidden_size, rep_size)
        self.sigma = nn.Linear(hidden_size, rep_size)

    def forward(self, x):
        x = self.emb(x)
        x, (h,c) = self.lstm(x)
        x = x[:, -1, :].unsqueeze(1)
        return self.mu(x), self.sigma(x)

class Decoder(nn.Module):
    def __init__(self, rep_size, input_size, emb_size, hidden_size, time_size, node_label_size, edge_label_size, num_layer=1):
        super(Decoder, self).__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.f_rep = nn.Linear(rep_size, input_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True)
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

    def forward(self, rep, x):
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
        rep = self.f_rep(rep)
        x = torch.cat((rep, x), dim=1)[:,:-1,:]
        x = self.emb(x)
        x, (h, c) = self.lstm(x)
        #x = self.dropout(x)
        tu = self.softmax(self.f_tu(x))
        tv = self.softmax(self.f_tv(x))
        lu = self.softmax(self.f_lu(x))
        lv = self.softmax(self.f_lv(x))
        le = self.softmax(self.f_le(x))
        return tu, tv, lu, lv, le

    def generate(self, rep, max_size=100):
        """
        生成時のforward. 生成したdfsコードを用いて、新たなコードを生成していく
        Args:
            rep: encoderの出力
            max_size: 生成を続ける最大サイズ(生成を続けるエッジの最大数)
        Returns:
        """
        rep = self.f_rep(rep)
        rep = self.emb(rep)
        x = rep
        batch_size = x.shape[0]
        h = torch.zeros(1, batch_size, x.shape[2])
        c = torch.zeros(1, batch_size, x.shape[2])

        tus = torch.LongTensor()
        tvs = torch.LongTensor()
        lus = torch.LongTensor()
        lvs = torch.LongTensor()
        les = torch.LongTensor()

        for i in range(max_size):
            if i == 0:
                x, (h, c) = self.lstm(x)
            else:
                x = self.emb(x)
                x, (h, c) = self.lstm(x, (h, c))
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
            tu = tu.squeeze().detach().numpy()
            tv = tv.squeeze().detach().numpy()
            lu = lu.squeeze().detach().numpy()
            lv = lv.squeeze().detach().numpy()
            le = le.squeeze().detach().numpy()

            tu = utils.convert2onehot(tu, self.time_size)
            tv = utils.convert2onehot(tv, self.time_size)
            lu = utils.convert2onehot(lu, self.time_size)
            lv = utils.convert2onehot(lv, self.time_size)
            le = utils.convert2onehot(le, self.time_size)
            x = torch.cat((tu, tv, lu, lv, le), dim=1).unsqueeze(1)
        return tus, tvs, lus, lvs, les

class VAE(nn.Module):
    def __init__(self, dfs_size, time_size, node_size, edge_size, model_param):
        super(VAE, self).__init__()
        emb_size = model_param["emb_size"]
        en_hidden_size = model_param["en_hidden_size"]
        de_hidden_size = model_param["de_hidden_size"]
        rep_size = model_param["rep_size"]
        self.encoder = Encoder(dfs_size, emb_size, en_hidden_size, rep_size)
        self.decoder = Decoder(rep_size, dfs_size, emb_size, de_hidden_size, time_size, node_size, edge_size)

    def noise_generator(self, batch_num = batch_size):
        return torch.randn(batch_num, rep_size)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = transformation(mu, sigma)
        tu, tv, lu, lv, le = self.decoder(z, x)
        return mu, sigma, tu, tv, lu, lv, le

    def generate(self, max_size=100):
        z = self.noise_generator().unsqueeze(1)
        tu, tv, lu, lv, le =\
            self.decoder.generate(z, 100)
        return tu, tv, lu, lv, le

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
