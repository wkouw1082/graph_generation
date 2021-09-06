import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dgl.dataloading import GraphDataLoader

import joblib

from config import *
import preprocess as pp

class LSTMDataset(Dataset):
    def __init__(self, data_type='train', transform=None, conditional=False):
        super(LSTMDataset, self).__init__()
        self.transform = transform

        self.data = joblib.load('./dataset/'+data_type+'/onehot')
        self.label = joblib.load('./dataset/'+data_type+'/label')
        self.datanum = len(self.data)

        if conditional:
            self.conditional = joblib.load('./dataset/'+data_type+'/conditional')
            self.conditional = torch.cat([self.conditional for _  in range(self.data.shape[1])],dim=1).unsqueeze(2)
            self.data = torch.cat((self.data,self.conditional),dim=2)

    def __len__(self):
        return self.datanum

    def __getitem__(self, index):
        out_data = self.data[index]
        out_label = self.label[index]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

class GCNDataset(Dataset):
    def __init__(self, data_type='train', transform=None, conditional=False):
        super(GCNDataset, self).__init__()
        self.transform = transform

        self.data = joblib.load('./dataset/'+data_type+'/onehot')
        self.label = joblib.load('./dataset/'+data_type+'/label')
        self.datanum = len(self.data)

        # TODO conditionの付与位置を2種類試す　1:ノードラベルとして与える 2:gcnでembeddingした後にラベルをconcatする
        if conditional:
            self.conditional = joblib.load('./dataset/'+data_type+'/conditional')
            self.conditional = torch.cat([self.conditional for _  in range(len(self.data[0].nodes()))],dim=1)
            for graph, condition in zip(self.data,self.conditional):
                node_feat = graph.ndata['feat'].unsqueeze(1)
                condition = condition.unsqueeze(1)
                node_feat = torch.cat([node_feat, condition],dim=1)
                graph.ndata['feat'] = node_feat

    def __len__(self):
        return self.datanum

    def __getitem__(self, index):
        out_data = self.data[index]
        out_label = self.label[index]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

if __name__=='__main__':
    test_dataset = GCNDataset('train', conditional=True)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=8, shuffle=True)
    print(len(test_dataset))
    for data, label in test_dataloader:
        print(data)
        print(label[:,0].shape)