import optuna
import yaml
import shutil
import joblib

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import preprocess as pp
import model
import utils
import self_loss
from config import *

required_dirs = ["param", "dataset"]
utils.make_dir(required_dirs)

args = utils.get_args()
is_preprocess = args.preprocess
# preprocess
if is_preprocess:
    shutil.rmtree("dataset")
    required_dirs = ["dataset", "dataset/train", "dataset/valid"]
    utils.make_dir(required_dirs)
    pp.preprocess(train_generate_detail, valid_generate_detail)

device = utils.get_gpu_info()

# data load
train_dataset = joblib.load("dataset/train/onehot")
train_label = joblib.load("dataset/train/label") 
train_conditional = joblib.load("dataset/train/conditional")
valid_dataset = joblib.load("dataset/valid/onehot")
valid_label = joblib.load("dataset/valid/label")
valid_conditional = joblib.load("dataset/valid/conditional")

train_correct=torch.LongTensor(
        [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in train_conditional])
valid_correct=torch.LongTensor(
        [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in valid_conditional])

time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size+conditional_size

train_dataset = utils.try_gpu(device,train_dataset)
valid_dataset = utils.try_gpu(device,valid_dataset)
train_correct = utils.try_gpu(device,train_correct)
valid_correct = utils.try_gpu(device,valid_correct)

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("--------------")

criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")

def tuning_trial(trial):
    batch_size = trial.suggest_int("batch_size", 16, 128)
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-2)
    model_param = {
        "emb_size" : trial.suggest_int("emb_size", 10, 256),
        "hidden_size" : trial.suggest_int("hidden_size", 10, 256),
    }

    #vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param)
    classifier=model.Classifier(dfs_size-conditional_size, model_param["emb_size"], model_param["hidden_size"])
    classifier = utils.try_gpu(device,classifier)
    opt = optim.Adam(classifier.parameters(), lr=lr)

    train_data_num = train_dataset.shape[0]
    train_label_args = torch.LongTensor(list(range(train_data_num)))

    train_dl = DataLoader(
            TensorDataset(train_label_args, train_dataset),\
            shuffle=True, batch_size=batch_size)
    train_min_loss = 1e10
    valid_min_loss = 1e10

    keys = ["tu", "tv", "lu", "lv", "le"]
    for epoch in range(1, classifier_epochs):
        print("Epoch: [%d/%d]:"%(epoch, classifier_epochs))

        # train
        loss_sum = 0
        for i, (args, datas) in enumerate(train_dl, 1):
            if i%100==0:
                print("step: [%d/%d]"%(i, train_data_num))
            classifier.train()
            opt.zero_grad()
            datas = utils.try_gpu(device,datas)
            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            #mu, sigma, *result = vae(datas)
            degree, cluster=classifier(datas)
            degree_loss = criterion(degree.squeeze(), train_correct[args][:, 0])
            cluster_loss = criterion(cluster.squeeze(), train_correct[args][:, 1])
            loss=degree_loss+cluster_loss
            loss.backward()
            loss_sum+=loss.item()
            opt.step()

        if train_min_loss>loss_sum:
           train_min_loss = loss_sum
        print("train loss: %lf"%(loss_sum))

        valid_loss_sum = 0
        classifier.eval()
        degree, cluster=classifier(valid_dataset)
        degree_loss = criterion(degree.squeeze(), valid_correct[:, 0])
        cluster_loss = criterion(cluster.squeeze(), valid_correct[:, 1])

        valid_loss=degree_loss+cluster_loss
        valid_loss_sum+=valid_loss.item()

        if valid_min_loss>valid_loss_sum:
            valid_min_loss = valid_loss_sum
        print(" valid loss: %lf"%(valid_loss_sum))
    return valid_min_loss

study = optuna.create_study()
study.optimize(tuning_trial, n_trials=opt_epoch)

print("--------------------------")
print(study.best_params)
print("--------------------------")

f = open("param/classifier_best_tune.yml", "w+")
f.write(yaml.dump(study.best_params))
f.close()
