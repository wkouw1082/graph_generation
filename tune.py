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
is_classifier = args.classifier

# preprocess
if is_preprocess:
    shutil.rmtree("dataset")
    required_dirs = ["dataset", "dataset/train", "dataset/valid"]
    utils.make_dir(required_dirs)
    pp.preprocess(train_generate_detail, valid_generate_detail)

# data load
train_dataset = joblib.load("dataset/train/onehot")
train_label = joblib.load("dataset/train/label") 
train_conditional = joblib.load("dataset/train/conditional")
valid_dataset = joblib.load("dataset/valid/onehot")
valid_label = joblib.load("dataset/valid/label")
valid_conditional = joblib.load("dataset/valid/conditional")
time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size+conditional_size
dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]

if is_classifier:
    # モデルの作成、重み読み込み、gpu化
    classifier=model.Classifier(dfs_size-conditional_size, classifier_param["emb_size"], classifier_param["hidden_size"])
    classifier.load_state_dict(torch.load("param/classifier_weight", map_location="cpu"))
    classifier = utils.try_gpu(classifier)

    # すべてのパラメータを固定
    for param in classifier.parameters():
        param.requires_grad = False

    # 分類用正解データの作成
    train_classifier_correct=torch.LongTensor(
            [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in train_conditional])
    valid_classifier_correct=torch.LongTensor(
            [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in valid_conditional])
    train_classifier_correct = utils.try_gpu(train_classifier_correct)
    valid_classifier_correct = utils.try_gpu(valid_classifier_correct)

train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1)
valid_conditional = torch.cat([valid_conditional for _  in range(valid_dataset.shape[1])],dim=1)

train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
valid_dataset = torch.cat((valid_dataset,valid_conditional),dim=2)

valid_dataset = utils.try_gpu(valid_dataset)

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("--------------")

criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")
encoder_criterion = self_loss.Encoder_Loss()

def tuning_trial(trial):
    batch_size = trial.suggest_int("batch_size", 16, 128)
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-2)
    decay = trial.suggest_loguniform("weight_decay", 1e-5, 0.1)
    clip_th = trial.suggest_loguniform("clip_th", 1e-5, 0.1)
    model_param = {
        "emb_size" : trial.suggest_int("emb_size", 10, 256),
        "en_hidden_size" : trial.suggest_int("en_hidden_size", 10, 256),
        "de_hidden_size" : trial.suggest_int("de_hidden_size", 10, 256),
        "rep_size" : trial.suggest_int("rep_size", 10, 256),
    }

    vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param)
    vae = utils.try_gpu(vae)
    opt = optim.Adam(vae.parameters(), lr=lr, weight_decay=decay)

    train_data_num = train_dataset.shape[0]
    train_label_args = torch.LongTensor(list(range(train_data_num)))

    train_dl = DataLoader(
            TensorDataset(train_label_args, train_dataset),\
            shuffle=True, batch_size=batch_size)
    train_min_loss = 1e10

    valid_min_loss = 1e10

    keys = ["tu", "tv", "lu", "lv", "le"]

    if is_classifier:
        keys+=["classifier"]

    for epoch in range(1, epochs):
        print("Epoch: [%d/%d]:"%(epoch, epochs))

        # train
        loss_sum = 0
        for i, (args, datas) in enumerate(train_dl, 1):
            if i%100==0:
                print("step: [%d/%d]"%(i, train_data_num))
            vae.train()
            opt.zero_grad()
            datas = utils.try_gpu(datas)
            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            mu, sigma, *result = vae(datas)
            encoder_loss = encoder_criterion(mu, sigma)
            loss = encoder_loss
            for j, pred in enumerate(result):
                current_key = keys[j]
                # loss calc
                correct = train_label[j]
                correct = correct[args]
                correct = utils.try_gpu(correct)
                tmp_loss = criterion(pred.transpose(2, 1), correct)
                loss+=tmp_loss

            if is_classifier:
                # とりあえずsamplingせずそのまま突っ込む
                pred_dfs=torch.cat(result, dim=2)
                degree, cluster=classifier(pred_dfs)
                degree_loss = criterion(degree.squeeze(), train_classifier_correct[args][:, 0])
                cluster_loss = criterion(cluster.squeeze(), train_classifier_correct[args][:, 1])
                loss+=degree_loss
                loss+=cluster_loss

            loss.backward()
            loss_sum+=loss.item()
            opt.step()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_th)

        if train_min_loss>loss_sum:
           train_min_loss = loss_sum
        print("train loss: %lf"%(loss_sum))

        valid_loss_sum = 0
        vae.eval()
        mu,sigma, *result = vae(valid_dataset)
        encoder_loss = encoder_criterion(mu, sigma)
        valid_loss = encoder_loss
        for j, pred in enumerate(result):
            # loss calc
            correct = valid_label[j]
            correct = utils.try_gpu(correct)
            tmp_loss = criterion(pred.transpose(2, 1), correct)
            valid_loss+=tmp_loss
        valid_loss_sum+=valid_loss.item()

        if is_classifier:
            # とりあえずsamplingせずそのまま突っ込む
            pred_dfs=torch.cat(result, dim=2)
            degree, cluster=classifier(pred_dfs)
            degree_loss = criterion(degree.squeeze(), valid_classifier_correct[:, 0])
            cluster_loss = criterion(cluster.squeeze(), valid_classifier_correct[:, 1])
            valid_loss_sum+=degree_loss
            valid_loss_sum+=cluster_loss

        if valid_min_loss>valid_loss_sum:
            valid_min_loss = valid_loss_sum
        print(" valid loss: %lf"%(valid_loss_sum))
    return train_min_loss

study = optuna.create_study()
study.optimize(tuning_trial, n_trials=opt_epoch)

print("--------------------------")
print(study.best_params)
print("--------------------------")

f = open("param/best_tune.yml", "w+")
f.write(yaml.dump(study.best_params))
f.close()
