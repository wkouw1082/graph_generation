import utils
import preprocess as pp
from graph_process import graph_statistic
from config import *
import model
import self_loss

from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import shutil

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

args = utils.get_args()
is_preprocess = args.preprocess

# recreate directory
if utils.is_dir_existed("train_result"):
    print("delete file...")
    print("- train_result")
    shutil.rmtree("./train_result")

required_dirs = ["param", "train_result", "dataset"]
utils.make_dir(required_dirs)
print("start preprocess...")

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

train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1)
valid_conditional = torch.cat([valid_conditional for _  in range(valid_dataset.shape[1])],dim=1)

for i in train_conditional:
    print(i)
    print(i.shape)

train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
valid_dataset = torch.cat((valid_dataset,valid_conditional),dim=2)

time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")

dfs_size = 2*time_size+2*node_size+edge_size+conditional_size

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("conditional size: %d"%(conditional_size))
print("--------------")

vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param)
vae = utils.try_gpu(vae)
opt = optim.Adam(vae.parameters(), lr=lr, weight_decay=decay)

train_data_num = train_dataset.shape[0]
train_label_args = torch.LongTensor(list(range(train_data_num)))
valid_data_num = valid_dataset.shape[0]
valid_label_args = torch.LongTensor(list(range(valid_data_num)))

train_dl = DataLoader(
        TensorDataset(train_label_args, train_dataset),\
        shuffle=True, batch_size=batch_size)
valid_dl = DataLoader(
        TensorDataset(valid_label_args, valid_dataset),\
        shuffle=True, batch_size=batch_size)



keys = ["tu", "tv", "lu", "lv", "le"]
train_loss = {key:[] for key in keys+["encoder"]}
train_acc = {key:[] for key in keys}
train_loss_sums = []
valid_loss = {key:[] for key in keys+["encoder"]}
valid_loss_sums = []
valid_acc = {key:[] for key in keys}
train_min_loss = 1e10

criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")
encoder_criterion = self_loss.Encoder_Loss()

for epoch in range(1, epochs):
    print("Epoch: [%d/%d]:"%(epoch, epochs))

    # train
    print("train:")
    current_train_loss = {key:[] for key in keys+["encoder"]}
    current_train_acc = {key:[] for key in keys}
    train_loss_sum = 0
    for i, (args, datas) in enumerate(train_dl, 1):
        if i%100==0:
            print("step: [%d/%d]"%(i, train_data_num))
        vae.train()
        opt.zero_grad()
        datas = utils.try_gpu(datas)

        # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
        mu, sigma, *result = vae(datas)
        encoder_loss = encoder_criterion(mu, sigma)
        current_train_loss["encoder"].append(encoder_loss.item())
        loss = encoder_loss
        for j, pred in enumerate(result):
            current_key = keys[j]
            # loss calc
            correct = train_label[j]
            correct = correct[args]
            correct = utils.try_gpu(correct)
            tmp_loss = criterion(pred.transpose(2, 1), correct)
            loss+=tmp_loss

            # save
            current_train_loss[current_key].append(tmp_loss.item())

            # acc calc
            pred = torch.argmax(pred, dim=2)  # predicted onehot->label
            pred = pred.view(-1)
            correct = correct.view(-1)
            score = utils.calc_calssification_acc(pred, correct, ignore_label)

            # save
            current_train_acc[current_key].append(score)
        loss.backward()
        train_loss_sum+=loss.item()
        opt.step()

        torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_th)

    # loss, acc save
    print("----------------------------")
    for key in keys:
        loss = np.average(current_train_loss[key])
        train_loss[key].append(loss)
        acc = np.average(current_train_acc[key])
        train_acc[key].append(acc)

        print(" %s:"%(key))
        print("     loss:%lf"%(loss))
        print("     acc:%lf"%(acc))
    ekey = "encoder"
    loss = np.average(current_train_loss[ekey])
    train_loss[ekey].append(loss)
    print(" %s:"%(ekey))
    print("     loss:%lf"%(loss))
    print("----------------------------")

    # memory free
    del current_train_loss, current_train_acc

    # valid
    print("valid:")
    current_valid_loss = {key:[] for key in keys+["encoder"]}
    current_valid_acc = {key:[] for key in keys}
    valid_loss_sum = 0
    for i, (args, datas) in enumerate(valid_dl):
        if i%1000==0:
            print("step: [%d/%d]"%(i, valid_data_num))
        vae.eval()
        opt.zero_grad()
        datas = utils.try_gpu(datas)
        # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
        mu, sigma, *result = vae(datas)
        encoder_loss = encoder_criterion(mu, sigma)
        current_valid_loss["encoder"].append(encoder_loss.item())
        loss = encoder_loss
        for j, pred in enumerate(result):
            current_key = keys[j]
            # loss calc
            correct = valid_label[j]
            correct = correct[args]
            correct = utils.try_gpu(correct)
            tmp_loss = criterion(pred.transpose(2, 1), correct)
            loss+=tmp_loss.item()

            # save
            current_valid_loss[current_key].append(tmp_loss.item())

            # acc calc
            pred = torch.argmax(pred, dim=2)  # predicted onehot->label
            pred = pred.view(-1)
            correct = correct.view(-1)
            score = utils.calc_calssification_acc(pred, correct, ignore_label)

            # save
            current_valid_acc[current_key].append(score)
        valid_loss_sum+=loss.item()

    # loss, acc save
    print("----------------------------")
    for key in keys:
        loss = np.average(current_valid_loss[key])
        valid_loss[key].append(loss)
        acc = np.average(current_valid_acc[key])
        valid_acc[key].append(acc)

        print(" %s:"%(key))
        print("     loss:%lf"%(loss))
        print("     acc:%lf"%(acc))

    ekey = "encoder"
    loss = np.average(current_valid_loss[ekey])
    valid_loss[ekey].append(loss)
    print(" %s:"%(ekey))
    print("     loss:%lf"%(loss))
    print("----------------------------")

    # output loss/acc transition
    utils.time_draw(range(epoch), train_loss, "train_result/train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
    utils.time_draw(range(epoch), train_acc, "train_result/train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
    for key in keys+["encoder"]:
        utils.time_draw(range(epoch), {key: train_loss[key]}, "train_result/train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
    utils.time_draw(range(epoch), valid_loss, "train_result/valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
    utils.time_draw(range(epoch), valid_acc, "train_result/valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

    train_loss_sums.append(train_loss_sum/train_data_num)
    valid_loss_sums.append(valid_loss_sum/valid_data_num)
    utils.time_draw(
            range(epoch),
            {"train": train_loss_sums, "valid": valid_loss_sums},
            "train_result/loss_transition.png", xlabel="Epoch", ylabel="Loss")

    # output weight if train loss is min
    if train_loss_sum<train_min_loss:
        train_min_loss = train_loss_sum
        torch.save(vae.state_dict(), "param/weight")
    print("\n")
