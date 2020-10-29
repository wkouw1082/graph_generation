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
if utils.is_dir_existed("classifier_train_result"):
    print("delete file...")
    print("- classifier_train_result")
    shutil.rmtree("./classifier_train_result")

required_dirs = ["param", "classifier_train_result", "dataset"]
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

train_correct=torch.LongTensor(
        [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in train_conditional])
valid_correct=torch.LongTensor(
        [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in valid_conditional])

time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")

dfs_size = 2*time_size+2*node_size+edge_size+conditional_size

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("conditional size: %d"%(conditional_size))
print("--------------")

classifier=model.Classifier(dfs_size-conditional_size, classifier_param["emb_size"], classifier_param["hidden_size"])
classifier = utils.try_gpu(classifier)
opt = optim.Adam(classifier.parameters(), lr=classifier_param["lr"])

train_data_num = train_dataset.shape[0]
train_label_args = torch.LongTensor(list(range(train_data_num)))
valid_data_num = valid_dataset.shape[0]
valid_label_args = torch.LongTensor(list(range(valid_data_num)))

train_dl = DataLoader(
        TensorDataset(train_label_args, train_dataset),\
        shuffle=True, batch_size=classifier_param["batch_size"])
valid_dl = DataLoader(
        TensorDataset(valid_label_args, valid_dataset),\
        shuffle=True, batch_size=classifier_param["batch_size"])


#keys = ["tu", "tv", "lu", "lv", "le"]
keys=["degree", "cluster"]
train_loss = {key:[] for key in keys}
train_acc = {key:[] for key in keys}
train_loss_sums = []
valid_loss = {key:[] for key in keys}
valid_loss_sums = []
valid_acc = {key:[] for key in keys}
train_min_loss = 1e10

criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")

for epoch in range(1, classifier_epochs):
    print("Epoch: [%d/%d]:"%(epoch, classifier_epochs))

    # train
    print("train:")
    current_train_loss = {key:[] for key in keys}
    current_train_acc = {key:[] for key in keys}
    train_loss_sum = 0
    for i, (args, datas) in enumerate(train_dl, 1):
        if i%100==0:
            print("step: [%d/%d]"%(i, train_data_num))
        classifier.train()
        opt.zero_grad()
        datas = utils.try_gpu(datas)

        result=classifier(datas)

        loss = 0
        for j, pred in enumerate(result):
            current_key = keys[j]
            # loss calc
            correct = train_correct[args]
            correct = utils.try_gpu(correct)
            tmp_loss = criterion(pred.squeeze(), correct[:, j])
            loss+=tmp_loss

            # save
            current_train_loss[current_key].append(tmp_loss.item())

            # acc calc
            pred = torch.argmax(pred, dim=2)  # predicted onehot->label
            pred = pred.view(-1)
            score = utils.calc_calssification_acc(pred, correct[:, j], ignore_label)

            # save
            current_train_acc[current_key].append(score)
        loss.backward()
        train_loss_sum+=loss.item()
        opt.step()

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
    print("----------------------------")

    # memory free
    del current_train_loss, current_train_acc

    # valid
    print("valid:")
    current_valid_loss = {key:[] for key in keys}
    current_valid_acc = {key:[] for key in keys}
    valid_loss_sum = 0
    for i, (args, datas) in enumerate(valid_dl):
        if i%1000==0:
            print("step: [%d/%d]"%(i, valid_data_num))
        classifier.eval()
        opt.zero_grad()
        datas = utils.try_gpu(datas)
        result = classifier(datas)
        loss=0
        for j, pred in enumerate(result):
            current_key = keys[j]
            # loss calc
            correct = valid_correct[args]
            correct = utils.try_gpu(correct)
            tmp_loss = criterion(pred.squeeze(), correct[:, j])
            loss+=tmp_loss

            # save
            current_valid_loss[current_key].append(tmp_loss.item())

            # acc calc
            pred = torch.argmax(pred, dim=2)  # predicted onehot->label
            pred = pred.view(-1)
            score = utils.calc_calssification_acc(pred, correct[:, j], ignore_label)

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
    print("----------------------------")

    # output loss/acc transition
    utils.time_draw(range(epoch), train_loss, "classifier_train_result/train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
    utils.time_draw(range(epoch), train_acc, "classifier_train_result/train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
    for key in keys:
        utils.time_draw(range(epoch), {key: train_loss[key]}, "classifier_train_result/train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
    utils.time_draw(range(epoch), valid_loss, "classifier_train_result/valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
    utils.time_draw(range(epoch), valid_acc, "classifier_train_result/valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

    train_loss_sums.append(train_loss_sum/train_data_num)
    valid_loss_sums.append(valid_loss_sum/valid_data_num)
    utils.time_draw(
            range(epoch),
            {"train": train_loss_sums, "valid": valid_loss_sums},
            "classifier_train_result/loss_transition.png", xlabel="Epoch", ylabel="Loss")

    # output weight if train loss is min
    if train_loss_sum<train_min_loss:
        train_min_loss = train_loss_sum
        torch.save(classifier.state_dict(), "param/classifier_weight")
    print("\n")
