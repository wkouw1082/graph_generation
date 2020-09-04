import utils
import preprocess as pp
from config import *
import model
import self_loss

from sklearn.model_selection import train_test_split
import joblib
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

args = utils.get_args()
is_preprocess = args.preprocess

# recreate directory
"""
if utils.is_dir_existed("train_result"):
    print("delete file...")
    print("- train_result")
    shutil.rmtree("./train_result")
"""

# preprocess
if is_preprocess:
    required_dirs = ["param", "train_result", "dataset", "dataset/train", "dataset/test"]
    utils.make_dir(required_dirs)
    print("start preprocess...")
    #pp.preprocess(train_generate_detail, "dataset/train")
    #pp.preprocess(test_generate_detail, "dataset/test")
    pp.preprocess(train_generate_detail, test_generate_detail)

# data load
train_dataset = joblib.load("dataset/train/onehot")[0]
train_label = joblib.load("dataset/train/label")
test_dataset = joblib.load("dataset/test/onehot")[0]
test_label = joblib.load("dataset/test/label")

time_size, node_size, edge_size = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size

vae = model.VAE(dfs_size, time_size, node_size, edge_size)
opt = optim.Adam(vae.parameters(), lr=lr)

train_data_num = train_dataset.shape[0]
train_label_args = torch.LongTensor(list(range(train_data_num)))
test_data_num = test_dataset.shape[0]
test_label_args = torch.LongTensor(list(range(test_data_num)))

train_dl = DataLoader(
        TensorDataset(train_label_args, train_dataset),\
        shuffle=True, batch_size=batch_size)
test_dl = DataLoader(
        TensorDataset(test_label_args, test_dataset),\
        shuffle=True, batch_size=batch_size)

keys = ["tu", "tv", "lu", "lv", "le"]
train_loss = {key:[] for key in keys+["encoder"]}
train_acc = {key:[] for key in keys}
test_loss = {key:[] for key in keys+["encoder"]}
test_acc = {key:[] for key in keys}
train_min_loss = 1e10

criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")
encoder_criterion = self_loss.Encoder_Loss()

for epoch in range(1, epochs):
    print("Epoch: [%d/%d]:"%(epoch, epochs))

    # train
    print("train:")
    current_train_loss = {key:[] for key in keys+["encoder"]}
    current_train_acc = {key:[] for key in keys}
    loss_sum = 0
    for i, (args, datas) in enumerate(train_dl):
        if i%1000==0:
            print("step: [%d/%d]"%(i, train_data_num))
        vae.train()
        opt.zero_grad()
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
        loss_sum+=loss.item()
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
    ekey = "encoder"
    loss = np.average(current_train_loss[ekey])
    train_loss[ekey].append(loss)
    print(" %s:"%(ekey))
    print("     loss:%lf"%(loss))
    print("----------------------------")

    # memory free
    del current_train_loss, current_train_acc

    # test
    print("test:")
    current_test_loss = {key:[] for key in keys+["encoder"]}
    current_test_acc = {key:[] for key in keys}
    for i, (args, datas) in enumerate(test_dl):
        if i%1000==0:
            print("step: [%d/%d]"%(i, test_data_num))
        vae.eval()
        opt.zero_grad()
        # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
        mu, sigma, *result = vae(datas)
        encoder_loss = encoder_criterion(mu, sigma)
        current_test_loss["encoder"].append(encoder_loss.item())
        loss = encoder_loss
        for j, pred in enumerate(result):
            current_key = keys[j]
            # loss calc
            correct = test_label[j]
            correct = correct[args]
            tmp_loss = criterion(pred.transpose(2, 1), correct)

            # save
            current_test_loss[current_key].append(tmp_loss.item())

            # acc calc
            pred = torch.argmax(pred, dim=2)  # predicted onehot->label
            pred = pred.view(-1)
            correct = correct.view(-1)
            score = utils.calc_calssification_acc(pred, correct, ignore_label)

            # save
            current_test_acc[current_key].append(score)

    # loss, acc save
    print("----------------------------")
    for key in keys:
        loss = np.average(current_test_loss[key])
        test_loss[key].append(loss)
        acc = np.average(current_test_acc[key])
        test_acc[key].append(acc)

        print(" %s:"%(key))
        print("     loss:%lf"%(loss))
        print("     acc:%lf"%(acc))

    ekey = "encoder"
    loss = np.average(current_test_loss[ekey])
    test_loss[ekey].append(loss)
    print(" %s:"%(ekey))
    print("     loss:%lf"%(loss))
    print("----------------------------")

    # output loss/acc transition
    utils.time_draw(range(epoch), train_loss, "train_result/train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
    utils.time_draw(range(epoch), train_acc, "train_result/train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
    utils.time_draw(range(epoch), test_loss, "train_result/test_loss_transition.png", xlabel="Epoch", ylabel="Loss")
    utils.time_draw(range(epoch), test_acc, "train_result/test_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

    # output weight if train loss is min
    if loss_sum<train_min_loss:
        train_min_loss = loss_sum
        torch.save(vae.state_dict(), "param/weight")
