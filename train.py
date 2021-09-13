# from data_input import GCNDataset
from typing_extensions import runtime

from matplotlib.pyplot import winter
import utils
import argparse
import preprocess as pp
from graph_process import graph_statistic, rewrite_dataset_condition
from config import *
import model
import self_loss

from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import shutil
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

import dgl
from dgl.dataloading import GraphDataLoader


def conditional_train(args):
    writer = SummaryWriter(log_dir="./logs")

    is_preprocess = args.preprocess
    is_classifier = args.classifier

    device = utils.get_gpu_info()

    # recreate directory
    if utils.is_dir_existed("train_result"):
        print("delete file...")
        print("- train_result")
        shutil.rmtree("./train_result")

    # required_dirs = ["param", "train_result", "dataset"]
    # utils.make_dir(required_dirs)
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
    time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
    
    # this_is(train_dataset, name="train_dataset (after load)")
    # this_is(train_label[0], name="train_label[0] (after load)")
    # this_is(train_conditional, name="train_conditional (after load)")

    dfs_size = 2*time_size+2*node_size+edge_size+conditional_size
    dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]

    if is_classifier:
        # モデルの作成、重み読み込み、gpu化
        classifier=model.Classifier(dfs_size-conditional_size, classifier_param["emb_size"], classifier_param["hidden_size"])
        classifier.load_state_dict(torch.load("param/classifier_weight", map_location="cpu"))
        classifier = utils.try_gpu(device,classifier)

        # すべてのパラメータを固定
        for param in classifier.parameters():
            param.requires_grad = False

        # 分類用正解データの作成
        train_classifier_correct=torch.LongTensor(
                [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in train_conditional])
        valid_classifier_correct=torch.LongTensor(
                [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in valid_conditional])
        train_classifier_correct = utils.try_gpu(device,train_classifier_correct)
        valid_classifier_correct = utils.try_gpu(device,valid_classifier_correct)

    train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1).unsqueeze(2)
    valid_conditional = torch.cat([valid_conditional for _  in range(valid_dataset.shape[1])],dim=1).unsqueeze(2)
    
    # this_is(train_conditional, name="train_conditional")    

    train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
    valid_dataset = torch.cat((valid_dataset,valid_conditional),dim=2)
    # print(train_dataset[1,:,-1*condition_size:])
    
    # this_is(train_dataset, name="train_dataset (after cat)")

    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("conditional size: %d"%(conditional_size))
    print("--------------")

    # model_param load
    model_param = utils.load_model_param()
    # print(f"model_param = {model_param}")

    vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param, device)
    vae = utils.try_gpu(device,vae)

    opt = optim.Adam(vae.parameters(), lr=0.001)


    train_data_num = train_dataset.shape[0]
    train_label_args = torch.LongTensor(list(range(train_data_num)))
    valid_data_num = valid_dataset.shape[0]
    valid_label_args = torch.LongTensor(list(range(valid_data_num)))

    # this_is(train_data_num, name="train_data_num")
    # this_is(train_label_args, name="train_label_args")

    train_dl = DataLoader(
            TensorDataset(train_label_args, train_dataset),\
            shuffle=True, batch_size=model_param["batch_size"])
    valid_dl = DataLoader(
            TensorDataset(valid_label_args, valid_dataset),\
            shuffle=False, batch_size=model_param["batch_size"])
    
    # this_is(train_dl, name="train_dl")

    keys = ["tu", "tv", "lu", "lv", "le"]
    if is_classifier:
        keys+=["classifier"]

    train_loss = {key:[] for key in keys+["encoder"]}
    train_acc = {key:[] for key in keys}
    train_loss_sums = []
    valid_loss = {key:[] for key in keys+["encoder"]}
    valid_loss_sums = []
    valid_acc = {key:[] for key in keys}
    train_min_loss = 1e10
    
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")
    encoder_criterion = self_loss.Encoder_Loss()
    timestep=0

    print("start conditional train...")

    for epoch in range(1, epochs+1):
        print("Epoch: [%d/%d]:"%(epoch, epochs))

        # train
        print("train:")
        current_train_loss = {key:[] for key in keys+["encoder"]}
        current_train_acc = {key:[] for key in keys}
        train_loss_sum = 0
        for i, (args, datas) in enumerate(train_dl, 1):
            
            # this_is(args, name="args")
            # this_is(datas, name="datas")
            
            if i%100==0:
                print("step: [%d/%d]"%(i, train_data_num))
            vae.train()
            opt.zero_grad()
            datas = utils.try_gpu(device,datas)

            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            #mu, sigma, *result = vae(datas, timestep)
            mu, sigma, *result = vae(datas, word_drop=word_drop_rate)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            this_is(encoder_loss, name="encoder_loss")
            this_is(encoder_loss.item(), name="encoder_loss.item()")
            current_train_loss["encoder"].append(encoder_loss.item())
            loss = encoder_loss
            for j, pred in enumerate(result):
                this_is(pred, name="pred")
                current_key = keys[j]
                # loss calc
                correct = train_label[j]
                this_is(train_label[j], name=f"train_label[{j}]")
                correct = correct[args]
                this_is(correct, name=f"correct")
                correct = utils.try_gpu(device,correct)
                tmp_loss = criterion(pred.transpose(2, 1), correct)
                this_is(pred.transpose(2, 1), name="transpose(pred) (2, 1)")
                this_is(tmp_loss, name="tmp_loss")
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

            timestep+=1
            if is_classifier:
                # とりあえずsamplingせずそのまま突っ込む
                pred_dfs=torch.cat(result, dim=2)
                degree, cluster=classifier(pred_dfs)
                degree_loss = criterion(degree.squeeze(), train_classifier_correct[args][:, 0])
                cluster_loss = criterion(cluster.squeeze(), train_classifier_correct[args][:, 1])
                current_train_loss["classifier"].append((degree_loss+cluster_loss).item()*classifier_bias)
                loss+=degree_loss*classifier_bias
                loss+=cluster_loss*classifier_bias

                # acc calc
                pred=degree
                pred = torch.argmax(pred, dim=2)  # predicted onehot->label
                pred = pred.view(-1)
                degreescore = utils.calc_calssification_acc(pred, train_classifier_correct[args][:, 0], ignore_label)

                pred=cluster
                pred = torch.argmax(pred, dim=2)  # predicted onehot->label
                pred = pred.view(-1)
                clusterscore = utils.calc_calssification_acc(pred, train_classifier_correct[args][:, 1], ignore_label)
                score=(degreescore+clusterscore)/2

                current_train_acc["classifier"].append(score)

            loss.backward()
            train_loss_sum+=loss.item()
            del loss
            opt.step()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), model_param["clip_th"])

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
        writer.add_scalar("train/train_loss", loss, epoch)

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
            datas = utils.try_gpu(device,datas)
            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            mu, sigma, *result = vae(datas)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            current_valid_loss["encoder"].append(encoder_loss.item())
            loss = encoder_loss
            for j, pred in enumerate(result):
                current_key = keys[j]
                # loss calc
                correct = valid_label[j]
                correct = correct[args]
                correct = utils.try_gpu(device,correct)
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

            if is_classifier:
                # とりあえずsamplingせずそのまま突っ込む
                pred_dfs=torch.cat(result, dim=2)
                degree, cluster=classifier(pred_dfs)
                degree_loss = criterion(degree.squeeze(), valid_classifier_correct[args][:, 0])
                cluster_loss = criterion(cluster.squeeze(), valid_classifier_correct[args][:, 1])
                current_valid_loss["classifier"].append((degree_loss+cluster_loss).item())
                loss+=degree_loss
                loss+=cluster_loss

                # acc calc
                pred=degree
                pred = torch.argmax(pred, dim=2)  # predicted onehot->label
                pred = pred.view(-1)
                degreescore = utils.calc_calssification_acc(pred, valid_classifier_correct[args][:, 0], ignore_label)

                pred=cluster
                pred = torch.argmax(pred, dim=2)  # predicted onehot->label
                pred = pred.view(-1)
                clusterscore = utils.calc_calssification_acc(pred, valid_classifier_correct[args][:, 1], ignore_label)
                score=(degreescore+clusterscore)/2

                current_valid_acc["classifier"].append(score)

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
        writer.add_scalar("train/vallid_loss", loss, epoch)

        # output loss/acc transition
        utils.time_draw(range(epoch), train_loss, "results/"+run_time+"/train/train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), train_acc, "results/"+run_time+"/train/train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        for key in keys+["encoder"]:
            utils.time_draw(range(epoch), {key: train_loss[key]}, "results/"+run_time+"/train/train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_loss, "results/"+run_time+"/train/valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_acc, "results/"+run_time+"/train/valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

        train_loss_sums.append(train_loss_sum/train_data_num)
        valid_loss_sums.append(valid_loss_sum/valid_data_num)
        utils.time_draw(
                range(epoch),
                {"train": train_loss_sums, "valid": valid_loss_sums},
                "results/"+run_time+"/train/loss_transition.png", xlabel="Epoch", ylabel="Loss")

        # output weight each 1000 epochs
        # if epoch % 1000 == 0:
        if epoch % 200 == 0:
            torch.save(vae.state_dict(), "param/weight_"+str(epoch))
            torch.save(vae.state_dict(), "results/" + run_time + "/train/weight_" + str(epoch))

        # 最も性能のいいモデルを保存
        if train_loss_sum<train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(vae.state_dict(), "results/" + run_time + "/train/weight")

        print("\n")
    
    writer.close()

def train(args):
    is_preprocess = args.preprocess
    is_classifier = args.classifier

    device = utils.get_gpu_info()

    # recreate directory
    if utils.is_dir_existed("train_result"):
        print("delete file...")
        print("- train_result")
        shutil.rmtree("./train_result")

    # required_dirs = ["param", "train_result", "dataset"]
    # utils.make_dir(required_dirs)
    print("start preprocess...")

    # preprocess
    if is_preprocess:
        shutil.rmtree("dataset")
        required_dirs = ["dataset", "dataset/train", "dataset/valid"]
        utils.make_dir(required_dirs)
        pp.preprocess_not_conditional(train_generate_detail, valid_generate_detail)

    # data load
    train_dataset = joblib.load("dataset/train/onehot")
    train_label = joblib.load("dataset/train/label") 
    valid_dataset = joblib.load("dataset/valid/onehot")
    valid_label = joblib.load("dataset/valid/label")
    time_size, node_size, edge_size = joblib.load("dataset/param")

    dfs_size = 2*time_size+2*node_size+edge_size
    dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]

    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("--------------")
    
    # model_param load
    import yaml
    with open('results/best_tune.yml', 'r') as yml:
        model_param = yaml.load(yml) 
    # print(f"model_param = {model_param}")

    vae = model.VAENonConditional(dfs_size, time_size, node_size, edge_size, model_param, device)
    vae = utils.try_gpu(device,vae)
    opt = optim.Adam(vae.parameters(), lr=model_param["lr"], weight_decay=model_param["weight_decay"])


    train_data_num = train_dataset.shape[0]
    train_label_args = torch.LongTensor(list(range(train_data_num)))
    valid_data_num = valid_dataset.shape[0]
    valid_label_args = torch.LongTensor(list(range(valid_data_num)))

    train_dl = DataLoader(
            TensorDataset(train_label_args, train_dataset),\
            shuffle=True, batch_size=model_param["batch_size"])
    valid_dl = DataLoader(
            TensorDataset(valid_label_args, valid_dataset),\
            shuffle=True, batch_size=model_param["batch_size"])

    keys = ["tu", "tv", "lu", "lv", "le"]
    if is_classifier:
        keys+=["classifier"]

    train_loss = {key:[] for key in keys+["encoder"]}
    train_acc = {key:[] for key in keys}
    train_loss_sums = []
    valid_loss = {key:[] for key in keys+["encoder"]}
    valid_loss_sums = []
    valid_acc = {key:[] for key in keys}
    train_min_loss = 1e10

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")
    encoder_criterion = self_loss.Encoder_Loss()
    timestep=0

    for epoch in range(1, epochs+1):
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
            datas = utils.try_gpu(device,datas)

            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            #mu, sigma, *result = vae(datas, timestep)
            mu, sigma, *result = vae(datas, word_drop=word_drop_rate)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            current_train_loss["encoder"].append(encoder_loss.item())
            loss = encoder_loss
            for j, pred in enumerate(result):
                current_key = keys[j]
                # loss calc
                correct = train_label[j]
                correct = correct[args]
                correct = utils.try_gpu(device,correct)
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

            timestep+=1

            loss.backward()
            train_loss_sum+=loss.item()
            opt.step()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), model_param["clip_th"])

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
            datas = utils.try_gpu(device,datas)
            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            mu, sigma, *result = vae(datas)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            current_valid_loss["encoder"].append(encoder_loss.item())
            loss = encoder_loss
            for j, pred in enumerate(result):
                current_key = keys[j]
                # loss calc
                correct = valid_label[j]
                correct = correct[args]
                correct = utils.try_gpu(device,correct)
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
        utils.time_draw(range(epoch), train_loss, "results/"+run_time+"/train/train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), train_acc, "results/"+run_time+"/train/train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        for key in keys+["encoder"]:
            utils.time_draw(range(epoch), {key: train_loss[key]}, "results/"+run_time+"/train/train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_loss, "results/"+run_time+"/train/valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_acc, "results/"+run_time+"/train/valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

        train_loss_sums.append(train_loss_sum/train_data_num)
        valid_loss_sums.append(valid_loss_sum/valid_data_num)
        utils.time_draw(
                range(epoch),
                {"train": train_loss_sums, "valid": valid_loss_sums},
                "results/"+run_time+"/train/loss_transition.png", xlabel="Epoch", ylabel="Loss")

        # output weight if train loss is min
        if train_loss_sum<train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(vae.state_dict(), "param/weight")
            torch.save(vae.state_dict(), "results/" + run_time + "/train/weight")
        print("\n")

def gcn_train(args):
    if args.preprocess:
        print("start preprocess...")
        pp.preprocess_gcn(train_generate_detail, valid_generate_detail,condition=args.condition)

    writer = SummaryWriter(log_dir="./logs")

    device = utils.get_gpu_info()

    time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")

    dfs_size = 2*time_size+2*node_size+edge_size+conditional_size
    dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]

    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("conditional size: %d"%(conditional_size))
    print("--------------")

    # model_param load
    model_param = utils.load_model_param()
    model_param = {'batch_size':8, 'clip_th': 0.03,'emb_size':150, 'en_hidden_size': 40, 'de_hidden_size': 230, 'rep_size': 175}
    # print(f"model_param = {model_param}")

    vae = model.GCNVAE(dfs_size, time_size, node_size, edge_size, model_param, device)
    vae = utils.try_gpu(device,vae)

    opt = optim.Adam(vae.parameters(), lr=0.001)

    train_dataset = GCNDataset('train', conditional=args.condition)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=model_param['batch_size'], shuffle=True)
    valid_dataset = GCNDataset('valid', conditional=args.condition)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=model_param['batch_size'], shuffle=False)

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
    timestep=0

    for epoch in range(1, epochs+1):
        print("Epoch: [%d/%d]:"%(epoch, epochs))

        # train
        print("train:")
        current_train_loss = {key:[] for key in keys+["encoder"]}
        current_train_acc = {key:[] for key in keys}
        train_loss_sum = 0
        for i, (datas, labels) in enumerate(train_dataloader, 1):
            vae.train()
            opt.zero_grad()
            datas = datas.to(device)
            graph_feats = datas.ndata['feat']

            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            #mu, sigma, *result = vae(datas, timestep)
            mu, sigma, *result = vae(datas, graph_feats, word_drop=word_drop_rate)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            current_train_loss["encoder"].append(encoder_loss.item())
            loss = encoder_loss
            for j, pred in enumerate(result):
                current_key = keys[j]
                # loss calc
                correct = labels[:,j]
                correct_copy = correct.tolist()
                ignore_index = correct_copy[0].index(ignore_label)

                correct = correct[:ignore_index+1]
                correct = utils.try_gpu(device,correct)
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

            timestep+=1

            loss.backward()
            train_loss_sum+=loss.item()
            # del loss
            opt.step()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), model_param["clip_th"])

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
        writer.add_scalar("train_condition/train_loss", loss, epoch)

        # memory free
        del current_train_loss, current_train_acc

        # valid
        print("valid:")
        current_valid_loss = {key:[] for key in keys+["encoder"]}
        current_valid_acc = {key:[] for key in keys}
        valid_loss_sum = 0
        for i, (datas, labels) in enumerate(valid_dataloader):
            vae.eval()
            opt.zero_grad()
            datas = datas.to(device)
            graph_feats = datas.ndata['feat']

            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            mu, sigma, *result = vae(datas, graph_feats)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            current_valid_loss["encoder"].append(encoder_loss.item())
            loss = encoder_loss
            for j, pred in enumerate(result):
                current_key = keys[j]
                # loss calc
                correct = labels[:,j]
                correct = utils.try_gpu(device,correct)
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
        writer.add_scalar("train_condition/vallid_loss", loss, epoch)

        # output loss/acc transition
        utils.time_draw(range(epoch), train_loss, "results/"+run_time+"/train/train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), train_acc, "results/"+run_time+"/train/train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        for key in keys+["encoder"]:
            utils.time_draw(range(epoch), {key: train_loss[key]}, "results/"+run_time+"/train/train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_loss, "results/"+run_time+"/train/valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_acc, "results/"+run_time+"/train/valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

        train_loss_sums.append(train_loss_sum/len(train_dataset))
        valid_loss_sums.append(valid_loss_sum/len(valid_dataset))
        utils.time_draw(
                range(epoch),
                {"train": train_loss_sums, "valid": valid_loss_sums},
                "results/"+run_time+"/train/loss_transition.png", xlabel="Epoch", ylabel="Loss")

        # output weight each 1000 epochs
        # if epoch % 1000 == 0:
        if epoch % 200 == 0:
            torch.save(vae.state_dict(), "param/weight_"+str(epoch))
            torch.save(vae.state_dict(), "results/" + run_time + "/train/weight_" + str(epoch))

        # 最も性能のいいモデルを保存
        if train_loss_sum<train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(vae.state_dict(), "results/" + run_time + "/train/weight")

        print("\n")
    
    writer.close()

def train_with_sequential_conditions(args):
    """conditionをlstmに入力した順に逐次作成して付与した学習
    
    lstmに５タプルを順に入力することで蓄積されたdfsコードからnetworkx形式のグラフを構築し、
    ５タプルに加えるconditionをそのグラフから計算して付与する。
    
    // Composition of one dfs_code
    *-------------*-------------*---------*---------*---------*-----------*
    | timestamp_u | timestamp_v | label_u | label_v | label_e | condition |
    *-------------*-------------*---------*---------*---------*-----------*

    Args:
        args (argparse.ArgumentParser.parse_args()): preprocessや使用modelなどデータ
    """
    writer = SummaryWriter(log_dir="./logs")

    is_preprocess = args.preprocess
    is_classifier = args.classifier

    device = utils.get_gpu_info()

    # recreate directory
    if utils.is_dir_existed("train_result"):
        print("delete file...")
        print("- train_result")
        shutil.rmtree("./train_result")

    # required_dirs = ["param", "train_result", "dataset"]
    # utils.make_dir(required_dirs)

    # preprocess
    if is_preprocess:
        print("start preprocess...")
        shutil.rmtree("dataset")
        required_dirs = ["dataset", "dataset/train", "dataset/valid"]
        utils.make_dir(required_dirs)
        pp.preprocess(train_generate_detail, valid_generate_detail)

    # data load
    ## train_dataset = [グラフの数, ５タプルの最大数, dfs_codeのサイズ], 値は0 or 1
    train_dataset = joblib.load("dataset/train/onehot")
    ## train_label = [5(5タプルの長さ), グラフの数, ５タプルの最大数], 値はid or ignore_label
    train_label = joblib.load("dataset/train/label") 
    ## train_conditional = [グラフの数, 1], 値はグラフの特性値(少数第１位で丸め)
    train_conditional = joblib.load("dataset/train/conditional")
    valid_dataset = joblib.load("dataset/valid/onehot")
    valid_label = joblib.load("dataset/valid/label")
    valid_conditional = joblib.load("dataset/valid/conditional")
    time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
    
    # dfs_codeのサイズ
    dfs_size = 2*time_size+2*node_size+edge_size+conditional_size
    dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]

    if is_classifier:
        # モデルの作成、重み読み込み、gpu化
        classifier=model.Classifier(dfs_size-conditional_size, classifier_param["emb_size"], classifier_param["hidden_size"])
        classifier.load_state_dict(torch.load("param/classifier_weight", map_location="cpu"))
        classifier = utils.try_gpu(device,classifier)

        # すべてのパラメータを固定
        for param in classifier.parameters():
            param.requires_grad = False

        # 分類用正解データの作成
        train_classifier_correct=torch.LongTensor(
                [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in train_conditional])
        valid_classifier_correct=torch.LongTensor(
                [[torch.argmax(tensor[:, :3],dim=1), torch.argmax(tensor[:, 3:],dim=1)] for tensor in valid_conditional])
        train_classifier_correct = utils.try_gpu(device,train_classifier_correct)
        valid_classifier_correct = utils.try_gpu(device,valid_classifier_correct)

    # train_conditional = [グラフの数, ５タプルの最大数, 1], 値はグラフの特性値
    train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1).unsqueeze(2)
    valid_conditional = torch.cat([valid_conditional for _  in range(valid_dataset.shape[1])],dim=1).unsqueeze(2)

    # train_dataset = [グラフの数, ５タプルの最大数, dfs_codeのサイズ+1], 値はone-hotの５タプルとグラフの特性値
    train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
    valid_dataset = torch.cat((valid_dataset,valid_conditional),dim=2)
    ## print(train_dataset[1,:,-1*condition_size:])
    
    ## rewrite condition of train_dataset
    train_dataset = rewrite_dataset_condition(train_dataset, time_size, dfs_size, rewrited_condition_dump_name=f"train_conditions", flag_dump=True)
    joblib.dump(train_dataset, "./train_dataset")
    # with open("./train_dataset", "rb") as f:
    #     train_dataset = joblib.load(f)

    ## rewrite condition of valid_dataset
    valid_dataset = rewrite_dataset_condition(valid_dataset, time_size, dfs_size, rewrited_condition_dump_name=f"valid_conditions", flag_dump=True)
    joblib.dump(valid_dataset, "./valid_dataset")
    # with open("./valid_dataset", "rb") as f:
    #     valid_dataset = joblib.load(f)
    
    
    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("conditional size: %d"%(conditional_size))
    print("--------------")

    # model_param load
    model_param = utils.load_model_param()
    ## print(f"model_param = {model_param}")

    # create VAE model
    vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param, device, decoder_type="+conditions")
    vae = utils.try_gpu(device,vae)

    # set optimizer
    opt = optim.Adam(vae.parameters(), lr=0.001)

    # train_data_num = グラフの数
    train_data_num = train_dataset.shape[0]
    # train_label_args = [グラフ数], 値は０〜（グラフ数−１）のindex
    train_label_args = torch.LongTensor(list(range(train_data_num)))
    valid_data_num = valid_dataset.shape[0]
    valid_label_args = torch.LongTensor(list(range(valid_data_num)))

    # create DataLoader
    train_dl = DataLoader(
            TensorDataset(train_label_args, train_dataset),\
            shuffle=True, batch_size=model_param["batch_size"])
    valid_dl = DataLoader(
            TensorDataset(valid_label_args, valid_dataset),\
            shuffle=False, batch_size=model_param["batch_size"])

    # keys : ５タプルの名前のリスト
    keys = ["tu", "tv", "lu", "lv", "le"]
    if is_classifier:
        keys+=["classifier"]

    # train_loss = {'tu': [], 'tv': [], 'lu': [], 'lv': [], 'le': [], 'encoder': []}
    train_loss = {key:[] for key in keys+["encoder"]+["conditions"]}
    # train_acc = {'tu': [], 'tv': [], 'lu': [], 'lv': [], 'le': []}
    train_acc = {key:[] for key in keys}
    train_loss_sums = []
    valid_loss = {key:[] for key in keys+["encoder"]+["conditions"]}
    valid_acc = {key:[] for key in keys}
    valid_loss_sums = []
    train_min_loss = 1e10
    
    # set Loss        
    ## tu, tv, lu, lv, leの誤差関数
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")
    ## conditionsの誤差関数
    conditions_criterion = nn.MSELoss(reduction="sum")
    ## encoderの誤差関数（KL-divercity）
    encoder_criterion = self_loss.Encoder_Loss()
    timestep=0

    print("start train_with_sequential_conditions ...")
    for epoch in range(1, epochs+1):
        print("Epoch: [%d/%d]:"%(epoch, epochs))

        # train
        print("train:")
        ## current_train_loss = {'tu': [], 'tv': [], 'lu': [], 'lv': [], 'le': [], 'encoder': []}
        current_train_loss = {key:[] for key in keys+["encoder"]+["conditions"]}
        ## current_train_acc = {'tu': [], 'tv': [], 'lu': [], 'lv': [], 'le': []}
        current_train_acc = {key:[] for key in keys}
        train_loss_sum = 0
        for i, (args, datas) in enumerate(train_dl, 1):
            ## args = [バッチサイズ], 値はどのグラフを参照するかを示すindex(0 ~ グラフ数-1)
            ## datas = [バッチサイズ, ５タプルの最大数, dfs_codeのサイズ+1], train_datasetのバッチサイズ分
            
            if i%100==0:
                print("step: [%d/%d]"%(i, train_data_num))
            vae.train()
            opt.zero_grad()
            datas = utils.try_gpu(device,datas)

            # mu,sigma, [tu, tv, lu, lv, le, conditions] = vae(datas)
            # mu, sigma, *result = vae(datas, timestep)
            ## mu = [バッチサイズ, 1, rep_size]
            ## sigma = [バッチサイズ, 1, rep_size]
            mu, sigma, *result, conds = vae(datas, word_drop=word_drop_rate)
            
            # generate graphs
            # bbb
            # ans = vae.generate(3, torch.tensor([[0.1]])[0])
            # this_is(ans,name="generated")
            # result = [code.unsqueeze(2) for code in ans]
            # this_is(result[0], name="result[0]")
            # dfs_code = torch.cat(result, dim=2)
            # this_is(dfs_code, name="dfs_code")
            # generated_graph = []
            # for code in dfs_code:
            #     this_is(code, name="code (in for loop)")
            #     import graph_process
            #     graph = graph_process.dfs_code_to_graph_obj(
            #             code.cpu().detach().numpy(),
            #             [time_size, time_size, node_size, node_size, edge_size])
            #     this_is(graph, name="graph")
            #     #if gp.is_connect(graph):
            #     if gp.is_connect(graph) and is_sufficient_size(graph):
            #         generated_graph.append(graph)
            # qqq
            
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            ## encoder_loss.item() = loss値
            current_train_loss["encoder"].append(encoder_loss.item())
            loss = encoder_loss
            
            for j, pred in enumerate(result):
                ## pred = [バッチサイズ, ５タプルの最大数, ５タプルの各ラベルのサイズ], 確率値(0 ~ 1)
                current_key = keys[j]
                
                # loss calc
                correct = train_label[j]
                ## correct = [バッチサイズ, ５タプルの最大数], ５タプルの各ラベルのid or ignore_label
                correct = correct[args]
                correct = utils.try_gpu(device,correct)
                ## tmp_loss.item() = loss値
                ## pred_transpose(2, 1) = [バッチサイズ, ５タプルの各サイズ, ５タプルの最大数], 確率値(0 ~ 1)
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
            
            # calc train conds' loss
            correct = train_conditional[args]
            correct = torch.squeeze(correct)
            conds = torch.squeeze(conds)
            conds_loss = conditions_criterion(conds, correct)
            # add conds_loss to VAE loss(named "loss")
            loss += conds_loss
            # save conds_loss
            current_train_loss["conditions"].append(conds_loss.item())
            
            timestep+=1
            if is_classifier:
                # とりあえずsamplingせずそのまま突っ込む
                pred_dfs=torch.cat(result, dim=2)
                degree, cluster=classifier(pred_dfs)
                degree_loss = criterion(degree.squeeze(), train_classifier_correct[args][:, 0])
                cluster_loss = criterion(cluster.squeeze(), train_classifier_correct[args][:, 1])
                current_train_loss["classifier"].append((degree_loss+cluster_loss).item()*classifier_bias)
                loss+=degree_loss*classifier_bias
                loss+=cluster_loss*classifier_bias

                # acc calc
                pred=degree
                pred = torch.argmax(pred, dim=2)  # predicted onehot->label
                pred = pred.view(-1)
                degreescore = utils.calc_calssification_acc(pred, train_classifier_correct[args][:, 0], ignore_label)

                pred=cluster
                pred = torch.argmax(pred, dim=2)  # predicted onehot->label
                pred = pred.view(-1)
                clusterscore = utils.calc_calssification_acc(pred, train_classifier_correct[args][:, 1], ignore_label)
                score=(degreescore+clusterscore)/2

                current_train_acc["classifier"].append(score)

            # backpropagation
            loss.backward()
            train_loss_sum+=loss.item()
            del loss
            opt.step()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), model_param["clip_th"])

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
        ekey = "conditions"
        loss = np.average(current_train_loss[ekey])
        train_loss[ekey].append(loss)
        print(" %s:"%(ekey))
        print("     loss:%lf"%(loss))
        print("----------------------------")
        writer.add_scalar("train/train__seq_cond_loss", loss, epoch)

        # memory free
        del current_train_loss, current_train_acc

        # valid
        print("valid:")
        current_valid_loss = {key:[] for key in keys+["encoder"]+["conditions"]}
        current_valid_acc = {key:[] for key in keys}
        valid_loss_sum = 0
        for i, (args, datas) in enumerate(valid_dl):
            if i%1000==0:
                print("step: [%d/%d]"%(i, valid_data_num))
            vae.eval()
            opt.zero_grad()
            datas = utils.try_gpu(device,datas)
            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            mu, sigma, *result, conds = vae(datas)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            current_valid_loss["encoder"].append(encoder_loss.item())
            loss = encoder_loss
            for j, pred in enumerate(result):
                current_key = keys[j]
                # loss calc
                correct = valid_label[j]
                correct = correct[args]
                correct = utils.try_gpu(device,correct)
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

            # calc valid conds' loss
            correct = valid_conditional[args]
            correct = torch.squeeze(correct)
            conds = torch.squeeze(conds)
            conds_loss = conditions_criterion(conds, correct)
            # add conds_loss to VAE loss(named "loss")
            loss += conds_loss
            # save conds_loss
            current_valid_loss["conditions"].append(conds_loss.item())
            
            if is_classifier:
                # とりあえずsamplingせずそのまま突っ込む
                pred_dfs=torch.cat(result, dim=2)
                degree, cluster=classifier(pred_dfs)
                degree_loss = criterion(degree.squeeze(), valid_classifier_correct[args][:, 0])
                cluster_loss = criterion(cluster.squeeze(), valid_classifier_correct[args][:, 1])
                current_valid_loss["classifier"].append((degree_loss+cluster_loss).item())
                loss+=degree_loss
                loss+=cluster_loss

                # acc calc
                pred=degree
                pred = torch.argmax(pred, dim=2)  # predicted onehot->label
                pred = pred.view(-1)
                degreescore = utils.calc_calssification_acc(pred, valid_classifier_correct[args][:, 0], ignore_label)

                pred=cluster
                pred = torch.argmax(pred, dim=2)  # predicted onehot->label
                pred = pred.view(-1)
                clusterscore = utils.calc_calssification_acc(pred, valid_classifier_correct[args][:, 1], ignore_label)
                score=(degreescore+clusterscore)/2

                current_valid_acc["classifier"].append(score)

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
        ekey = "conditions"
        loss = np.average(current_valid_loss[ekey])
        valid_loss[ekey].append(loss)
        print(" %s:"%(ekey))
        print("     loss:%lf"%(loss))
        print("----------------------------")
        writer.add_scalar("train/vallid_seq_conds_loss", loss, epoch)

        # output loss/acc transition
        utils.time_draw(range(epoch), train_loss, "results/"+run_time+"/train/train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), train_acc, "results/"+run_time+"/train/train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        for key in keys+["encoder"]+["conditions"]:
            utils.time_draw(range(epoch), {key: train_loss[key]}, "results/"+run_time+"/train/train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
            writer.add_scalar(f"train/{key}", train_loss[key], epoch)
        utils.time_draw(range(epoch), valid_loss, "results/"+run_time+"/train/valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_acc, "results/"+run_time+"/train/valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        writer.add_scalar("train/train_seq_acc", train_acc, epoch)
        writer.add_scalar("train/valid_seq_acc", valid_acc, epoch)

        train_loss_sums.append(train_loss_sum/train_data_num)
        valid_loss_sums.append(valid_loss_sum/valid_data_num)
        utils.time_draw(
                range(epoch),
                {"train": train_loss_sums, "valid": valid_loss_sums},
                "results/"+run_time+"/train/loss_transition.png", xlabel="Epoch", ylabel="Loss")

        # output weight each 200 epochs
        if epoch % 200 == 0:
            torch.save(vae.state_dict(), "param/weight_"+str(epoch))
            torch.save(vae.state_dict(), "results/" + run_time + "/train/weight_" + str(epoch))

        # 最も性能のいいモデルを保存
        if train_loss_sum<train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(vae.state_dict(), "results/" + run_time + "/train/weight")

        print("\n")
    
    writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='訓練するプログラム')
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--classifier',action='store_true')
    parser.add_argument('--condition', action='store_true')
    parser.add_argument('--seq_condition', action='store_true')
    parser.add_argument('--use_model')

    args = parser.parse_args()
    if args.use_model == 'LSTM' or args.use_model == 'lstm':
        if args.condition:
            conditional_train(args)
        elif args.seq_condition:
            train_with_sequential_conditions(args)
        else:
            train(args)
    elif args.use_model == 'GCN' or args.use_model == 'gcn':
        gcn_train(args)
