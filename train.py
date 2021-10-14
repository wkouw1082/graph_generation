# from data_input import GCNDataset
from typing_extensions import runtime

from numpy.core.fromnumeric import mean

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
import os

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

    # 必須ディレクトリの作成
    required_dirs = ["dataset", "param", "results"]
    remove_dirs = []
    for dir in required_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        required_dirs.remove(dir)
    if len(required_dirs) > 0:
        utils.make_dir(required_dirs)

    # results内のディレクトリの候補を作成
    result_dirs = ["results/"+run_time, "results/"+run_time+"/train", "results/"+run_time+"/eval", "results/"+run_time+"/visualize"]
    train_dir = "./" + result_dirs[1] + "/"
    remove_dirs = []
    for dir in result_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        result_dirs.remove(dir)

    # preprocess
    if is_preprocess:
        print("start preprocess...")
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
    # train_conditional = torch.cat([train_conditional for _  in range(train_dataset.shape[1])],dim=1)
    # valid_conditional = torch.cat([valid_conditional for _  in range(valid_dataset.shape[1])],dim=1)

    train_dataset = torch.cat((train_dataset,train_conditional),dim=2)
    valid_dataset = torch.cat((valid_dataset,valid_conditional),dim=2)
    print(train_dataset[1,:,-1*condition_size:])

    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("conditional size: %d"%(conditional_size))
    print("--------------")

    # model_param load
    model_param = utils.load_model_param(file_path=args.model_param)
    print(f"model_param = {model_param}")

    vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param, device)
    vae = utils.try_gpu(device,vae)

    opt = optim.Adam(vae.parameters(), lr=0.001)


    train_data_num = train_dataset.shape[0]
    train_label_args = torch.LongTensor(list(range(train_data_num)))
    valid_data_num = valid_dataset.shape[0]
    valid_label_args = torch.LongTensor(list(range(valid_data_num)))

    train_dl = DataLoader(
            TensorDataset(train_label_args, train_dataset),\
            shuffle=True, batch_size=model_param["batch_size"])
    valid_dl = DataLoader(
            TensorDataset(valid_label_args, valid_dataset),\
            shuffle=False, batch_size=model_param["batch_size"])
    

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
    
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="mean")
    encoder_criterion = self_loss.Encoder_Loss()
    timestep=0
    best_epoch = 0

    print("start conditional train...")

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

        # make result_dirs once
        if len(result_dirs) > 0:
            utils.make_dir(result_dirs)
            result_dirs = []

        # output loss/acc transition
        utils.time_draw(range(epoch), train_loss, train_dir + "train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), train_acc, train_dir + "train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        for key in keys+["encoder"]:
            utils.time_draw(range(epoch), {key: train_loss[key]}, train_dir + "train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_loss, train_dir + "valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_acc, train_dir + "valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

        train_loss_sums.append(train_loss_sum/train_data_num)
        valid_loss_sums.append(valid_loss_sum/valid_data_num)
        utils.time_draw(
                range(epoch),
                {"train": train_loss_sums, "valid": valid_loss_sums},
                train_dir + "loss_transition.png", xlabel="Epoch", ylabel="Loss")

        # output weight each 1000 epochs
        if epoch % 1000 == 0:
            torch.save(vae.state_dict(), "param/weight_"+str(epoch))
            torch.save(vae.state_dict(), train_dir + "weight_" + str(epoch))

        # 最も性能のいいモデルを保存
        if train_loss_sum<train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(vae.state_dict(), train_dir + "weight")
            best_epoch = epoch

        print("\n")
    print(f"best weight epoch = {best_epoch}")    
    writer.close()

def train(args):
    is_preprocess = args.preprocess
    is_classifier = args.classifier

    # device = utils.get_gpu_info()
    device = 'cpu'

    # recreate directory
    if utils.is_dir_existed("train_result"):
        print("delete file...")
        print("- train_result")
        shutil.rmtree("./train_result")

    # 必須ディレクトリの作成
    required_dirs = ["dataset", "param", "results"]
    remove_dirs = []
    for dir in required_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        required_dirs.remove(dir)
    if len(required_dirs) > 0:
        utils.make_dir(required_dirs)

    # results内のディレクトリの候補を作成
    result_dirs = ["results/"+run_time, "results/"+run_time+"/train", "results/"+run_time+"/eval", "results/"+run_time+"/visualize"]
    train_dir = "./" + result_dirs[1] + "/"
    remove_dirs = []
    for dir in result_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        result_dirs.remove(dir)

    # preprocess
    if is_preprocess:
        print("start preprocess...")
        shutil.rmtree("dataset")
        required_dirs = ["dataset", "dataset/train", "dataset/valid"]
        utils.make_dir(required_dirs)
        pp.preprocess_not_conditional(train_generate_detail, valid_generate_detail)

    # data load
    train_dataset = joblib.load("dataset/train/onehot")
    train_label = joblib.load("dataset/train/label") 
    valid_dataset = joblib.load("dataset/valid/onehot")
    valid_label = joblib.load("dataset/valid/label")
    time_size, node_size, edge_size, max_sequence_length = joblib.load("dataset/param")

    dfs_size = 2*time_size+2*node_size+edge_size
    dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]
    dfs_split_list = []

    tmp_split = 0
    for size in dfs_size_list:
        dfs_split_list.append([tmp_split, tmp_split + size])
        tmp_split += size
        

    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("--------------")
    
    # model_param load
    model_param = utils.load_model_param(file_path=args.model_param)
    print(f"model_param = {model_param}")

    vae = model.VAENonConditional(dfs_size, time_size, node_size, edge_size, model_param, device)
    vae = utils.try_gpu(device,vae)
    opt = optim.Adam(vae.parameters(), lr=0.001)


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

    # pos_weight = torch.ones((max_sequence_length, dfs_size)).cuda(device)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    encoder_criterion = self_loss.Encoder_Loss()
    timestep=0
    best_epoch = 0

    for epoch in range(1, epochs+1):
        print("Epoch: [%d/%d]:"%(epoch, epochs))

        # train
        print("train:")
        current_train_loss = {key:[] for key in keys+["encoder"]}
        current_train_acc = {key:[] for key in keys}
        train_acc = []
        train_loss_sum = 0
        for i, (args, datas) in enumerate(train_dl, 1):
            if i%100==0:
                print("step: [%d/%d]"%(i, train_data_num))
            vae.train()
            opt.zero_grad()
            datas = utils.try_gpu(device,datas)

            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            mu, sigma, outputs, *result= vae(datas, timestep)
            # mu, sigma, outputs = vae(datas, word_drop=word_drop_rate)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            loss = encoder_loss
            current_train_loss["encoder"].append(encoder_loss.item())
            mask_tensors = torch.Tensor()
            # 各バッチごとにpaddingされた位置をgetする
            for i in range(datas.size(0)):
                masked_index = (datas[i] == ignore_label).nonzero(as_tuple=False).tolist()
                if len(masked_index) == 0:
                    masked_index = len(datas[i])
                else:
                    masked_index = masked_index[0][0]


                no_mask_tensor = torch.ones((1, masked_index, dfs_size))
                mask_tensor = torch.zeros((1, len(datas[i])-masked_index, dfs_size))
                mask_tensor = torch.cat((no_mask_tensor, mask_tensor), dim=1)
                mask_tensors = torch.cat((mask_tensors, mask_tensor), dim=0)

            datas = datas*mask_tensors
            outputs = outputs*mask_tensors
            bce_loss = criterion(outputs, datas)
            loss = encoder_loss + bce_loss
            train_acc.append(utils.classification_metric(outputs, datas))
            train_loss_sum+=loss.item()

            loss.backward()
            opt.step()

        # loss, acc save
        print("----------------------------")
        print('loss    : {}'.format(loss))
        print('accracy : {}'.format(mean(train_acc)))
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
            mu, sigma, outputs, *result = vae(datas)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            current_valid_loss["encoder"].append(encoder_loss.item())
            mask_tensors = torch.Tensor()
            # 各バッチごとにpaddingされた位置をgetする
            for i in range(datas.size(0)):
                masked_index = (datas[i] == ignore_label).nonzero(as_tuple=False).tolist()
                if len(masked_index) == 0:
                    masked_index = len(datas[i])
                else:
                    masked_index = masked_index[0][0]

                no_mask_tensor = torch.ones((1, masked_index, dfs_size))
                mask_tensor = torch.zeros((1, len(datas[i])-masked_index, dfs_size))
                mask_tensor = torch.cat((no_mask_tensor, mask_tensor), dim=1)
                mask_tensors = torch.cat((mask_tensors, mask_tensor), dim=0)

            datas = datas*mask_tensors
            outputs = outputs*mask_tensors
            loss = encoder_loss
            loss += criterion(outputs, datas)
            accuracy = utils.classification_metric(outputs, datas)

            valid_loss_sum+=loss.item()

        # loss, acc save
        print("----------------------------")
        print('loss : {}'.format(loss))
        print('accracy : {}'.format(accuracy))
        print("----------------------------")

        # make result_dirs once
        if len(result_dirs) > 0:
            utils.make_dir(result_dirs)
            result_dirs = []

        # output loss/acc transition
        utils.time_draw(range(epoch), train_loss, train_dir + "train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        # utils.time_draw(range(epoch), train_acc, train_dir + "train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        for key in keys+["encoder"]:
            utils.time_draw(range(epoch), {key: train_loss[key]}, train_dir + "train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_loss, train_dir + "valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_acc, train_dir + "valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

        train_loss_sums.append(train_loss_sum/train_data_num)
        valid_loss_sums.append(valid_loss_sum/valid_data_num)
        utils.time_draw(
                range(epoch),
                {"train": train_loss_sums, "valid": valid_loss_sums},
                train_dir + "loss_transition.png", xlabel="Epoch", ylabel="Loss")

        # output weight each 1000 epochs
        if epoch % 1000 == 0:
            torch.save(vae.state_dict(), "param/weight_"+str(epoch))
            torch.save(vae.state_dict(), train_dir + "weight_" + str(epoch))

        # output weight if train loss is min
        if train_loss_sum<train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(vae.state_dict(), "param/weight")
            torch.save(vae.state_dict(), train_dir + "weight")
            best_epoch = epoch
        print("\n")
    print(f"best weight epoch = {best_epoch}")

def gcn_train(args):

    # 必須ディレクトリの作成
    required_dirs = ["dataset", "param", "results"]
    remove_dirs = []
    for dir in required_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        required_dirs.remove(dir)
    if len(required_dirs) > 0:
        utils.make_dir(required_dirs)

    # results内のディレクトリの候補を作成
    result_dirs = ["results/"+run_time, "results/"+run_time+"/train", "results/"+run_time+"/eval", "results/"+run_time+"/visualize"]
    train_dir = "./" + result_dirs[1] + "/"
    remove_dirs = []
    for dir in result_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        result_dirs.remove(dir)


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
    model_param = utils.load_model_param(file_path=args.model_param)
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

        # make result_dirs once
        if len(result_dirs) > 0:
            utils.make_dir(result_dirs)
            result_dirs = []

        # output loss/acc transition
        utils.time_draw(range(epoch), train_loss, train_dir + "train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), train_acc, train_dir + "train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        for key in keys+["encoder"]:
            utils.time_draw(range(epoch), {key: train_loss[key]}, train_dir + "train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_loss, train_dir + "valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_acc, train_dir + "valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")

        train_loss_sums.append(train_loss_sum/len(train_dataset))
        valid_loss_sums.append(valid_loss_sum/len(valid_dataset))
        utils.time_draw(
                range(epoch),
                {"train": train_loss_sums, "valid": valid_loss_sums},
                train_dir + "loss_transition.png", xlabel="Epoch", ylabel="Loss")

        # output weight each 1000 epochs
        # if epoch % 1000 == 0:
        if epoch % 200 == 0:
            torch.save(vae.state_dict(), "param/weight_"+str(epoch))
            torch.save(vae.state_dict(), train_dir + "weight_" + str(epoch))

        # 最も性能のいいモデルを保存
        if train_loss_sum<train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(vae.state_dict(), train_dir + "weight")

        print("\n")
    
    writer.close()

def train_with_sequential_conditions(args):
    """conditionをlstmに入力した順に逐次作成して付与した学習
    
    lstmに５タプルを順に入力することで蓄積されたdfsコードからnetworkx形式のグラフを構築し、
    ５タプルに加えるグラフの特徴量をそのグラフから計算して付与する。
    
    // Composition of one dfs_code
    *-------------*-------------*---------*---------*---------*-------------*
    | timestamp_u | timestamp_v | label_u | label_v | label_e | graph_param |
    *-------------*-------------*---------*---------*---------*-------------*

    Args:
        args (argparse.ArgumentParser.parse_args()): preprocessや使用modelなどデータ
    """
    # tensorboard_dir_name = input("tensorboardのディレクトリ名を入力：")
    tensorboard_dir_name = args.log_name
    writer = SummaryWriter(log_dir="./logs/" + tensorboard_dir_name + "/")

    is_preprocess = args.preprocess
    is_classifier = args.classifier

    device = utils.get_gpu_info()

    # recreate directory
    if utils.is_dir_existed("train_result"):
        print("delete file...")
        print("- train_result")
        shutil.rmtree("./train_result")

    # 必須ディレクトリの作成
    required_dirs = ["dataset", "param", "results"]
    remove_dirs = []
    for dir in required_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        required_dirs.remove(dir)
    if len(required_dirs) > 0:
        utils.make_dir(required_dirs)
        
    # results内のディレクトリの候補を作成
    result_dirs = ["results/"+run_time, "results/"+run_time+"/train", "results/"+run_time+"/eval", "results/"+run_time+"/visualize"]
    train_dir = "./" + result_dirs[1] + "/"
    remove_dirs = []
    for dir in result_dirs:
        if os.path.exists("./" + dir):
            remove_dirs.append(dir)
    for dir in remove_dirs:
        result_dirs.remove(dir)
    
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

    ## なんか追加（BCE実装に伴って）
    dfs_split_list = []
    tmp_split = 0
    for size in dfs_size_list:
        dfs_split_list.append([tmp_split, tmp_split + size])
        tmp_split += size


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
    # train_dataset = rewrite_dataset_condition(train_dataset, time_size, dfs_size, rewrited_condition_dump_name=f"train_conditions", flag_dump=True)
    # joblib.dump(train_dataset, "./train_dataset")
    with open("./train_dataset", "rb") as f:
        train_dataset = joblib.load(f)

    ## rewrite condition of valid_dataset
    # valid_dataset = rewrite_dataset_condition(valid_dataset, time_size, lessdfs_size, rewrited_condition_dump_name=f"valid_conditions", flag_dump=True)
    # joblib.dump(valid_dataset, "./valid_dataset")
    with open("./valid_dataset", "rb") as f:
        valid_dataset = joblib.load(f)
    
    
    print("--------------")
    print("time size: %d"%(time_size))
    print("node size: %d"%(node_size))
    print("edge size: %d"%(edge_size))
    print("conditional size: %d"%(conditional_size))
    print("--------------")

    # model_param load
    model_param = utils.load_model_param(file_path=args.model_param)
    print(f"model_param = {model_param}")

    # create VAE model
    vae = model.VAE(dfs_size, time_size, node_size, edge_size, model_param, device)
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
    train_loss = {key:[] for key in keys+["encoder"]}
    # train_acc = {'tu': [], 'tv': [], 'lu': [], 'lv': [], 'le': []}
    train_acc = {key:[] for key in keys}
    train_loss_sums = []
    valid_loss = {key:[] for key in keys+["encoder"]}
    valid_acc = {key:[] for key in keys}
    valid_loss_sums = []
    train_min_loss = 1e10
    
    # set Loss        
    ## tu, tv, lu, lv, leの誤差関数
    if args.loss == "BCE" or args.loss == "bce":
        print("loss : BCE")
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss == "CE" or args.loss == "ce":
        print("loss : Cross Entropy")
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="sum")
    else:
        print("[ERROR] args.loss を正しく設定してください。")
        exit()
    ## encoderの誤差関数（KL-divercity）
    encoder_criterion = self_loss.Encoder_Loss()
    timestep=0
    ## best weightのepoch
    best_epoch = 0

    print("start train_with_sequential_conditions ...")
    for epoch in range(1, epochs+1):
        print("Epoch: [%d/%d]:"%(epoch, epochs))

        # train
        print("train:")
        ## current_train_loss = {'tu': [], 'tv': [], 'lu': [], 'lv': [], 'le': [], 'encoder': []}
        current_train_loss = {key:[] for key in keys+["encoder"]}
        ## current_train_acc = {'tu': [], 'tv': [], 'lu': [], 'lv': [], 'le': []}
        current_train_acc = {key:[] for key in keys}
        train_loss_sum = 0
        for i, (args_i, datas) in enumerate(train_dl, 1):
            ## args_i = [バッチサイズ], 値はどのグラフを参照するかを示すindex(0 ~ グラフ数-1)
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
            mu, sigma, *result = vae(datas, word_drop=word_drop_rate)
            
            # print(f"mu ({mu.shape}) = ")
            # print(f"sigma ({sigma.shape}) = ")
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            # print(f"encoder_loss ({encoder_loss.shape}) = {encoder_loss}")
            ## encoder_loss.item() = loss値
            # current_train_loss["encoder"].append(encoder_loss.item())
            current_train_loss["encoder"].append(encoder_loss.mean().item())
            # loss = encoder_loss
            loss = torch.Tensor()

            if args.loss == "BCE" or args.loss == "bce":
                outputs = torch.cat((result[0], result[1], result[2], result[3], result[4]), dim=2)
                accuracy = utils.classification_metric(outputs, datas)
                writer.add_scalar(f"train/train_accuracy", accuracy, epoch)

            for j, pred in enumerate(result):
                ## pred = [バッチサイズ, ５タプルの最大数, ５タプルの各ラベルのサイズ], 確率値(0 ~ 1)
                current_key = keys[j]
                
                # loss calc
                if args.loss == "BCE" or args.loss == "bce":
                    correct_split = dfs_split_list[j]
                    correct = datas[:,:,correct_split[0]:correct_split[1]]
                    correct = utils.try_gpu(device,correct)
                    # print(f"pred ({pred.shape}) = ")
                    # print(f"correct ({correct.shape}) = ")
                    ## pred ([21, 357, 51(34,2)]),  correct ([21, 357, 51(34,2)])
                    ## tmp_loss ([21, 357, 51(24,2)])
                    tmp_loss = criterion(pred, correct)
                    # print(f"tmp_loss ({tmp_loss.shape}) = ")
                elif args.loss == "CE" or args.loss == "ce":
                    correct = train_label[j]
                    ## correct = [バッチサイズ, ５タプルの最大数], ５タプルの各ラベルのid or ignore_label
                    correct = correct[args_i]
                    correct = utils.try_gpu(device,correct)
                    ## pred_transpose(2, 1) = [バッチサイズ, ５タプルの各サイズ, ５タプルの最大数], 確率値(0 ~ 1)
                    tmp_loss = criterion(pred.transpose(2, 1), correct)
                else:
                    print("[ERROR] args.loss を正しく設定してください。")
                    exit()

                ## cross entropy loss
                # loss+=tmp_loss
                ## binary cross entropy loss
                if j == 0:
                    loss = tmp_loss
                else:
                    loss = torch.cat((loss, tmp_loss), dim=2)

                # save
                ## CE
                # current_train_loss[current_key].append(tmp_loss.item())
                ## BCE
                current_train_loss[current_key].append(tmp_loss.sum(axis=2).sum(axis=1).mean().item())

                
                if args.loss == "CE" or args.loss == "ce":
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
                degree_loss = criterion(degree.squeeze(), train_classifier_correct[args_i][:, 0])
                cluster_loss = criterion(cluster.squeeze(), train_classifier_correct[args_i][:, 1])
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

            # calc CVAE loss
            # print(f"loss ({loss.shape}) = ")
            bce_loss = loss.sum(axis=2).sum(axis=1)
            # print(f"bce loss ({bce_loss.shape}) = {bce_loss}")
            loss = (encoder_loss + bce_loss).mean()
            # print(f"CVAE loss = {loss}")
            
            # backpropagation
            loss.backward()
            train_loss_sum+=loss.item()
            del loss
            opt.step()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), model_param["clip_th"])

        # make result_dirs once
        if len(result_dirs) > 0:
            utils.make_dir(result_dirs)
            result_dirs = []

        # loss, acc save
        print("----------------------------")
        if args.loss == "CE" or args.loss == "ce":
            for key in keys:
                loss = np.average(current_train_loss[key])
                train_loss[key].append(loss)
                
                acc = np.average(current_train_acc[key])
                train_acc[key].append(acc)

                print(" %s:"%(key))
                print("     loss:%lf"%(loss))
                print("     acc:%lf"%(acc))

                # dump loss to csv_file
                try:
                    with open(f"{train_dir}train_{key}_loss.csv", mode="x") as f:
                        f.write(f'{epoch},{loss}\n')
                except:
                    with open(f"{train_dir}train_{key}_loss.csv", mode="a") as f:
                        f.write(f'{epoch},{loss}\n')
                # dump acc to csv_file
                try:
                    with open(f"{train_dir}train_{key}_acc.csv", mode="x") as f:
                        f.write(f'{epoch},{acc}\n')
                except:
                    with open(f"{train_dir}train_{key}_acc.csv", mode="a") as f:
                        f.write(f'{epoch},{acc}\n')

        elif args.loss == "BCE" or args.loss == "bce":
            for key in keys:
                loss = np.average(current_train_loss[key])
                # loss = current_train_loss[key]
                train_loss[key].append(loss)

                print(" %s:"%(key))
                print("     loss:%lf"%(loss))

                # dump loss to csv_file
                try:
                    with open(f"{train_dir}train_{key}_loss.csv", mode="x") as f:
                        f.write(f'{epoch},{loss}\n')
                except:
                    with open(f"{train_dir}train_{key}_loss.csv", mode="a") as f:
                        f.write(f'{epoch},{loss}\n')
            print(f"accuracy:{accuracy}")
            # dump acc to csv_file
            try:
                with open(f"{train_dir}train_5tuples_acc.csv", mode="x") as f:
                    f.write(f'{epoch},{accuracy}\n')
            except:
                with open(f"{train_dir}train_5tuples_acc.csv", mode="a") as f:
                    f.write(f'{epoch},{accuracy}\n')

        ekey = "encoder"
        loss = np.average(current_train_loss[ekey])
        train_loss[ekey].append(loss)
        print(" %s:"%(ekey))
        print("     loss:%lf"%(loss))

        # dump loss to csv_file
        try:
            with open(f"{train_dir}train_{ekey}_loss.csv", mode="x") as f:
                f.write(f'{epoch},{loss}\n')
        except:
            with open(f"{train_dir}train_{ekey}_loss.csv", mode="a") as f:
                f.write(f'{epoch},{loss}\n')
        print("----------------------------")

        # memory free
        del current_train_loss, current_train_acc

        # valid
        print("valid:")
        current_valid_loss = {key:[] for key in keys+["encoder"]}
        current_valid_acc = {key:[] for key in keys}
        valid_loss_sum = 0
        for i, (args_i, datas) in enumerate(valid_dl):
            if i%1000==0:
                print("step: [%d/%d]"%(i, valid_data_num))
            vae.eval()
            opt.zero_grad()
            datas = utils.try_gpu(device,datas)
            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            mu, sigma, *result = vae(datas)
            encoder_loss = encoder_criterion(mu, sigma)*encoder_bias
            current_valid_loss["encoder"].append(encoder_loss.mean().item())
            loss = encoder_loss

            if args.loss == "BCE" or args.loss == "bce":
                outputs = torch.cat((result[0], result[1], result[2], result[3], result[4]), dim=2)
                accuracy = utils.classification_metric(outputs, datas)
                writer.add_scalar(f"valid/valid_accuracy", accuracy, epoch)

            for j, pred in enumerate(result):
                current_key = keys[j]

                # loss calc
                if args.loss == "BCE" or args.loss == "bce":
                    correct_split = dfs_split_list[j]
                    correct = datas[:,:,correct_split[0]:correct_split[1]]
                    correct = utils.try_gpu(device,correct)
                    # tmp_loss = criterion(pred, correct).sum(axis=0)
                    tmp_loss = criterion(pred, correct)
                elif args.loss == "CE" or args.loss == "ce":
                    correct = valid_label[j]
                    ## correct = [バッチサイズ, ５タプルの最大数], ５タプルの各ラベルのid or ignore_label
                    correct = correct[args_i]
                    correct = utils.try_gpu(device,correct)
                    ## pred_transpose(2, 1) = [バッチサイズ, ５タプルの各サイズ, ５タプルの最大数], 確率値(0 ~ 1)
                    tmp_loss = criterion(pred.transpose(2, 1), correct)
                else:
                    print("[ERROR] args.loss を正しく設定してください。")
                    exit()
                
                ## cross entropy loss
                # loss+=tmp_loss.item()
                ## binary cross entropy loss
                if j == 0:
                    loss = tmp_loss
                else:
                    loss = torch.cat((loss, tmp_loss), dim=2)

                # save
                ## CE
                # current_valid_loss[current_key].append(tmp_loss.item())
                ## BCE
                current_valid_loss[current_key].append(tmp_loss.sum(axis=2).sum(axis=1).mean().item())

                if args.loss == "CE" or args.loss == "ce":
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
                degree_loss = criterion(degree.squeeze(), valid_classifier_correct[args_i][:, 0])
                cluster_loss = criterion(cluster.squeeze(), valid_classifier_correct[args_i][:, 1])
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

            # valid_loss_sum+=loss.item()

            # calc CVAE loss
            # print(f"loss ({loss.shape}) = ")
            bce_loss = loss.sum(axis=2).sum(axis=1)
            # print(f"bce loss ({bce_loss.shape}) = {bce_loss}")
            loss = (encoder_loss + bce_loss).mean()
            # print(f"CVAE loss = {loss}")

        # loss, acc save
        print("----------------------------")
        if args.loss == "CE" or args.loss == "ce":
            for key in keys:
                loss = np.average(current_valid_loss[key])
                valid_loss[key].append(loss)
                acc = np.average(current_valid_acc[key])
                valid_acc[key].append(acc)

                print(" %s:"%(key))
                print("     loss:%lf"%(loss))
                print("     acc:%lf"%(acc))

                # dump loss to csv_file
                try:
                    with open(f"{train_dir}valid_{key}_loss.csv", mode="x") as f:
                        f.write(f'{epoch},{loss}\n')
                except:
                    with open(f"{train_dir}valid_{key}_loss.csv", mode="a") as f:
                        f.write(f'{epoch},{loss}\n')
                # dump acc to csv_file
                try:
                    with open(f"{train_dir}valid_{key}_acc.csv", mode="x") as f:
                        f.write(f'{epoch},{acc}\n')
                except:
                    with open(f"{train_dir}valid_{key}_acc.csv", mode="a") as f:
                        f.write(f'{epoch},{acc}\n')

        elif args.loss == "BCE" or args.loss == "bce":
            for key in keys:
                loss = np.average(current_valid_loss[key])
                valid_loss[key].append(loss)

                print(" %s:"%(key))
                print("     loss:%lf"%(loss))

                # dump loss to csv_file
                try:
                    with open(f"{train_dir}valid_{key}_loss.csv", mode="x") as f:
                        f.write(f'{epoch},{loss}\n')
                except:
                    with open(f"{train_dir}valid_{key}_loss.csv", mode="a") as f:
                        f.write(f'{epoch},{loss}\n')
            print(f"accuracy:{accuracy}")
            # dump acc to csv_file
            try:
                with open(f"{train_dir}valid_5tuples_acc.csv", mode="x") as f:
                    f.write(f'{epoch},{accuracy}\n')
            except:
                with open(f"{train_dir}valid_5tuples_acc.csv", mode="a") as f:
                    f.write(f'{epoch},{accuracy}\n')

        ekey = "encoder"
        loss = np.average(current_valid_loss[ekey])
        valid_loss[ekey].append(loss)
        print(" %s:"%(ekey))
        print("     loss:%lf"%(loss))

        # dump loss to csv_file
        try:
            with open(f"{train_dir}valid_{ekey}_loss.csv", mode="x") as f:
                f.write(f'{epoch},{loss}\n')
        except:
            with open(f"{train_dir}valid_{ekey}_loss.csv", mode="a") as f:
                f.write(f'{epoch},{loss}\n')

        print("----------------------------")

        # output loss/acc transition
        utils.time_draw(range(epoch), train_loss, train_dir + "train_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), train_acc, train_dir + "train_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        writer.add_scalars("train/train_each_loss_transition", {"tu":train_loss["tu"][-1],
                                                                "tv":train_loss["tv"][-1],
                                                                "lu":train_loss["lu"][-1],
                                                                "lv":train_loss["lv"][-1],
                                                                "le":train_loss["le"][-1],
                                                                "encoder":train_loss["encoder"][-1]}, epoch)
        if args.loss == "ce" or args.loss == "CE":
            writer.add_scalars("train/train_each_acc_transition", {"tu":train_acc["tu"][-1],
                                                                "tv":train_acc["tv"][-1],
                                                                "lu":train_acc["lu"][-1],
                                                                "lv":train_acc["lv"][-1],
                                                                "le":train_acc["le"][-1]}, epoch)
        for key in keys+["encoder"]:
            utils.time_draw(range(epoch), {key: train_loss[key]}, train_dir + "train_%sloss_transition.png"%(key), xlabel="Epoch", ylabel="Loss")
            # each loss
            writer.add_scalar(f"train/{key}_loss", train_loss[key][-1], epoch)
        utils.time_draw(range(epoch), valid_loss, train_dir + "valid_loss_transition.png", xlabel="Epoch", ylabel="Loss")
        utils.time_draw(range(epoch), valid_acc, train_dir + "valid_acc_transition.png", xlabel="Epoch", ylabel="Accuracy")
        writer.add_scalars("valid/valid_each_loss_transition", {"tu":valid_loss["tu"][-1],
                                                                "tv":valid_loss["tv"][-1],
                                                                "lu":valid_loss["lu"][-1],
                                                                "lv":valid_loss["lv"][-1],
                                                                "le":valid_loss["le"][-1],
                                                                "encoder":valid_loss["encoder"][-1]}, epoch)
        if args.loss == "ce" or args.loss == "CE":
            writer.add_scalars("valid/valid_each_acc_transition", {"tu":valid_acc["tu"][-1],
                                                                "tv":valid_acc["tv"][-1],
                                                                "lu":valid_acc["lu"][-1],
                                                                "lv":valid_acc["lv"][-1],
                                                                "le":valid_acc["le"][-1]}, epoch)

        train_loss_sums.append(train_loss_sum/train_data_num)
        valid_loss_sums.append(valid_loss_sum/valid_data_num)
        utils.time_draw(
                range(epoch),
                {"train": train_loss_sums, "valid": valid_loss_sums},
                train_dir + "loss_transition.png", xlabel="Epoch", ylabel="Loss")
        
        # all sums loss
        writer.add_scalar("train/train_sum_loss", train_loss_sums[-1], epoch)
        writer.add_scalar("valid/vallid_sum_loss", valid_loss_sums[-1], epoch)
        writer.add_scalars("sum_loss", {"train sum loss":train_loss_sums[-1], "valid sum loss":valid_loss_sums[-1]}, epoch)
        # dump all sums loss to csv_file
        try:
            with open(f"{train_dir}train_loss_sums.csv", mode="x") as f:
                f.write(f'{epoch},{train_loss_sums[-1]}\n')
        except:
            with open(f"{train_dir}train_loss_sums.csv", mode="a") as f:
                f.write(f'{epoch},{train_loss_sums[-1]}\n')
        try:
            with open(f"{train_dir}valid_loss_sums.csv", mode="x") as f:
                f.write(f'{epoch},{valid_loss_sums[-1]}\n')
        except:
            with open(f"{train_dir}valid_loss_sums.csv", mode="a") as f:
                f.write(f'{epoch},{valid_loss_sums[-1]}\n')

        # output weight each 1000 epochs
        if epoch % 100 == 0:
            torch.save(vae.state_dict(), "param/weight_"+str(epoch))
            torch.save(vae.state_dict(), train_dir + "weight_" + str(epoch))

        # 最も性能のいいモデルを保存
        if train_loss_sum<train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(vae.state_dict(), train_dir + "weight")
            best_epoch = epoch
            print(f"update best_epoch = {best_epoch}")

        print("\n")
    print(f"best_weight_epoch = {best_epoch}")
    writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='訓練するプログラム')
    parser.add_argument('--preprocess',action='store_true')
    parser.add_argument('--classifier',action='store_true')
    parser.add_argument('--condition', action='store_true')
    parser.add_argument('--seq_condition', action='store_true')

    parser.add_argument('--use_model', default="lstm")
    parser.add_argument('--model_param')
    parser.add_argument('--result_dir')

    parser.add_argument('--log_name')
    parser.add_argument('--loss', default="BCE")


    args = parser.parse_args()
    if args.use_model == 'LSTM' or args.use_model == 'lstm':
        if args.condition:
            conditional_train(args)
        elif args.seq_condition:
            print("calling train_with_sequential_conditions() ...")
            train_with_sequential_conditions(args)
        else:
            train(args)
    elif args.use_model == 'GCN' or args.use_model == 'gcn':
        gcn_train(args)
