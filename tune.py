import optuna
import yaml
import utils
import joblib
import model
from torch import nn
from torch import optim
import torch
import self_loss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from config import *

args = utils.get_args()
is_preprocess = args.preprocess
# preprocess
if is_preprocess:
    shutil.rmtree("dataset")
    required_dirs = ["dataset", "dataset/train", "dataset/test"]
    utils.make_dir(required_dirs)
    pp.preprocess(train_generate_detail, test_generate_detail)

# data load
train_dataset = joblib.load("dataset/train/onehot")[0]
train_label = joblib.load("dataset/train/label")
test_dataset = joblib.load("dataset/test/onehot")[0]
test_label = joblib.load("dataset/test/label")

time_size, node_size, edge_size = joblib.load("dataset/param")
dfs_size = 2*time_size+2*node_size+edge_size

print("--------------")
print("time size: %d"%(time_size))
print("node size: %d"%(node_size))
print("edge size: %d"%(edge_size))
print("--------------")

criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
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
    opt = optim.Adam(vae.parameters(), lr=lr, weight_decay=decay)

    train_data_num = train_dataset.shape[0]
    train_label_args = torch.LongTensor(list(range(train_data_num)))

    train_dl = DataLoader(
            TensorDataset(train_label_args, train_dataset),\
            shuffle=True, batch_size=batch_size)
    train_min_loss = 1e10

    keys = ["tu", "tv", "lu", "lv", "le"]

    for epoch in range(1, epochs):
        print("Epoch: [%d/%d]:"%(epoch, epochs))

        # train
        loss_sum = 0
        for i, (args, datas) in enumerate(train_dl, 1):
            if i%100==0:
                print("step: [%d/%d]"%(i, train_data_num))
            vae.train()
            opt.zero_grad()
            # mu,sigma, [tu, tv, lu, lv, le] = vae(datas)
            mu, sigma, *result = vae(datas)
            encoder_loss = encoder_criterion(mu, sigma)
            loss = encoder_loss
            for j, pred in enumerate(result):
                current_key = keys[j]
                # loss calc
                correct = train_label[j]
                correct = correct[args]
                tmp_loss = criterion(pred.transpose(2, 1), correct)
                loss+=tmp_loss
            loss.backward()
            loss_sum+=loss.item()
            opt.step()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_th)
        if train_min_loss>loss_sum:
            train_min_loss = loss_sum
        print(" loss: %lf"%(loss_sum))
    return train_min_loss

study = optuna.create_study()
study.optimize(tuning_trial, n_trials=opt_epoch)

print("--------------------------")
print(study.best_params)
print("--------------------------")

f = open("param/best_tune.yml", "w+")
f.write(yaml.dump(study.best_params))
f.close()
