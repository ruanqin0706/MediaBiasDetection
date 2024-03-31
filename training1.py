import os
import pickle
import shutil
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from datas.bias_data import bias_data_collator, read_data, BiasArray
from inference import valid_performance
from models.model import DetectionModel, weights_init
from params.bias_params import parse_args


def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def trainer(x_train, y_train, x_valid, y_valid, x, y, fold_no, args):
    train_set = BiasArray(x_train, y_train)
    valid_set = BiasArray(x_valid, y_valid)
    data_collator = lambda b: bias_data_collator(b, max_sent=args.max_sent, device_no=args.device_no)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, collate_fn=data_collator)

    data_set = BiasArray(x, y)
    data_collator = lambda b: bias_data_collator(b, max_sent=args.max_sent, device_no=args.device_no)
    data_loader = DataLoader(data_set, batch_size=args.bs, shuffle=True, collate_fn=data_collator)

    detection_model = DetectionModel()
    detection_model.apply(weights_init)
    if args.device_no > -1:
        detection_model = detection_model.cuda(device=args.device_no)

    # create loss
    criterion = nn.BCELoss()

    # create optimizer
    optim = torch.optim.Adam(detection_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    score_dict = {
        'f1_mode_f1': 0.0, 'f1_mode_acc': 0.0, 'f1_mode_recall': 0.0, 'f1_mode_precision': 0.0, 'f1_path': '',
        'acc_mode_f1': 0.0, 'acc_mode_acc': 0.0, 'acc_mode_recall': 0.0, 'acc_mode_precision': 0.0, 'acc_path': '',
    }
    for epoch_no in range(args.max_epoch):
        detection_model.train()
        for step, data_batch in enumerate(train_loader):
            optim.zero_grad()
            output = detection_model(data_batch[0])
            loss = criterion(output, data_batch[1])
            loss.backward()
            optim.step()

        acc, recall, precision, f1 = valid_performance(net=detection_model,
                                                       data_set=valid_set,
                                                       batch_size=args.bs,
                                                       max_sent=args.max_sent,
                                                       device_no=args.device_no)

        if acc > score_dict["acc_mode_acc"]:
            model_file_path = os.path.join(args.model_dir, f"model_vacc{acc:.4f}_fold{fold_no}_ep{epoch_no}.pth")
            torch.save(detection_model.state_dict(), model_file_path)
            print(f"save model at epoch: {epoch_no}, fold_no: {fold_no}, path: {model_file_path}")
            score_dict['acc_mode_f1'] = f1
            score_dict['acc_mode_acc'] = acc
            score_dict['acc_mode_recall'] = recall
            score_dict['acc_mode_precision'] = precision
            score_dict['acc_path'] = model_file_path
        if f1 > score_dict["f1_mode_f1"]:
            model_file_path = os.path.join(args.model_dir, f"model_vf1{f1:.4f}_fold{fold_no}_ep{epoch_no}.pth")
            torch.save(detection_model.state_dict(), model_file_path)
            print(f"save model at epoch: {epoch_no}, fold_no: {fold_no}, path: {model_file_path}")
            score_dict['f1_mode_f1'] = f1
            score_dict['f1_mode_acc'] = acc
            score_dict['f1_mode_recall'] = recall
            score_dict['f1_mode_precision'] = precision
            score_dict['f1_path'] = model_file_path

        detection_model.train()
        for step, data_batch in enumerate(data_loader):
            optim.zero_grad()
            output = detection_model(data_batch[0])
            loss = 0.5 * criterion(output, data_batch[1])
            loss.backward()
            optim.step()

        acc, recall, precision, f1 = valid_performance(net=detection_model,
                                                       data_set=valid_set,
                                                       batch_size=args.bs,
                                                       max_sent=args.max_sent,
                                                       device_no=args.device_no)
        if acc > score_dict["acc_mode_acc"]:
            model_file_path = os.path.join(args.model_dir, f"model_vacc{acc:.4f}_fold{fold_no}_ep{epoch_no}.pth")
            torch.save(detection_model.state_dict(), model_file_path)
            print(f"save model at epoch: {epoch_no}, fold_no: {fold_no}, path: {model_file_path}")
            score_dict['acc_mode_f1'] = f1
            score_dict['acc_mode_acc'] = acc
            score_dict['acc_mode_recall'] = recall
            score_dict['acc_mode_precision'] = precision
            score_dict['acc_path'] = model_file_path
        if f1 > score_dict["f1_mode_f1"]:
            model_file_path = os.path.join(args.model_dir, f"model_vf1{f1:.4f}_fold{fold_no}_ep{epoch_no}.pth")
            torch.save(detection_model.state_dict(), model_file_path)
            print(f"save model at epoch: {epoch_no}, fold_no: {fold_no}, path: {model_file_path}")
            score_dict['f1_mode_f1'] = f1
            score_dict['f1_mode_acc'] = acc
            score_dict['f1_mode_recall'] = recall
            score_dict['f1_mode_precision'] = precision
            score_dict['f1_path'] = model_file_path
    return score_dict


def cv_trainer(args):
    create_dir(dir_path=args.model_dir)
    import time
    start_time = time.time()
    article_list, label_list = read_data(args.elmo_file_path)
    x_list, y_list = read_data("/home/ruanqin/data/media_bias/processed_data/30000elmo/byarticle.proportion_646.csv.test")
    end_time = time.time()
    print(f"it costs: {(end_time - start_time) / 60} min.")
    kfold = StratifiedKFold(n_splits=args.n_split, shuffle=True, random_state=args.seed)
    score_dict_list = []
    for idx, (train_idx, test_idx) in enumerate(kfold.split(article_list, label_list)):
        article_train = []
        label_train = []
        article_valid = []
        label_valid = []
        for idx_, (article, label) in enumerate(zip(article_list, label_list)):
            if idx_ in train_idx:
                article_train.append(article)
                label_train.append(label)
            else:
                article_valid.append(article)
                label_valid.append(label)
        print(len(article_train), len(label_train), len(article_valid), len(label_valid))
        score_dict = trainer(x_train=article_train,
                             y_train=label_train,
                             x_valid=article_valid,
                             y_valid=label_valid,
                             x=x_list,
                             y=y_list,
                             fold_no=idx,
                             args=args)
        score_dict_list.append(score_dict)

    acc_mode_f1_list = []
    acc_mode_precision_list = []
    acc_mode_recall_list = []
    acc_mode_acc_list = []
    f1_mode_f1_list = []
    f1_mode_precision_list = []
    f1_mode_recall_list = []
    f1_mode_acc_list = []
    for score_dict in score_dict_list:
        acc_mode_f1_list.append(score_dict['acc_mode_f1'])
        acc_mode_precision_list.append(score_dict['acc_mode_precision'])
        acc_mode_recall_list.append(score_dict['acc_mode_recall'])
        acc_mode_acc_list.append(score_dict['acc_mode_acc'])
        f1_mode_f1_list.append(score_dict['f1_mode_f1'])
        f1_mode_precision_list.append(score_dict['f1_mode_precision'])
        f1_mode_recall_list.append(score_dict['f1_mode_recall'])
        f1_mode_acc_list.append(score_dict['f1_mode_acc'])
    print(f"acc mode: \n"
          f"f1: {np.mean(acc_mode_f1_list)}, {np.std(acc_mode_f1_list, ddof=1)}\n"
          f"precision: {np.mean(acc_mode_precision_list)}, {np.std(acc_mode_precision_list, ddof=1)}\n"
          f"recall: {np.mean(acc_mode_recall_list)}, {np.std(acc_mode_recall_list, ddof=1)}\n"
          f"acc: {np.mean(acc_mode_acc_list)}, {np.std(acc_mode_acc_list, ddof=1)}\n")
    print(f"f1 mode: \n"
          f"f1: {np.mean(f1_mode_f1_list)}, {np.std(f1_mode_f1_list, ddof=1)}\n"
          f"precision: {np.mean(f1_mode_precision_list)}, {np.std(f1_mode_precision_list, ddof=1)}\n"
          f"recall: {np.mean(f1_mode_recall_list)}, {np.std(f1_mode_recall_list, ddof=1)}\n"
          f"acc: {np.mean(f1_mode_acc_list)}, {np.std(f1_mode_acc_list, ddof=1)}\n")

    with open(args.pickle_path, "wb") as f:
        pickle.dump(score_dict_list, f)


if __name__ == '__main__':
    args = parse_args()
    cv_trainer(args)
