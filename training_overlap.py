import os
import time
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from datas.bias_data import BiasArray, bias_data_collator, read_data
from inference import valid_performance
from models.model import DetectionModel, weights_init
from params.overlap_params import parse_args


def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def trainer(x_label, y_label, x_auto, y_auto, x_valid, y_valid, fold_no, args):
    # data setting:
    label_set = BiasArray(x_label, y_label)
    auto_set = BiasArray(x_auto, y_auto)
    valid_set = BiasArray(x_valid, y_valid)
    data_collator = lambda b: bias_data_collator(b, max_sent=args.max_sent, device_no=args.device_no)
    label_loader = DataLoader(label_set, batch_size=args.bs, shuffle=True, collate_fn=data_collator)
    auto_label_loader = DataLoader(auto_set, batch_size=args.auto_bs, shuffle=True, collate_fn=data_collator)

    label_iter = iter(label_loader)
    auto_label_iter = iter(auto_label_loader)

    # model setting:
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

    for step in range(args.start_step, args.total_step + 1):
        try:
            x_l, y_l = label_iter.next()
        except:
            label_iter = iter(label_loader)
            x_l, y_l = label_iter.next()

            acc, recall, precision, f1 = valid_performance(net=detection_model,
                                                           data_set=valid_set,
                                                           batch_size=args.bs,
                                                           max_sent=args.max_sent,
                                                           device_no=args.device_no)
            if acc > score_dict["acc_mode_acc"]:
                model_file_path = os.path.join(args.model_dir, f"model_vacc{acc:.4f}_fold{fold_no}_step{step}.pth")
                torch.save(detection_model.state_dict(), model_file_path)
                print(f"save model at step: {step}, fold_no: {fold_no}, path: {model_file_path}")
                score_dict['acc_mode_f1'] = f1
                score_dict['acc_mode_acc'] = acc
                score_dict['acc_mode_recall'] = recall
                score_dict['acc_mode_precision'] = precision
                score_dict['acc_path'] = model_file_path
            if f1 > score_dict["f1_mode_f1"]:
                model_file_path = os.path.join(args.model_dir, f"model_vf1{f1:.4f}_fold{fold_no}_step{step}.pth")
                torch.save(detection_model.state_dict(), model_file_path)
                print(f"save model at step: {step}, fold_no: {fold_no}, path: {model_file_path}")
                score_dict['f1_mode_f1'] = f1
                score_dict['f1_mode_acc'] = acc
                score_dict['f1_mode_recall'] = recall
                score_dict['f1_mode_precision'] = precision
                score_dict['f1_path'] = model_file_path

        try:
            x_auto, y_auto = auto_label_iter.next()
        except:
            auto_label_iter = iter(auto_label_loader)
            x_auto, y_auto = auto_label_iter.next()

            acc, recall, precision, f1 = valid_performance(net=detection_model,
                                                           data_set=valid_set,
                                                           batch_size=args.bs,
                                                           max_sent=args.max_sent,
                                                           device_no=args.device_no)
            if acc > score_dict["acc_mode_acc"]:
                model_file_path = os.path.join(args.model_dir, f"model_vacc{acc:.4f}_fold{fold_no}_step{step}.pth")
                torch.save(detection_model.state_dict(), model_file_path)
                print(f"save model at step: {step}, fold_no: {fold_no}, path: {model_file_path}")
                score_dict['acc_mode_f1'] = f1
                score_dict['acc_mode_acc'] = acc
                score_dict['acc_mode_recall'] = recall
                score_dict['acc_mode_precision'] = precision
                score_dict['acc_path'] = model_file_path
            if f1 > score_dict["f1_mode_f1"]:
                model_file_path = os.path.join(args.model_dir, f"model_vf1{f1:.4f}_fold{fold_no}_step{step}.pth")
                torch.save(detection_model.state_dict(), model_file_path)
                print(f"save model at step: {step}, fold_no: {fold_no}, path: {model_file_path}")
                score_dict['f1_mode_f1'] = f1
                score_dict['f1_mode_acc'] = acc
                score_dict['f1_mode_recall'] = recall
                score_dict['f1_mode_precision'] = precision
                score_dict['f1_path'] = model_file_path

        optim.zero_grad()
        output_l = detection_model(x_l)
        loss_l = args.ratio_label * criterion(output_l, y_l)
        output_auto = detection_model(x_auto)
        loss_auto = args.ratio_auto * criterion(output_auto, y_auto)
        loss = loss_l + loss_auto
        loss.backward()
        optim.step()

    return score_dict


def cv_trainer(args):
    create_dir(dir_path=args.model_dir)
    start_time = time.time()
    article_list, label_list = read_data(args.elmo_label_path)
    auto_article_list, auto_label_list = read_data(args.elmo_auto_path)
    end_time = time.time()
    print(f"it costs: {(end_time - start_time) / 60} min.")

    kfold = StratifiedKFold(n_splits=args.n_split, shuffle=True, random_state=args.seed)
    score_dict_list = []

    for fold_no, (train_idx, test_idx) in enumerate(kfold.split(article_list, label_list)):
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
        score_dict = trainer(x_label=article_train,
                             y_label=label_train,
                             x_auto=auto_article_list,
                             y_auto=auto_label_list,
                             x_valid=article_valid,
                             y_valid=label_valid,
                             fold_no=fold_no,
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