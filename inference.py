import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from datas.bias_data import bias_data_collator


def predict_res(net, data_loader):
    y_prediction_arr = []

    net.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            y_prediction = net(x)
            y_prediction_arr.extend(y_prediction.tolist())
    y_prediction_arr = np.asarray(y_prediction_arr)

    net.train()

    return y_prediction_arr


def valid_performance(net, data_set, batch_size, max_sent, device_no=-1):
    data_collator = lambda b: bias_data_collator(b, max_sent=max_sent, device_no=device_no)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size,
                             collate_fn=data_collator)
    y_prediction_arr = predict_res(net, data_loader)  # return probability
    y_pred = y_prediction_arr > 0.5
    y_pred = y_pred.astype('int32')
    y_true = np.array(data_set.label_list).astype('int32')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = (tn + tp) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (recall * precision) / (recall + precision)
    return acc, recall, precision, f1
