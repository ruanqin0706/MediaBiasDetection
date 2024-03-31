import pickle
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from datas.bias_data import bias_data_collator, BiasArray, read_data
from inference import predict_res

from models.model import DetectionModel


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--numpy_path", type=str)
    parser.add_argument("--elmo_file_path", type=str)
    parser.add_argument("--pickle_path", type=str)
    parser.add_argument("--device_no", type=int)
    parser.add_argument("--max_sent", type=int)
    parser.add_argument("--bs", type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args.numpy_path)
    import time
    start_time = time.time()
    x, y = read_data(elmo_file_path=args.elmo_file_path)
    end_time = time.time()
    print(f"it costs: {(end_time - start_time) / 60} min")

    test_data = BiasArray(x, y)

    with open(args.pickle_path, "rb") as f:
        score_dict_list = pickle.load(f)
        file_path_list = [score_dict['f1_path'] for score_dict in score_dict_list]
    print(file_path_list)

    y_pred_list = []
    for file_path in file_path_list:
        net = DetectionModel()
        net.load_state_dict(torch.load(file_path))
        if args.device_no > -1:
            net = net.cuda(device=args.device_no)

        data_collator = lambda b: bias_data_collator(b, max_sent=args.max_sent, device_no=args.device_no)
        data_loader = DataLoader(dataset=test_data,
                                 batch_size=args.bs,
                                 collate_fn=data_collator)

        y_prediction_arr = predict_res(net, data_loader)
        # print(y_prediction_arr.shape)  # (645, )
        y_pred_list.append(y_prediction_arr)
    y_arr = np.stack(y_pred_list, axis=1)
    y_arr = np.mean(y_arr, axis=1)

    with open(args.numpy_path, 'wb') as f:
        np.save(f, y_arr)