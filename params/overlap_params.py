import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_sent", type=int)
    parser.add_argument("--start_step", type=int)
    parser.add_argument("--total_step", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--auto_bs", type=int)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--elmo_label_path", type=str)
    parser.add_argument("--elmo_auto_path", type=str)
    parser.add_argument("--pickle_path", type=str)
    parser.add_argument("--device_no", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--ratio_label", type=float)
    parser.add_argument("--ratio_auto", type=float)
    parser.add_argument("--n_split", type=int, )
    parser.add_argument("--seed", type=int, )
    return parser.parse_args()