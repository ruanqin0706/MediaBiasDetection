import json
import argparse

from datas.bias_data import BiasData
from processing.elmo_util import ElmoEmbedder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_path", type=str)
    parser.add_argument("--outfile_path", type=str)
    parser.add_argument("--max_sent", type=int)
    parser.add_argument("--max_tok", type=int)
    parser.add_argument("--elmo_bs", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--elmo_option_path", type=str)
    parser.add_argument("--elmo_weight_path", type=str)
    parser.add_argument("--device_no", type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    elmo_embedder = ElmoEmbedder(options_file=args.elmo_option_path,
                                 weight_file=args.elmo_weight_path,
                                 cuda_device=args.device_no)
    bias_dataset = BiasData(text_path=args.text_path, max_sent=args.max_sent, max_tok=args.max_tok,
                            elmo_embedder=elmo_embedder, elmo_bs=args.elmo_bs)
    with open(args.text_path, "rt", encoding="utf8") as inp:
        line_list = inp.readlines()

    assert len(line_list) == len(bias_dataset.article_list)

    with open(args.outfile_path, "wt", encoding="utf8") as outp:
        for line, article in zip(line_list, bias_dataset.article_list):
            fields = line.split("\t")
            outs = [a.tolist() for a in article]
            print(fields[0], fields[1], fields[2], fields[3], json.dumps(outs), sep="\t", file=outp)
