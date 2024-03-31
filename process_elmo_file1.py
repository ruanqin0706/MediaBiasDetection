import json
import argparse
import numpy as np
from elmo_util import ElmoEmbedder
import gc


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
    parser.add_argument("--num_process", type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    elmo_embedder = ElmoEmbedder(options_file=args.elmo_option_path,
                                 weight_file=args.elmo_weight_path,
                                 cuda_device=args.device_no)

    article_list = []
    slice_list = [0]
    label_list = []
    fields0_list = []
    fields1_list = []
    fields2_list = []
    fields3_list = []
    # format of text_path: id, label, None, None, content, title
    with open(args.text_path, "rt", encoding="utf8") as inp:
        for idx, line in enumerate(inp):
            try:
                fields = line.split("\t")
                label = 1 if fields[1] == "true" else 0
                title = fields[5]
                tmp = fields[4]
                tmp = tmp.split(" <splt> ")[:args.max_sent]
                sents = [title]
                sents.extend(tmp)
                sents = [s.split()[:args.max_tok] for s in sents]
                article_list.extend(sents)
                label_list.append(label)
                slice_list.append(slice_list[len(slice_list) - 1] + len(sents))

                fields0_list.append(fields[0])
                fields1_list.append(fields[1])
                fields2_list.append(fields[2])
                fields3_list.append(fields[3])

            except:
                print(f"skip {idx}")
            else:
                if len(label_list) >= args.num_process:
                    print("start to write")
                    ret = list(elmo_embedder.embed_sentences(article_list, batch_size=args.elmo_bs))
                    ret_temp_list = []
                    for start_pos, end_pos in zip(slice_list[:-1], slice_list[1:]):
                        ret_temp = [np.average(x, axis=1) for x in ret[start_pos: end_pos]]
                        ret_temp = [np.average(x, axis=0) for x in ret_temp]
                        ret_temp_list.append(ret_temp)

                    assert len(ret_temp_list) == len(label_list)

                    with open(args.outfile_path, "a", encoding="utf8") as outp:
                        for fields0, fields1, fields2, fields3, article in zip(fields0_list, fields1_list, fields2_list,
                                                                               fields3_list, ret_temp_list):
                            fields = line.split("\t")
                            outs = [a.tolist() for a in article]
                            print(fields[0], fields[1], fields[2], fields[3], json.dumps(outs), sep="\t", file=outp)
                    print("end to write")

                    del article_list
                    del ret_temp_list
                    del slice_list
                    del label_list
                    del fields0_list
                    del fields1_list
                    del fields2_list
                    del fields3_list
                    gc.collect()

                    article_list = []
                    slice_list = [0]
                    label_list = []
                    fields0_list = []
                    fields1_list = []
                    fields2_list = []
                    fields3_list = []
                    print("reset variables")