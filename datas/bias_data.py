from torch.utils import data
from processing.keras_sequence import pad_sequences
import numpy as np
import torch
import ast


def read_data(elmo_file_path, contain_id=False):
    article_list, label_list = [], []
    if contain_id:
        id_list = []
    with open(elmo_file_path, 'rb') as inf:
        for idx, line in enumerate(inf):
            try:
                gzip_fields = line.decode('utf-8').split('\t')
                gzip_label = 1 if gzip_fields[1] == "true" else 0
                elmo_embd_str = gzip_fields[4].strip()
                elmo_embd_list = ast.literal_eval(elmo_embd_str)
                elmo_embd_array = np.array(elmo_embd_list)
                article_list.append(elmo_embd_array)
                label_list.append(gzip_label)
                if contain_id:
                    gzip_id = gzip_fields[0]
                    id_list.append(gzip_id)
            except Exception as e:
                print(e)
                print(f"jump line: {idx} at file: {elmo_file_path}")
                print("*"*10)

    if contain_id:
        return article_list, label_list, id_list
    else:
        return article_list, label_list


class BiasArray(data.Dataset):

    def __init__(self, article_list, label_list):
        super().__init__()
        self.article_list, self.label_list = article_list, label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        return self.article_list[idx], self.label_list[idx]


class BiasData(data.Dataset):

    def __init__(self, text_path, max_sent, max_tok, elmo_embedder, elmo_bs):
        self.article_list, self.label_list = self.process_data(text_path, max_sent, max_tok, elmo_embedder, elmo_bs)

    def __len__(self):
        return len(self.article_list)

    def __getitem__(self, idx):
        return self.article_list[idx], self.label_list[idx]

    @staticmethod
    def process_data(text_path, max_sent, max_tok, elmo_embedder, elmo_bs):
        article_list = []
        label_list = []
        slice_list = [0]
        # format of text_path: id, label, None, None, content, title
        with open(text_path, "rt", encoding="utf8") as inp:
            for line in inp:
                fields = line.split("\t")
                label = 1 if fields[1] == "true" else 0
                title = fields[5]
                tmp = fields[4]
                tmp = tmp.split(" <splt> ")[:max_sent]
                sents = [title]
                sents.extend(tmp)
                sents = [s.split()[:max_tok] for s in sents]
                article_list.extend(sents)
                label_list.append(label)
                slice_list.append(slice_list[len(slice_list) - 1] + len(sents))

        ret = list(elmo_embedder.embed_sentences(article_list, batch_size=elmo_bs))
        ret_temp_list = []
        for start_pos, end_pos in zip(slice_list[:-1], slice_list[1:]):
            ret_temp = [np.average(x, axis=1) for x in ret[start_pos: end_pos]]
            ret_temp = [np.average(x, axis=0) for x in ret_temp]
            ret_temp_list.append(ret_temp)

        assert len(ret_temp_list) == len(label_list)

        return ret_temp_list, label_list


def bias_data_collator(data_batch, max_sent, device_no):
    article_list = []
    label_list = []
    for data in data_batch:
        article_list.append(data[0])
        label_list.append(data[1])

    padded_ret = pad_sequences(article_list, maxlen=max_sent, dtype='float32')
    padded_ret = torch.tensor(padded_ret).permute(0, 2, 1)
    label_list = torch.tensor(label_list).float()

    if device_no > -1:
        padded_ret = padded_ret.cuda(device=device_no)
        label_list = label_list.cuda(device=device_no)

    return padded_ret, label_list