import ast
import json
import numpy as np
import pandas as pd


n_rows = 1740
df_path = "/home/ruanqin/data/media_bias/processed_data/30000elmo/elmo.tsv"
write_path = "/home/ruanqin/data/media_bias/processed_data/30000elmo/elmo_1740.tsv"
df = pd.read_csv(df_path, sep=",", header=None, nrows=n_rows)
with open(write_path, "w", encoding="utf8") as outp:
    for index, row in df.iterrows():
        elmo_embd_list = ast.literal_eval(row[4])
        elmo_embd_array = np.array(elmo_embd_list)
        outs = [val.tolist() for val in elmo_embd_array]
        print(row[0], row[1], row[2], row[3], json.dumps(outs), sep="\t", file=outp)
