import ast
import json
import linecache
import os
import glob
import numpy as np
import pandas as pd


numpy_dir = "/home/ruanqin/project/media_bias"
numpy_file_list = sorted(glob.glob(os.path.join(numpy_dir, "*.npy")))
print(numpy_file_list)


my_data = list()
for idx in range(len(numpy_file_list)):
    with open(numpy_file_list[idx], "rb") as f:
            arr = np.load(f)
            my_data.append(arr)
data_arr = np.concatenate(tuple(my_data))
print(data_arr.shape)


# 抽出来所有正确的1
# 抽出来所有正确的0
df_path = "/home/ruanqin/data/media_bias/processed_data/30000elmo/elmo.tsv"

df = pd.read_csv(df_path, sep=",", header=None, usecols=[1])
assert len(df) == len(data_arr)
df.columns = ["label"]
# print(df.shape)
# print(df.head())

df['prob'] = data_arr.tolist()
df['prob_bool'] = df['prob'] > 0.5


positive_df = df[(df['label'] == True) & (df['prob_bool'] == True)]
positive_df = positive_df.sort_values(by='prob', ascending=False)
# print(positive_df.head(), positive_df.shape, positive_df.index)

negative_df = df[(df['label'] == False) & (df['prob_bool'] == False)]
negative_df = negative_df.sort_values(by='prob', ascending=True)
# print(negative_df.head(), negative_df.shape, negative_df.index)

positive_list = positive_df.index.tolist()
negative_list = negative_df.index.tolist()
print(len(positive_list), len(negative_list))
del df, positive_df, negative_df

for num in [323, 646, 1292]:
    temp_pos_list = positive_list[:num]
    temp_neg_list = negative_list[:num]

    # positive_df = pd.read_csv(df_path, sep=",", header=None, skiprows=lambda x: x not in positive_list[:num])
    negative_df = pd.read_csv(df_path, sep=",", header=None, skiprows=lambda x: x not in negative_list[:num])

    with open(f'/home/ruanqin/data/media_bias/processed_data/30000elmo/proportion_neg_{int(num*2)}.tsv', "a", encoding="utf8") as outp:
        # for index, row in positive_df.iterrows():
        #     elmo_embd_list = ast.literal_eval(row[4])
        #     elmo_embd_array = np.array(elmo_embd_list)
        #     outs = [val.tolist() for val in elmo_embd_array]
        #     print(row[0], row[1], row[2], row[3], json.dumps(outs), sep="\t", file=outp)
        for index, row in negative_df.iterrows():
            elmo_embd_list = ast.literal_eval(row[4])
            elmo_embd_array = np.array(elmo_embd_list)
            outs = [val.tolist() for val in elmo_embd_array]
            print(row[0], row[1], row[2], row[3], json.dumps(outs), sep="\t", file=outp)

# positive_df.head(323).to_csv('/home/ruanqin/data/media_bias/processed_data/30000elmo/proportion_646.csv', mode='a', header=False)
# negative_df.head(323).to_csv('/home/ruanqin/data/media_bias/processed_data/30000elmo/proportion_646.csv', mode='a', header=False)
#
#
# positive_df.head(646).to_csv('/home/ruanqin/data/media_bias/processed_data/30000elmo/proportion_1292.csv', mode='a', header=False)
# negative_df.head(646).to_csv('/home/ruanqin/data/media_bias/processed_data/30000elmo/proportion_1292.csv', mode='a', header=False)
#
# positive_df.head(1292).to_csv('/home/ruanqin/data/media_bias/processed_data/30000elmo/proportion_2584.csv', mode='a', header=False)
# negative_df.head(1292).to_csv('/home/ruanqin/data/media_bias/processed_data/30000elmo/proportion_2584.csv', mode='a', header=False)


# print(df.head())

# import os
# import glob
# import numpy as np
# import pandas as pd
#
#
# skip_line = [
#     [253, 705, 1689, 2649, 2721, 2756],
#     [275, 404, 855, 1386, 1548, 2596, 2913, 2988],
#     [897, 1566, 1675, 2484, 2561, 2978],
#     [499, 988, 1489, 2327, 2331],
#     [345, 617, 838, 873, 934, 1018, 1312, 1946, 2129, 2466, 2543, 2617],
#     [540, 802, 833, 936, 1287, 1410, 1570, 2167, 2232, 2855],
#     [239, 266, 465, 609, 916, 1370, 1547, 2608, 2893],
#     [620, 909, 1174, 1598, 1786, 1900, 2292, 2374, 2626, 2996],
#     [409, 2074, 2138, 2510, 2911],
#     [3, 118, 503, 708, 777, 1389, 1756, 1758, 1933, 2251, 2255, 2272, 2539]
# ]
#
#
# numpy_dir = "/home/ruanqin/project/media_bias"
#
# numpy_file_list = sorted(glob.glob(os.path.join(numpy_dir, "*.npy")))
# print(numpy_file_list)
#
#
# ## check if it does true
# # assert len(skip_line) == len(numpy_file_list)
# # for idx in range(len(skip_line)):
# #     with open(numpy_file_list[idx], "rb") as f:
# #         arr = np.load(f)
# #     print(idx, len(skip_line[idx]), len(arr),  len(skip_line[idx]) + len(arr), numpy_file_list[idx])
#
# elmo_dir = "/home/ruanqin/data/media_bias/processed_data/"
#
# elmo_file_list = sorted(glob.glob(os.path.join(elmo_dir, "xa*")))
# print(elmo_file_list)
#
# write_dir = "/home/ruanqin/data/media_bias/processed_data/30000elmo"
#
# for idx, (elmo_file, numpy_file) in enumerate(zip(elmo_file_list, numpy_file_list)):
#     print(elmo_file, numpy_file)
#
#     df = pd.read_csv(elmo_file, sep="\t", header=None)
#     df = df[~df.index.isin(skip_line[idx])]
#     print(len(df), len(skip_line[idx]), len(df) + len(skip_line[idx]))
#
#     elmo_suffix = elmo_file.split("/")[-1]
#     df.to_csv(os.path.join(write_dir, elmo_suffix), header=None, index=None)
#
#     # bool_list = df[1].tolist()
#     #
#     # with open(numpy_file_list[idx], "rb") as f:
#     #     prob_arr = np.load(f)
#     #
#     # pred_bool_list = (prob_arr > 0.5).tolist()
#     #
#     # print(bool_list)
#     # print("*"*10)
#     # print(pred_bool_list)