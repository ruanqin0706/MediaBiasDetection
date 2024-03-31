python cal_prob.py \
       --elmo_file_path "/home/ruanqin/data/media_bias/processed_data/30000.bypublisher.train.elmo.tsv" \
       --pickle_path "/home/ruanqin/project/models/base/res_info.pkl" \
       --numpy_path $1 \
       --device_no $2 \
       --max_sent 200 \
       --bs 128