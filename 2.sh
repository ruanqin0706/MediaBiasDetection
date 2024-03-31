python process_elmo_file.py \
        --text_path $1 \
        --outfile_path $2 \
        --max_sent 200 \
        --max_tok 200 \
        --elmo_option_path "/home/ruanqin/data/media_bias/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json" \
        --elmo_weight_path "/home/ruanqin/data/media_bias/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" \
        --elmo_bs 80 \
        --bs 32 \
        --device_no $3 >> $4