python -m processing.xml2line \
        -A /home/ruanqin/data/media_bias/original/articles-training-bypublisher-20181122.xml \
        -T /home/ruanqin/data/media_bias/original/ground-truth-training-bypublisher-20181122.xml \
        -F article_sent,title_sent \
        /home/ruanqin/data/media_bias/processed_data/bypublisher.train.text.tsv > logs/1_bypublisher.log