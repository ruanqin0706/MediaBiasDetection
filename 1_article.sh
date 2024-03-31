python -m processing.xml2line \
        -A /home/ruanqin/data/media_bias/original/articles-training-byarticle-20181122.xml \
        -T /home/ruanqin/data/media_bias/original/ground-truth-training-byarticle-20181122.xml \
        -F article_sent,title_sent \
        /home/ruanqin/data/media_bias/processed_data/byarticle.train.text.tsv > logs/1_article.log