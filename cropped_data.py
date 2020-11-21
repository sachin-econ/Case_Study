def raw_data():###data_download
    !wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
    !wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar

    ###unzip
    !tar -xf imdb_crop.tar
    !tar -xf wiki_crop.tar###unzip
raw_data()
