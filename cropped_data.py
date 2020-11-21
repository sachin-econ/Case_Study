def raw_data():
    !wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
    !wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar

    !tar -xf imdb_crop.tar
    !tar -xf wiki_crop.tar

raw_data()
