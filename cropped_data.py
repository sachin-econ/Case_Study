try:
  try:
    !wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
    !wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
    print('Download Complete')
  except:
    print('Download Failed')
  try:
    !tar - xf wiki_crop.tar
    !tar - xf imdb_crop.tar
    print('Extarction Complete')
  except:
    print('Extarction Failed')
except:
  print('Process Failed')
